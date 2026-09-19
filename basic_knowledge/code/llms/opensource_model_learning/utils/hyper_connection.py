# Copyright 2026 the HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""GLM5-Next Manifold-Constrained Hyper-Connection components."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


class UnweightedRMSNorm(nn.Module):
    """Source GLM5-Next RMS normalization without learnable parameters.

    Parameters:
        eps: Numerical stability constant.
    """

    def __init__(self, eps: float = 1.0e-6) -> None:
        """Store the source normalization epsilon.

        Parameters:
            eps: Numerical stability constant.
        """
        super().__init__()
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply source RMS rescaling without a learned scale.

        Parameters:
            hidden_states: Tensor to normalize.
        """
        return hidden_states * torch.rsqrt(hidden_states.float().square().mean(-1, keepdim = True) + self.eps).to(hidden_states.dtype)


class HyperConnection(nn.Module):
    r"""Manifold-Constrained Hyper-Connections (mHC) source implementation.

    Parameters:
        config: GLM5-Next text configuration exposing Hyper-Connection fields.
    """

    def __init__(self, config: Any) -> None:
        """Create the source collapse, expand, and stream-mixing parameters.

        Parameters:
            config: GLM5-Next text configuration.
        """
        super().__init__()
        self.hc_mult = config.hc_mult
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        self.input_norm = UnweightedRMSNorm(eps = config.rms_norm_eps)
        mix = (2 + self.hc_mult) * self.hc_mult
        self.fn = nn.Parameter(torch.empty(mix, self.hc_mult * config.hidden_size))
        self.base = nn.Parameter(torch.empty(mix))
        # 3 = number of outputs from the mHC mapping: `pre` (input projection
        # weights), `post` (sublayer output projection weights), `comb` (the
        # H×H residual combine matrix that gets Sinkhorn-projected onto the
        # doubly-stochastic manifold). Each output gets its own learned scale.
        self.scale = nn.Parameter(torch.empty(3))

    def forward(self, hidden_streams: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""Compute `pre`, `post`, `comb` from the mHC mapping (paper §2.2 eq. 8).

        Parameters:
            hidden_streams: Parallel residual streams shaped ``[B, S, H, D]``.
        """
        hc = self.hc_mult
        flat = self.input_norm(hidden_streams.flatten(start_dim = 2).float())
        pre_weight, post_weight, comb_weight = F.linear(flat, self.fn.float()).split([hc, hc, hc * hc], dim = -1)
        pre_bias, post_bias, comb_bias = self.base.split([hc, hc, hc * hc])
        pre_scale, post_scale, comb_scale = self.scale.unbind(0)
        pre = torch.sigmoid(pre_weight * pre_scale + pre_bias) + self.hc_eps
        post = 2 * torch.sigmoid(post_weight * post_scale + post_bias)
        comb_logits = comb_weight.view(*comb_weight.shape[:-1], hc, hc) * comb_scale + comb_bias.view(hc, hc)
        comb = torch.softmax(comb_logits, dim = -1) + self.hc_eps
        comb = comb / (comb.sum(dim = -2, keepdim = True) + self.hc_eps)
        for _ in range(self.hc_sinkhorn_iters - 1):
            comb = comb / (comb.sum(dim = -1, keepdim = True) + self.hc_eps)
            comb = comb / (comb.sum(dim = -2, keepdim = True) + self.hc_eps)
        # Collapse the `hc_mult` parallel streams down to a single sequence using
        # the `pre` weights: one weighted sum across the stream axis, ready for
        # the sublayer (attn / MLP).
        collapsed = (pre.unsqueeze(-1) * hidden_streams).sum(dim = 2).to(hidden_streams.dtype)
        return post, comb, collapsed


class HyperHead(nn.Module):
    """Final GLM5-Next HC-stream collapse, an unweighted mean.

    Parameters:
        None.
    """

    def forward(self, hidden_streams: torch.Tensor) -> torch.Tensor:
        """Average source residual streams.

        Parameters:
            hidden_streams: Parallel residual streams shaped ``[B, S, H, D]``.
        """
        return hidden_streams.mean(dim = 2)


class ForgetGate(nn.Module):
    """Official GLM5-Next KDA decay gate.

    Parameters:
        config: GLM5-Next text configuration exposing linear-attention fields.
    """

    def __init__(self, config: Any) -> None:
        """Create low-rank forget-gate projections and decay parameters.

        Parameters:
            config: GLM5-Next text configuration.
        """
        super().__init__()
        self.head_dim = config.linear_head_dim
        self.num_heads = config.linear_num_heads
        self.qkv_dim = self.head_dim * self.num_heads
        self.f_a_proj = nn.Linear(config.hidden_size, self.head_dim, bias = False)
        self.f_b_proj = nn.Linear(self.head_dim, self.qkv_dim, bias = False)
        self.dt_bias = nn.Parameter(torch.empty(self.qkv_dim))
        self.A_log = nn.Parameter(torch.empty(self.num_heads))
        self.safe_gate_lower_bound = config.linear_lower_bound

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute source bounded or softplus KDA log-decays.

        Parameters:
            hidden_states: Input text hidden states.
        """
        hidden_shape = (*hidden_states.shape[:2], -1, self.head_dim)
        forget_gate = self.f_b_proj(self.f_a_proj(hidden_states))
        gate = (forget_gate.float() + self.dt_bias.float().view(1, 1, -1)).view(hidden_shape)
        decay_rate = torch.exp(self.A_log.float().view(1, 1, self.num_heads, 1))
        # Safe lower bound decay
        if self.safe_gate_lower_bound is not None:
            return self.safe_gate_lower_bound * torch.sigmoid(decay_rate * gate)
        # Softplus "log(1 + exp(x))" with uper bound restraint to avoid overflows
        # NOTE: Softplus for larger values (e.g. 20+), Softplus(x) == x
        gate_softplus = torch.where(gate > 20.0, gate, torch.log(1.0 + torch.exp(gate)))
        return -decay_rate * gate_softplus
