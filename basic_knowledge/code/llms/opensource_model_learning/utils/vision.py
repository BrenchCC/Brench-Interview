# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
"""Vision utility functions extracted from the official Transformers source."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def get_vision_cu_seqlens(
    grid_thw: torch.Tensor,
    merge_temporal: bool = False,
    kwargs: dict[str, Any] | None = None,
) -> torch.Tensor:
    """Get cumulative sequence lengths from vision grid info.

    Parameters:
        grid_thw: ``(num_images_or_videos, 3)`` temporal, height, width entries.
        merge_temporal: Whether all frames of each clip share an attention segment.
        kwargs: Optional caller values containing precomputed ``cu_seqlens``.
    """
    if kwargs is not None and (cu_seqlens := kwargs.pop("cu_seqlens", None)) is not None:
        return cu_seqlens
    dtype = grid_thw.dtype if torch.jit.is_tracing() else torch.int32
    if merge_temporal:
        seqlens = grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]
    else:
        seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0])
    return F.pad(seqlens.cumsum(dim = 0, dtype = dtype), (1, 0), value = 0)


def get_vision_attention_seqlens(
    grid_thw: torch.Tensor,
    config: Any,
    merge_temporal: bool = False,
    kwargs: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, int | None]:
    """Get cumulative and maximum lengths for packed vision attention.

    Parameters:
        grid_thw: Packed temporal-height-width grid values.
        config: Vision configuration whose optional max sequence setting is read.
        merge_temporal: Whether attention spans complete clips.
        kwargs: Optional caller values containing precomputed tensors.
    """
    cu_seqlens = get_vision_cu_seqlens(grid_thw, merge_temporal = merge_temporal, kwargs = kwargs)
    if kwargs is not None and (max_seqlen := kwargs.pop("max_seqlen", None)) is not None:
        return cu_seqlens, int(max_seqlen)
    lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    return cu_seqlens, int(lengths.max().item()) if len(lengths) else None


def get_vision_position_ids(
    grid_thw: torch.Tensor,
    spatial_merge_size: int | torch.Tensor,
    include_temporal: bool = False,
    kwargs: dict[str, Any] | None = None,
) -> torch.Tensor:
    """Get position IDs in the official spatial-merge block ordering.

    Parameters:
        grid_thw: Packed temporal-height-width grid values.
        spatial_merge_size: Scalar or per-image spatial merge factor.
        include_temporal: Whether temporal rotary indices are included.
        kwargs: Optional caller values containing precomputed ``position_ids``.
    """
    if kwargs is not None and (position_ids := kwargs.pop("position_ids", None)) is not None:
        return position_ids

    device = grid_thw.device
    if isinstance(spatial_merge_size, int):
        spatial_merge_size = torch.tensor([spatial_merge_size], device = device).expand(len(grid_thw))

    position_ids = []
    for (temporal, height, width), merge_size in zip(grid_thw.tolist(), spatial_merge_size.tolist()):
        hpos_ids, wpos_ids = torch.meshgrid(
            torch.arange(height, device = device),
            torch.arange(width, device = device),
            indexing = "ij",
        )
        block_shape = (height // merge_size, merge_size, width // merge_size, merge_size)
        hpos_ids = hpos_ids.reshape(block_shape).transpose(1, 2).flatten()
        wpos_ids = wpos_ids.reshape(block_shape).transpose(1, 2).flatten()
        if include_temporal:
            tpos_ids = torch.arange(temporal, device = device).repeat_interleave(height * width)
            position_ids.append(torch.stack([tpos_ids, hpos_ids.repeat(temporal), wpos_ids.repeat(temporal)], dim = -1))
        else:
            position_ids.append(torch.stack([hpos_ids, wpos_ids], dim = -1).repeat(temporal, 1))
    return torch.cat(position_ids, dim = 0)


def _interpolation_axis_taps_weights(
    index: torch.Tensor,
    size: torch.Tensor,
    side: int,
    mode: str,
    align_corners: bool,
    padding: str = "border",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute source-table taps and weights for one interpolation axis.

    Parameters:
        index: Target coordinate indices.
        size: Per-target source-grid sizes.
        side: Side length of the learned square source table.
        mode: ``bilinear`` or ``bicubic`` interpolation mode.
        align_corners: Whether interpolation aligns grid endpoints.
        padding: ``border`` or ``zeros`` out-of-range behavior.
    """
    index = index.to(torch.float32)
    if align_corners:
        # Closed form of `torch.linspace(0, side-1, size)[index]` — endpoints map to 0 and side-1.
        # `clamp(min=1)` avoids a divide-by-zero when size == 1 (index is 0, so src is 0 too).
        source = index * (side - 1) / torch.clamp(size - 1, min = 1)
    else:
        source = (index + 0.5) * side / size - 0.5  # half-pixel centres (align_corners=False)
    floor = torch.floor(source)
    if mode == "bilinear":
        offsets = torch.arange(0, 2, device = index.device)  # floor, floor+1
    elif mode == "bicubic":
        offsets = torch.arange(-1, 3, device = index.device)  # floor-1 .. floor+2
    else:
        raise ValueError(f"Unsupported interpolation mode {mode!r} (expected 'bilinear' or 'bicubic').")
    raw_taps = floor.long()[:, None] + offsets
    taps = raw_taps.clamp(0, side - 1)
    distance = (source[:, None] - floor[:, None] - offsets).abs()
    if mode == "bilinear":
        weights = (1 - distance).clamp(min = 0)  # linear hat kernel
    else:
        alpha = -0.75
        near = ((alpha + 2) * distance - (alpha + 3)) * distance * distance + 1
        far = ((alpha * distance - 5 * alpha) * distance + 8 * alpha) * distance - 4 * alpha
        weights = torch.where(distance <= 1, near, far)
    if padding == "zeros":
        # Out-of-range taps were clamped to the border above; zero their weight so they contribute nothing.
        weights = weights * ((raw_taps >= 0) & (raw_taps <= side - 1))
    return taps, weights


def get_vision_interpolation_indices_and_weights(
    grid_thw: torch.Tensor,
    num_grid_per_side: int,
    mode: str = "bilinear",
    align_corners: bool = False,
    spatial_merge_size: int = 1,
    padding: str = "border",
    kwargs: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resample a learned square position table using the official gather formulation.

    Parameters:
        grid_thw: Packed temporal-height-width grid values.
        num_grid_per_side: Side length of the learned position table.
        mode: ``bilinear`` or ``bicubic`` interpolation mode.
        align_corners: Whether interpolation aligns grid endpoints.
        spatial_merge_size: Block order used by vision patch merging.
        padding: ``border`` or ``zeros`` out-of-range behavior.
        kwargs: Optional caller values containing precomputed interpolation tensors.
    """
    if kwargs is not None:
        indices = kwargs.pop("interp_indices", None)
        weights = kwargs.pop("interp_weights", None)
        if indices is not None and weights is not None:
            return indices, weights

    side = num_grid_per_side
    merge = spatial_merge_size
    device = grid_thw.device
    counts = grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]
    heights = torch.repeat_interleave(grid_thw[:, 1], counts)
    widths = torch.repeat_interleave(grid_thw[:, 2], counts)
    starts = torch.repeat_interleave(F.pad(counts.cumsum(0)[:-1], (1, 0)), counts)
    # Position within a single frame's flat patch sequence (0 .. h*w-1), repeating across the t frames.
    within = (torch.arange(counts.sum(), device = device) - starts) % (heights * widths)
    # Decode `within` into (row, col): raster order when merge == 1, else spatial-merge-block order.
    blocks_w = widths // merge
    in_col = within % merge
    in_row = (within // merge) % merge
    block_col = (within // (merge * merge)) % blocks_w
    block_row = within // (merge * merge * blocks_w)
    row = block_row * merge + in_row
    col = block_col * merge + in_col
    h_taps, h_weights = _interpolation_axis_taps_weights(row, heights, side, mode, align_corners, padding)
    w_taps, w_weights = _interpolation_axis_taps_weights(col, widths, side, mode, align_corners, padding)
    num_taps = h_taps.shape[1]
    indices = (h_taps[:, :, None] * side + w_taps[:, None, :]).reshape(-1, num_taps * num_taps)
    weights = (h_weights[:, :, None] * w_weights[:, None, :]).reshape(-1, num_taps * num_taps)
    return indices, weights
