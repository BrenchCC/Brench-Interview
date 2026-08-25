import os
import sys
import math
import logging

logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.query_proj = nn.Linear(hidden_dim, hidden_dim, bias = False)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim, bias = False)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim, bias = False)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim, bias = False)

        self.attention_dropout = nn.Dropout(0.1)

    def forward(self, x_query, x_key_value, mask=None):
        """
        x_query: (batch, seq_query, hidden_dim)
        x_key_value: (batch, seq_key_value, hidden_dim)
        mask: (seq_query, seq_key_value), True=mask, broadcast to batch dimension
        """
        # Compute QKV
        Q = self.query_proj(x_query)   # Q: (batch, seq_query, hidden_dim)
        K = self.key_proj(x_key_value)  # K: (batch, seq_key_value, hidden_dim)
        V = self.value_proj(x_key_value)  # V: (batch, seq_key_value, hidden_dim)
        
        # Compute attention scores and apply mask
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.hidden_dim)  # scores: (batch, seq_query, seq_key_value)
        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))
        
        # Compute attention weights and output
        attn_weights = F.softmax(scores, dim = -1)  # attn_weights: (batch, seq_query, seq_key_value)
        attn_weights = self.attention_dropout(attn_weights)
        attention_output = torch.matmul(attn_weights, V)  # out: (batch, seq_query, hidden_dim)
        output = self.output_proj(attention_output)  # output: (batch, seq_query, hidden_dim)
        
        return output  # return: (batch, seq_len, hidden_dim)