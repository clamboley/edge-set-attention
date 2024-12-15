"""ESA architecture model.

Pytorch implementation of the paper:
  - An end-to-end attention-based approach for learning on graphs
  - https://arxiv.org/abs/2402.10793
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn


class MLP(nn.Module):
    """Multilayer perceptron module."""

    def __init__(self, config: ESAConfig) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply MLP to inputs."""
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return self.dropout(x)


class SelfAttention(nn.Module):
    """Self Attention.

    A simple causal self attention module that uses flash attention.
    """

    def __init__(self, config: ESAConfig) -> None:
        super().__init__()
        if config.n_embd % config.n_head == 0:
            msg = (
                "n_embd must be divisible by n_head."
                f" Found {config.n_embd} and {config.n_head}."
            )
            raise ValueError(msg)

        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)

        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the self attention block to inputs."""
        bsz, seq_length, emb_dim = x.size()

        # Calculate query, key, values for all heads in batch and move head forward to batch dim
        # q, k, and v shapes: (bsz, config.n_head, seq_length, head_size)
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(bsz, seq_length, self.n_head, emb_dim // self.n_head).transpose(1, 2)
        q = q.view(bsz, seq_length, self.n_head, emb_dim // self.n_head).transpose(1, 2)
        v = v.view(bsz, seq_length, self.n_head, emb_dim // self.n_head).transpose(1, 2)

        # Causal self-attention
        # Self-attend: (bsz, nh, L, hs) x (bsz, nh, hs, L) -> (bsz, nh, L, L)
        # Then re-assemble all head outputs side by side
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(bsz, seq_length, emb_dim)

        # output projection
        return self.resid_dropout(self.c_proj(y))


class MaskedAttention(nn.Module):
    """Masked Attention."""

    def __init__(self, config: ESAConfig) -> None:
        super().__init__()
        if config.n_embd % config.n_head == 0:
            msg = (
                "n_embd must be divisible by n_head."
                f" Found {config.n_embd} and {config.n_head}."
            )
            raise ValueError(msg)

        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)

        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

    def forward(self, x: torch.Tensor, edge_adjacency_matrix: torch.Tensor) -> torch.Tensor:
        """Apply the masked attention to inputs."""
        bsz, seq_length, emb_dim = x.size()
        mask = self._edge_adjacency_matrix_to_mask(edge_adjacency_matrix)

        # Calculate query, key, values for all heads in batch and move head forward to batch dim
        # q, k, and v shapes: (bsz, config.n_head, seq_length, head_size)
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(bsz, seq_length, self.n_head, emb_dim // self.n_head).transpose(1, 2)
        q = q.view(bsz, seq_length, self.n_head, emb_dim // self.n_head).transpose(1, 2)
        v = v.view(bsz, seq_length, self.n_head, emb_dim // self.n_head).transpose(1, 2)

        # Manual implementation of attention with an additive mask for connected edges
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att + mask  # Additive mask
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v

        y = y.transpose(1, 2).contiguous().view(bsz, seq_length, emb_dim)

        # output projection
        return self.resid_dropout(self.c_proj(y))

    def _edge_adjacency_matrix_to_mask(self, matrix: torch.Tensor) -> torch.Tensor:
        """Tranform an edge adjacency matrix into a usable mask."""
        return matrix


class SAB(nn.Module):
    """Self Attention Block.

    Like a Masked Attention Block, but without edge mask.
    Not having a mask allows to use flash attention.
    """

    def __init__(self, config: ESAConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = SelfAttention(
            config.n_embd,
            config.n_head,
            bias=config.bias,
            dropout=config.dropout,
        )
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config.n_embd, bias=config.bias, dropout=config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SAB to the graph edges."""
        x = self.ln_1(x)
        x = x + self.attn(x)
        return x + self.mlp(self.ln_2(x))


class MAB(nn.Module):
    """Masked Attention Block.

    Self attention on edges with an additive edge mask.
    """

    def __init__(self, config: ESAConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = MaskedAttention(
            config.n_embd,
            config.n_head,
            bias=config.bias,
            dropout=config.dropout,
        )
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config.n_embd, bias=config.bias, dropout=config.dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Apply MAB to the graph edges and mask."""
        x = self.ln_1(x)
        x = x + self.attn(x, mask)
        return x + self.mlp(self.ln_2(x))

class PoolingModule(nn.Module):
    def __init__(self, config: ESAConfig) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Apply MAB to the graph edges and mask."""


@dataclass
class ESAConfig:
    """Config parameters for ESA architecture."""

    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True
