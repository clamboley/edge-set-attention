"""ESA architecture model.

Pytorch implementation of the paper:
  - An end-to-end attention-based approach for learning on graphs
  - https://arxiv.org/abs/2402.10793
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn


class MLP(nn.Module):
    """Simple Multi Layer Perceptron module."""

    def __init__(self, config: ESAConfig) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply MLP to input tensors."""
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return self.dropout(x)


class SelfAttention(nn.Module):
    """Basic self attention module using Flash Attention."""

    def __init__(self, config: ESAConfig) -> None:
        super().__init__()
        if config.n_embd % config.n_head != 0:
            msg = f"n_embd must be divisible by n_head. Found {config.n_embd} and {config.n_head}."
            raise ValueError(msg)

        # Key, query, value projections for all heads, in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)

        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Apply SDPA (Scaled Dot-Product Attention).

        Args:
            x (torch.Tensor): Features (batch_size, context, n_embd).
            mask (torch.Tensor | None, optional): Attention mask.

        Returns:
            torch.Tensor: Output projection.
        """
        bsz, seq_length, emb_dim = x.size()

        # Calculate query, key, values for all heads in batch and move head forward to batch dim
        # q, k, and v shapes: (bsz, config.n_head, seq_length, head_size)
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(bsz, seq_length, self.n_head, emb_dim // self.n_head).transpose(1, 2)
        q = q.view(bsz, seq_length, self.n_head, emb_dim // self.n_head).transpose(1, 2)
        v = v.view(bsz, seq_length, self.n_head, emb_dim // self.n_head).transpose(1, 2)

        # Make the mask broadcastable
        if mask is not None:
            mask = mask.unsqueeze(1)  # (bsz, 1, L, L)

        # Self-attention: (bsz, nh, L, hs) x (bsz, nh, hs, L) -> (bsz, nh, L, L)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.dropout)
        y = y.transpose(1, 2).contiguous().view(bsz, seq_length, emb_dim)

        # output projection
        return self.resid_dropout(self.c_proj(y))


class AttentionBlock(nn.Module):
    """Attention Block ('AB' from the ESA paper)."""

    def __init__(self, config: ESAConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = SelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Apply Self or Masked AB to edge features and mask.

        Args:
            x (torch.Tensor): Edge features (batch_size, num_edges, n_embd)
            mask (torch.Tensor | None, optional): Edge adjacency matrix

        Returns:
            torch.Tensor: Block output projection.
        """
        x = self.ln_1(x)
        x = x + self.attn(x, mask)
        return x + self.mlp(self.ln_2(x))


class PoolingModule(nn.Module):
    """Pooling by Multi-Head Attention (PMA) module.

    For aggregating N edge-level representations into K output representations.
    """

    def __init__(self, config: ESAConfig) -> None:
        super().__init__()
        self.seed_vectors = nn.Parameter(torch.randn(config.seeds, config.n_embd))

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=config.n_embd,
            num_heads=config.n_head,
            dropout=config.dropout,
            batch_first=True,
        )
        self.sab_layers = nn.ModuleList([AttentionBlock(config) for _ in range(config.pool_layers)])
        self.layer_norm = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Apply PMA to create a graph representation from encoder output.

        Args:
            z (torch.Tensor): Tensor of shape (batch_size, num_edges, n_embd)
                containing edge-level encoder representations.

        Returns:
            torch.Tensor: Output representations of shape (batch_size, n_seeds, n_embd)
        """
        batch_size = z.size(0)

        # Expand seed vectors to batch size
        seed_queries = self.seed_vectors.expand(batch_size, -1, -1)

        # Cross-attention
        out, _ = self.cross_attention(seed_queries, z, z)
        out = self.layer_norm(out)
        out = out + self.mlp(out)

        # Applying 'p' SAB layers (p not given in paper)
        for sab in self.sab_layers:
            out = sab(out)

        return out


@dataclass
class ESAConfig:
    """Config parameters for ESA architecture."""

    layers: str = "MMSMMSPS"
    n_head: int = 8
    n_embd: int = 256
    dropout: float = 0.0
    bias: bool = True
    seeds: int = 32
    pool_layers: int = 2


class ESA(nn.Module):
    """ESA model.

    Implementation of the Edge Set Attention architecture.
    The model is composed of a succession of Self and Masked Attention Blocks as an encoder,
    followed by a Pooling by Multi Head Attention module. Some Self Attention Blocks can be
    added after the pooling module.
    """

    def __init__(self, config: ESAConfig) -> None:
        super().__init__()
        if not bool(re.match(r"^[SM]*P[S]*$", config.layers)):
            msg = 'Unsupported layers configuration. (Template: "^[SM]*P[S]*$")'
            raise ValueError(msg)

        self.config = config
        self.enc_blocks, self.out_blocks = config.layers.split("P")

        self.transformer = nn.ModuleDict(
            {
                "attn_block": nn.ModuleList([AttentionBlock(config) for _ in self.enc_blocks]),
                "pool_mod": PoolingModule(config),
                "out_attn_block": nn.Sequential(*[AttentionBlock(config) for _ in self.out_blocks]),
            },
        )

        # Apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * len(config.layers)))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Apply ESA model to a graph."""
        for layer, block in zip(self.enc_blocks, self.transformer.attn_block, strict=True):
            if layer == "S":
                x = block(x)
            elif layer == "M":
                x = block(x, mask)

        x = self.transformer.pool_mod(x)
        x = self.transformer.out_attn_block(x)
        return x.mean(dim=1)
