from __future__ import annotations

"""
Context encoder: Transformer over the (L, F) feature matrix.

Architecture
------------
  Linear projection: F → model_dim
  Sinusoidal positional encoding added to projected tokens
  4 × TransformerBlock (self-attention + FFN, pre-norm)
  Mean-pool over L tokens → context vector of dim model_dim

# DESIGN: pre-norm (LayerNorm before sub-layer) chosen over post-norm for
  training stability; this is the standard in modern Transformers.

# DESIGN: sinusoidal positional encodings (fixed, not learned) to stay
  parameter-free and length-generalisable, matching the NWP encoder convention.
"""

import math
from typing import Any

import jax
import jax.numpy as jnp
import flax.linen as nn


def sinusoidal_positional_encoding(seq_len: int, model_dim: int) -> jnp.ndarray:
    """
    Fixed sinusoidal positional encoding, shape (seq_len, model_dim).

    Follows Vaswani et al. (2017): PE[pos, 2i] = sin(pos/10000^{2i/d}),
    PE[pos, 2i+1] = cos(pos/10000^{2i/d}).
    """
    positions = jnp.arange(seq_len)[:, None]        # (L, 1)
    dims = jnp.arange(model_dim // 2)[None, :]      # (1, d/2)
    freqs = jnp.exp(-jnp.log(10000.0) * 2 * dims / model_dim)
    args = positions * freqs                          # (L, d/2)
    enc = jnp.concatenate([jnp.sin(args), jnp.cos(args)], axis=-1)  # (L, d)
    if model_dim % 2 == 1:
        enc = jnp.concatenate([enc, jnp.zeros((seq_len, 1))], axis=-1)
    return enc


class TransformerBlock(nn.Module):
    """Single pre-norm Transformer block (self-attention + MLP FFN)."""

    model_dim: int
    num_heads: int
    dropout_rate: float

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        # Pre-norm self-attention
        h = nn.LayerNorm()(x)
        h = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate,
        )(h, deterministic=deterministic)
        h = nn.Dropout(rate=self.dropout_rate)(h, deterministic=deterministic)
        x = x + h

        # Pre-norm FFN
        h = nn.LayerNorm()(x)
        h = nn.Dense(self.model_dim * 4)(h)
        h = nn.gelu(h)
        h = nn.Dense(self.model_dim)(h)
        h = nn.Dropout(rate=self.dropout_rate)(h, deterministic=deterministic)
        x = x + h
        return x


class ContextEncoder(nn.Module):
    """
    Transformer encoder over the (L, F) feature matrix.

    Parameters
    ----------
    model_dim   : width of the Transformer (default 128)
    num_heads   : attention heads (default 4)
    num_layers  : number of TransformerBlocks (default 4)
    dropout_rate: dropout probability (default 0.1)

    Input  : (B, L, F)
    Output : (B, model_dim)  — mean-pooled context vector
    """

    model_dim: int = 128
    num_heads: int = 4
    num_layers: int = 4
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        # x: (B, L, F)
        B, L, F = x.shape

        # Linear projection to model_dim
        x = nn.Dense(self.model_dim)(x)  # (B, L, model_dim)

        # Add sinusoidal positional encoding (broadcast over batch)
        pos_enc = sinusoidal_positional_encoding(L, self.model_dim)  # (L, model_dim)
        x = x + pos_enc[None, :, :]  # (B, L, model_dim)

        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)

        for _ in range(self.num_layers):
            x = TransformerBlock(
                model_dim=self.model_dim,
                num_heads=self.num_heads,
                dropout_rate=self.dropout_rate,
            )(x, deterministic=deterministic)

        # Final LayerNorm then mean-pool over the sequence dimension
        x = nn.LayerNorm()(x)
        return jnp.mean(x, axis=1)  # (B, model_dim)
