from __future__ import annotations

"""
Velocity field v_θ(x_t, t, context) → R^H.

Architecture
------------
  t → sinusoidal embedding (time_emb_dim) → 2-layer MLP → time_emb_dim
  Concatenate [x_t ∈ R^H, t_emb ∈ R^time_emb_dim, context ∈ R^context_dim]
  6-layer MLP, width ``hidden_dim``, SiLU activations
  Linear head → R^H

# DESIGN: sinusoidal time embedding with a small learned MLP head (2 layers)
  rather than learned-only, to encourage well-conditioned gradients at t≈0
  and t≈1 where sinusoidal features are distinct.
"""

import jax.numpy as jnp
import flax.linen as nn


def sinusoidal_time_embedding(t: jnp.ndarray, dim: int = 128) -> jnp.ndarray:
    """
    Sinusoidal embedding for scalar flow-time t.

    Parameters
    ----------
    t   : (B,) — flow time values in [0, 1]
    dim : embedding dimension (must be even)

    Returns
    -------
    jnp.ndarray, shape (B, dim)
    """
    assert dim % 2 == 0, "dim must be even"
    half = dim // 2
    # Log-spaced frequencies: 1 → 10000
    freqs = jnp.exp(
        -jnp.log(10000.0) * jnp.arange(half, dtype=jnp.float32) / (half - 1)
    )  # (half,)
    args = t[:, None] * freqs[None, :]   # (B, half)
    return jnp.concatenate([jnp.sin(args), jnp.cos(args)], axis=-1)  # (B, dim)


class VelocityField(nn.Module):
    """
    Conditional velocity field for flow matching.

    Parameters
    ----------
    horizon      : path length H (output dimension)
    hidden_dim   : MLP width (default 256)
    num_layers   : MLP depth (default 6)
    context_dim  : context vector dimension (must match ContextEncoder.model_dim)
    time_emb_dim : sinusoidal time embedding dimension (default 128)

    Inputs
    ------
    x_t      : (B, H)          — noisy path at flow time t
    t        : (B,)            — flow time in [0, 1]
    context  : (B, context_dim) — encoded context from ContextEncoder

    Output
    ------
    (B, H) — predicted conditional vector field
    """

    horizon: int
    hidden_dim: int = 256
    num_layers: int = 6
    context_dim: int = 128
    time_emb_dim: int = 128

    @nn.compact
    def __call__(
        self,
        x_t: jnp.ndarray,
        t: jnp.ndarray,
        context: jnp.ndarray,
        deterministic: bool = True,  # kept for API symmetry with encoder
    ) -> jnp.ndarray:
        # Time embedding: sinusoidal → small MLP
        t_emb = sinusoidal_time_embedding(t, self.time_emb_dim)   # (B, time_emb_dim)
        t_emb = nn.Dense(self.time_emb_dim)(t_emb)
        t_emb = nn.silu(t_emb)
        t_emb = nn.Dense(self.time_emb_dim)(t_emb)
        t_emb = nn.silu(t_emb)

        # Concatenate all inputs
        h = jnp.concatenate([x_t, t_emb, context], axis=-1)
        # h: (B, H + time_emb_dim + context_dim)

        for _ in range(self.num_layers):
            h = nn.Dense(self.hidden_dim)(h)
            h = nn.silu(h)

        return nn.Dense(self.horizon)(h)  # (B, H)
