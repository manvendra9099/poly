from __future__ import annotations

"""
ODE samplers for the flow-matching model.

All solvers:
  - Use jax.lax.fori_loop for the time-step iteration (no Python loops at
    trace time, compatible with jit).
  - Accept an already-encoded context vector to avoid re-encoding per sample.
  - Are invoked via jax.vmap over the N initial noise samples.

Available solvers
-----------------
  heun  : Heun's method (2nd-order Runge-Kutta), recommended default.
  euler : Explicit Euler (1st order), faster but less accurate.

Usage example::

    context = model.apply(params, features[None], method=model.encode)
    ensemble = generate_ensemble(
        model, params, context[0], key, horizon=60,
        n_samples=500, n_steps=50, solver="heun"
    )  # shape (500, 60)
"""

import functools
import logging
from typing import Literal

import jax
import jax.numpy as jnp

logger = logging.getLogger(__name__)

Solver = Literal["heun", "euler"]


def _vel(model, params, x: jnp.ndarray, t: jnp.ndarray, context: jnp.ndarray) -> jnp.ndarray:
    """
    Evaluate the velocity field for a single (unbatched) sample.

    x       : (H,)
    t       : scalar
    context : (context_dim,)
    """
    v = model.apply(
        params,
        x[None],        # (1, H)
        t[None],        # (1,)
        context[None],  # (1, context_dim)
        method=model.predict_velocity,
        deterministic=True,
    )
    return v[0]  # (H,)


def _euler_step(model, params, context, n_steps: int, x0: jnp.ndarray) -> jnp.ndarray:
    """Euler solver for a single trajectory starting at x0: (H,)."""
    dt = 1.0 / n_steps

    def body(i, x):
        t = jnp.array(i * dt, dtype=jnp.float32)
        return x + dt * _vel(model, params, x, t, context)

    return jax.lax.fori_loop(0, n_steps, body, x0)


def _heun_step(model, params, context, n_steps: int, x0: jnp.ndarray) -> jnp.ndarray:
    """Heun (RK2) solver for a single trajectory starting at x0: (H,)."""
    dt = 1.0 / n_steps

    def body(i, x):
        t = jnp.array(i * dt, dtype=jnp.float32)
        t_next = jnp.array((i + 1) * dt, dtype=jnp.float32)
        k1 = _vel(model, params, x, t, context)
        x_pred = x + dt * k1
        k2 = _vel(model, params, x_pred, t_next, context)
        return x + dt * (k1 + k2) * 0.5

    return jax.lax.fori_loop(0, n_steps, body, x0)


def generate_ensemble(
    model,
    params,
    context: jnp.ndarray,
    key: jax.Array,
    horizon: int,
    n_samples: int = 500,
    n_steps: int = 50,
    solver: Solver = "heun",
) -> jnp.ndarray:
    """
    Generate an ensemble of path samples from the flow model.

    The ODE solve is fully traced through jax.lax.fori_loop and vmapped
    over N initial noise draws; wrap this call with jax.jit at the call site
    (static_argnames=["model", "horizon", "n_samples", "n_steps", "solver"]).

    Parameters
    ----------
    model    : BTCFlowModel instance (used as static/traced arg)
    params   : model parameters pytree
    context  : (context_dim,) — encoded context for ONE forecast issue time
    key      : PRNGKey
    horizon  : H — path length (must match what model was trained with)
    n_samples: ensemble size N
    n_steps  : ODE integrator steps
    solver   : "heun" | "euler"

    Returns
    -------
    jnp.ndarray, shape (n_samples, H)
    """
    keys = jax.random.split(key, n_samples)                          # (N,)
    x0s = jax.vmap(lambda k: jax.random.normal(k, (horizon,)))(keys)  # (N, H)

    if solver == "heun":
        solve = functools.partial(_heun_step, model, params, context, n_steps)
    elif solver == "euler":
        solve = functools.partial(_euler_step, model, params, context, n_steps)
    else:
        raise ValueError(f"Unknown solver: {solver!r}. Choose 'heun' or 'euler'.")

    return jax.vmap(solve)(x0s)  # (N, H)


def generate_ensemble_batched(
    model,
    params,
    contexts: jnp.ndarray,
    key: jax.Array,
    horizon: int,
    n_samples: int = 500,
    n_steps: int = 50,
    solver: Solver = "heun",
) -> jnp.ndarray:
    """
    generate_ensemble applied to a batch of context vectors.

    Parameters
    ----------
    contexts : (B, context_dim)
    key      : PRNGKey — split per context item

    Returns
    -------
    jnp.ndarray, shape (B, n_samples, H)
    """
    B = contexts.shape[0]
    keys = jax.random.split(key, B)

    def single(ctx, k):
        return generate_ensemble(model, params, ctx, k, horizon, n_samples, n_steps, solver)

    return jax.vmap(single)(contexts, keys)  # (B, N, H)
