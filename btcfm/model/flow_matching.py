from __future__ import annotations

"""
Flow-matching combined model, loss, EMA, and TrainState.

Mathematical objective (rectified / linear interpolant):
  x_t = (1 - t) * x_0 + t * x_1,   t ~ U(0, 1)
  u_t(x_t | x_0, x_1) = x_1 - x_0          (conditional vector field)

  L(θ) = E_{t, x_0, (x_1, c)} || v_θ(x_t, t, c) - (x_1 - x_0) ||²

EMA:
  ema_θ ← decay * ema_θ + (1 - decay) * θ     after every gradient step

EMA parameters are carried alongside TrainState as a separate pytree (not
embedded in TrainState), which keeps the Flax API usage clean. At inference,
always sample from ema_params.
"""

import functools
import logging
import pickle
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training import train_state

from btcfm.model.encoder import ContextEncoder
from btcfm.model.velocity import VelocityField

logger = logging.getLogger(__name__)


class BTCFlowModel(nn.Module):
    encoder: ContextEncoder
    velocity: VelocityField

    def encode(self, features: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        """Encode (B, L, F) → (B, context_dim)."""
        return self.encoder(features, deterministic=deterministic)

    def predict_velocity(
        self,
        x_t: jnp.ndarray,
        t: jnp.ndarray,
        context: jnp.ndarray,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        return self.velocity(x_t, t, context, deterministic=deterministic)

    @nn.compact
    def __call__(
        self,
        x_t: jnp.ndarray,
        t: jnp.ndarray,
        features: jnp.ndarray,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        context = self.encode(features, deterministic=deterministic)
        return self.predict_velocity(x_t, t, context, deterministic=deterministic)


class FlowTrainState(train_state.TrainState):
    """Named alias for legibility; no extra fields (EMA is kept external)."""
    pass


def create_model(
    horizon: int,
    num_features: int,
    context_len: int,
    encoder_dim: int = 128,
    encoder_heads: int = 4,
    encoder_layers: int = 4,
    encoder_dropout: float = 0.1,
    velocity_hidden: int = 256,
    velocity_layers: int = 6,
    time_emb_dim: int = 128,
) -> BTCFlowModel:
    encoder = ContextEncoder(
        model_dim=encoder_dim,
        num_heads=encoder_heads,
        num_layers=encoder_layers,
        dropout_rate=encoder_dropout,
    )
    velocity = VelocityField(
        horizon=horizon,
        hidden_dim=velocity_hidden,
        num_layers=velocity_layers,
        context_dim=encoder_dim,
        time_emb_dim=time_emb_dim,
    )
    return BTCFlowModel(encoder=encoder, velocity=velocity)


def create_train_state(
    model: BTCFlowModel,
    key: jax.Array,
    horizon: int,
    context_len: int,
    num_features: int,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-4,
    warmup_steps: int = 1000,
    total_steps: int = 50000,
) -> FlowTrainState:
    """
    Initialise parameters and FlowTrainState with AdamW + cosine LR schedule.

    The cosine schedule decays from `learning_rate` to `learning_rate/100`
    over `total_steps`, with a linear warmup over `warmup_steps`.
    """
    dummy_x_t = jnp.zeros((1, horizon))
    dummy_t = jnp.zeros((1,))
    dummy_features = jnp.zeros((1, context_len, num_features))

    key, dropout_key = jax.random.split(key)
    params = model.init(
        {"params": key, "dropout": dropout_key},
        dummy_x_t, dummy_t, dummy_features, deterministic=False,
    )

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
        end_value=learning_rate * 0.01,
    )
    tx = optax.adamw(learning_rate=schedule, weight_decay=weight_decay)
    state = FlowTrainState.create(apply_fn=model.apply, params=params, tx=tx)

    n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    logger.info("Model initialised: %d parameters", n_params)
    return state


def make_ema_updater(decay: float):
    """
    Return a jit-compiled EMA update function that captures decay at
    compile time (no retracing when decay changes).
    """
    @jax.jit
    def _update(ema_params: Any, new_params: Any) -> Any:
        return jax.tree_util.tree_map(
            lambda e, p: decay * e + (1.0 - decay) * p,
            ema_params, new_params,
        )
    return _update


@jax.jit
def train_step(
    state: FlowTrainState,
    x1: jnp.ndarray,
    features: jnp.ndarray,
    key: jax.Array,
) -> tuple[FlowTrainState, jnp.ndarray]:
    """One jit-compiled training step. Returns (new_state, scalar_loss)."""
    t_key, x0_key, dropout_key = jax.random.split(key, 3)
    B, H = x1.shape

    t = jax.random.uniform(t_key, (B,))
    x0 = jax.random.normal(x0_key, (B, H))
    x_t = (1.0 - t[:, None]) * x0 + t[:, None] * x1
    target = x1 - x0

    def loss_fn(params: Any) -> jnp.ndarray:
        v_pred = state.apply_fn(
            params, x_t, t, features,
            deterministic=False, rngs={"dropout": dropout_key},
        )
        return jnp.mean((v_pred - target) ** 2)

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads), loss


@functools.partial(jax.jit, static_argnames=("compute_dtype",))
def train_step_with_stats(
    state: FlowTrainState,
    x1: jnp.ndarray,
    features: jnp.ndarray,
    key: jax.Array,
    compute_dtype=jnp.float32,
) -> tuple[FlowTrainState, jnp.ndarray, jnp.ndarray]:
    """
    Training step with gradient norm and optional mixed precision.

    Mixed-precision strategy
    -------------------------
    - Parameters and EMA stay **fp32** throughout.
    - When ``compute_dtype=jnp.bfloat16``, inputs and params are cast to
      bf16 for the forward pass; the loss is cast back to fp32 before the
      VJP.  JAX's autodiff through the cast accumulates gradients in fp32
      (the cotangent inherits the primal dtype), so the optimiser update
      is entirely fp32.  No loss scaling is required — flow-matching MSE
      losses are numerically well-behaved.
    - ``compute_dtype=jnp.float32`` is a no-op cast (identity); the
      compiled code is identical to ``train_step``.

    # DESIGN: two separate compiled functions (train_step / train_step_with_stats)
    # so that the simpler train_step used in unit tests has no extra arguments
    # and retraces less frequently.

    Returns
    -------
    (new_state, scalar_loss_fp32, grad_norm_fp32)
    """
    t_key, x0_key, dropout_key = jax.random.split(key, 3)
    B, H = x1.shape

    t  = jax.random.uniform(t_key,  (B,))
    x0 = jax.random.normal(x0_key, (B, H))
    x_t    = (1.0 - t[:, None]) * x0 + t[:, None] * x1
    target = x1 - x0  # fp32

    # Cast interpolated path, time, and features to compute dtype.
    x_t_c   = x_t.astype(compute_dtype)
    t_c     = t.astype(compute_dtype)
    feat_c  = features.astype(compute_dtype)
    target_c = target.astype(compute_dtype)

    def loss_fn(params: Any) -> jnp.ndarray:
        # Cast fp32 params to compute dtype for the forward pass.
        # VJP through astype: cotangents arrive in compute_dtype and are
        # cast back to fp32 before accumulation — fp32 gradient update.
        params_c = jax.tree_util.tree_map(
            lambda p: p.astype(compute_dtype), params
        )
        v_pred = state.apply_fn(
            params_c, x_t_c, t_c, feat_c,
            deterministic=False, rngs={"dropout": dropout_key},
        )
        return jnp.mean((v_pred - target_c) ** 2).astype(jnp.float32)

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    grad_norm = optax.global_norm(grads)
    return state.apply_gradients(grads=grads), loss, grad_norm


def eval_loss(
    params: Any,
    apply_fn,
    x1: jnp.ndarray,
    features: jnp.ndarray,
    key: jax.Array,
    compute_dtype=jnp.float32,
) -> jnp.ndarray:
    """
    FM loss without gradient (used with EMA params for validation).
    Not jit-compiled here; wrap at the call site with jax.jit if calling
    in a hot loop.
    """
    t_key, x0_key = jax.random.split(key)
    B, H = x1.shape
    t = jax.random.uniform(t_key, (B,))
    x0 = jax.random.normal(x0_key, (B, H))
    x_t = (1.0 - t[:, None]) * x0 + t[:, None] * x1
    target = x1 - x0

    x_t_c = x_t.astype(compute_dtype)
    t_c   = t.astype(compute_dtype)
    feat_c = features.astype(compute_dtype)
    target_c = target.astype(compute_dtype)
    params_c = jax.tree_util.tree_map(lambda p: p.astype(compute_dtype), params)

    v_pred = apply_fn(params_c, x_t_c, t_c, feat_c, deterministic=True)
    return jnp.mean((v_pred - target_c) ** 2).astype(jnp.float32)


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------

def save_checkpoint(
    path: Path,
    state: FlowTrainState,
    ema_params: Any,
    metadata: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "params": state.params,
        "ema_params": ema_params,
        "step": int(state.step),
        **metadata,
    }
    with open(path, "wb") as f:
        pickle.dump(ckpt, f)
    logger.info("Checkpoint saved → %s (step=%d)", path, int(state.step))


def load_checkpoint(path: Path) -> dict:
    with open(path, "rb") as f:
        ckpt = pickle.load(f)
    logger.info("Checkpoint loaded from %s (step=%d)", path, ckpt.get("step", -1))
    return ckpt
