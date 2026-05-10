from __future__ import annotations

"""
Unit test: bf16 and fp32 training steps produce numerically compatible gradients.

The test verifies that:
  1. train_step_with_stats runs without error in both fp32 and bf16 modes.
  2. The gradient norms agree to within 10 % relative tolerance.
     bf16 computes in ~3 decimal digits of precision; gradient norms
     accumulated from ~44k parameter paths can differ by a few percent —
     10 % is a generous but meaningful bound.
  3. Loss values agree to within 10 % relative tolerance.
  4. The returned loss and grad_norm are fp32 scalars in both modes.

These checks ensure the mixed-precision path does not silently produce NaNs,
overflow, or catastrophically wrong gradients.
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp

from btcfm.config import load_config
from btcfm.model.flow_matching import create_model, create_train_state, train_step_with_stats

CONFIG_PATH = "configs/small.yaml"
RTOL = 0.10          # 10 % relative tolerance between fp32 and bf16
NUM_FEATURES = 4


@pytest.fixture(scope="module")
def precision_fixtures():
    """Initialise a tiny model and a fixed batch for the precision tests."""
    config = load_config(CONFIG_PATH)
    H = config.data.horizon
    L = config.data.context_length

    model = create_model(
        horizon=H,
        num_features=NUM_FEATURES,
        context_len=L,
        encoder_dim=config.model.encoder_dim,
        encoder_heads=config.model.encoder_heads,
        encoder_layers=config.model.encoder_layers,
        encoder_dropout=config.model.encoder_dropout,
        velocity_hidden=config.model.velocity_hidden,
        velocity_layers=config.model.velocity_layers,
        time_emb_dim=config.model.time_emb_dim,
    )

    key = jax.random.PRNGKey(0)
    key, init_key = jax.random.split(key)
    state = create_train_state(
        model=model, key=init_key,
        horizon=H, context_len=L, num_features=NUM_FEATURES,
        learning_rate=3e-4, weight_decay=1e-4,
        warmup_steps=config.train.warmup_steps,
        total_steps=200,
    )

    key, data_key, step_key = jax.random.split(key, 3)
    x1       = jax.random.normal(data_key,  (config.train.batch_size, H))
    features = jax.random.normal(step_key,  (config.train.batch_size, L, NUM_FEATURES)) * 0.1

    return state, x1, features, key


class TestMixedPrecision:

    def test_fp32_runs(self, precision_fixtures):
        state, x1, features, key = precision_fixtures
        key, step_key = jax.random.split(key)
        new_state, loss, grad_norm = train_step_with_stats(
            state, x1, features, step_key, compute_dtype=jnp.float32,
        )
        assert jnp.isfinite(loss),     f"fp32 loss is not finite: {loss}"
        assert jnp.isfinite(grad_norm), f"fp32 grad_norm is not finite: {grad_norm}"

    def test_bf16_runs(self, precision_fixtures):
        state, x1, features, key = precision_fixtures
        key, step_key = jax.random.split(key)
        new_state, loss, grad_norm = train_step_with_stats(
            state, x1, features, step_key, compute_dtype=jnp.bfloat16,
        )
        assert jnp.isfinite(loss),     f"bf16 loss is not finite: {loss}"
        assert jnp.isfinite(grad_norm), f"bf16 grad_norm is not finite: {grad_norm}"

    def test_outputs_are_fp32(self, precision_fixtures):
        """Loss and grad_norm must be fp32 regardless of compute_dtype."""
        state, x1, features, key = precision_fixtures
        key, step_fp32, step_bf16 = jax.random.split(key, 3)

        _, loss_fp32, gn_fp32 = train_step_with_stats(
            state, x1, features, step_fp32, compute_dtype=jnp.float32,
        )
        _, loss_bf16, gn_bf16 = train_step_with_stats(
            state, x1, features, step_bf16, compute_dtype=jnp.bfloat16,
        )

        assert loss_fp32.dtype == jnp.float32, f"fp32 loss dtype: {loss_fp32.dtype}"
        assert gn_fp32.dtype   == jnp.float32, f"fp32 grad_norm dtype: {gn_fp32.dtype}"
        assert loss_bf16.dtype == jnp.float32, f"bf16 loss dtype: {loss_bf16.dtype}"
        assert gn_bf16.dtype   == jnp.float32, f"bf16 grad_norm dtype: {gn_bf16.dtype}"

    def test_grad_norm_close(self, precision_fixtures):
        """
        bf16 and fp32 gradient norms should agree within RTOL=10 %.

        Same key, same batch.  The only difference is the compute dtype.
        """
        state, x1, features, key = precision_fixtures
        key, step_key = jax.random.split(key)  # same key for both modes

        _, loss_fp32, gn_fp32 = train_step_with_stats(
            state, x1, features, step_key, compute_dtype=jnp.float32,
        )
        _, loss_bf16, gn_bf16 = train_step_with_stats(
            state, x1, features, step_key, compute_dtype=jnp.bfloat16,
        )

        gn_fp32_np = float(gn_fp32)
        gn_bf16_np = float(gn_bf16)
        rel_diff = abs(gn_fp32_np - gn_bf16_np) / max(gn_fp32_np, 1e-8)

        assert rel_diff < RTOL, (
            f"Gradient norm relative difference {rel_diff:.3f} exceeds {RTOL}.\n"
            f"fp32 grad_norm={gn_fp32_np:.6f}, bf16 grad_norm={gn_bf16_np:.6f}\n"
            f"bf16 forward pass may be numerically unstable for this model size."
        )

    def test_loss_close(self, precision_fixtures):
        """bf16 and fp32 loss values should agree within RTOL=10 %."""
        state, x1, features, key = precision_fixtures
        key, step_key = jax.random.split(key)

        _, loss_fp32, _ = train_step_with_stats(
            state, x1, features, step_key, compute_dtype=jnp.float32,
        )
        _, loss_bf16, _ = train_step_with_stats(
            state, x1, features, step_key, compute_dtype=jnp.bfloat16,
        )

        l32 = float(loss_fp32)
        l16 = float(loss_bf16)
        rel_diff = abs(l32 - l16) / max(l32, 1e-8)

        assert rel_diff < RTOL, (
            f"Loss relative difference {rel_diff:.3f} exceeds {RTOL}.\n"
            f"fp32 loss={l32:.6f}, bf16 loss={l16:.6f}"
        )
