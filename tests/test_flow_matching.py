from __future__ import annotations

"""
Flow-matching end-to-end test on a 2-D synthetic Gaussian mixture.

Synthetic data distribution
---------------------------
Context feature c[0] ~ Uniform(-1, 1) at the last timestep.
Target path x1:
    c[0] > 0  →  x1 ~ N(+0.5 · 1_H, 0.3² · I_H)
    c[0] ≤ 0  →  x1 ~ N(-0.5 · 1_H, 0.3² · I_H)

Tests
-----
1. Training loss decreases (final < initial over 200 steps).
2. Model CRPS < 0.9 × *climatological* CRPS.
   The climatological baseline is CRPS of the empirical marginal distribution
   of training targets — i.e. sampling x1 at random without using the
   conditioning signal c at all.  A model that never looks at context can
   exactly match climatological CRPS; a model that has learned to condition
   must beat it by ≥ 10 %.  This is stricter than comparing against N(0,1).
3. Ensemble shape is correct.
4. Parameter count is below 1 M.
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp

from btcfm.config import load_config
from btcfm.model.flow_matching import create_model, create_train_state, train_step, make_ema_updater
from btcfm.model.sampler import generate_ensemble
from btcfm.verification.crps import crps_per_lead


CONFIG_PATH = "configs/small.yaml"
NUM_FEATURES = 4     # small F for speed
NUM_TRAIN_STEPS = 400  # enough for convergence; ~30 s on CPU
N_CLIM = 2000        # climatological ensemble size
N_EVAL = 50          # number of test cases for averaging


def _make_batch(key, batch_size, context_len, num_features, horizon):
    """Conditional Gaussian mixture batch generator."""
    k1, k2, k3 = jax.random.split(key, 3)
    features = jax.random.normal(k1, (batch_size, context_len, num_features)) * 0.1
    cond = jax.random.uniform(k2, (batch_size,), minval=-1.0, maxval=1.0)
    features = features.at[:, -1, 0].set(cond)
    noise = jax.random.normal(k3, (batch_size, horizon)) * 0.3
    mean = jnp.where(cond[:, None] > 0, 0.5, -0.5) * jnp.ones((batch_size, horizon))
    x1 = mean + noise
    return features, x1


def _climatological_crps(key, config, n_features: int = NUM_FEATURES) -> tuple[float, jax.Array]:
    """
    CRPS of the empirical marginal distribution of training targets.

    The marginal ignores the conditioning signal (uses targets drawn from both
    mixture components equally).  This is the honest baseline: a model with
    no access to context can achieve exactly this score.

    Returns (mean_clim_crps, updated_key).
    """
    L = config.data.context_length
    H = config.data.horizon

    # Sample the full marginal (both components in equal proportion)
    key, batch_key = jax.random.split(key)
    _, clim_x1 = _make_batch(batch_key, N_CLIM, L, n_features, H)
    clim_ens = np.asarray(clim_x1)  # (N_CLIM, H) – empirical marginal

    crps_vals = []
    for _ in range(N_EVAL):
        key, eval_key = jax.random.split(key)
        _, test_x1 = _make_batch(eval_key, 1, L, n_features, H)
        y = np.asarray(test_x1[0])
        crps_vals.append(float(crps_per_lead(clim_ens, y).mean()))

    return float(np.mean(crps_vals)), key


def _model_crps(model, state, key, config, n_features: int = NUM_FEATURES, ema_params=None) -> tuple[float, jax.Array]:
    """
    CRPS of the trained model ensemble over N_EVAL test cases.
    Uses ema_params if provided, otherwise falls back to state.params.
    """
    L = config.data.context_length
    H = config.data.horizon
    params = ema_params if ema_params is not None else state.params
    crps_vals = []

    for _ in range(N_EVAL):
        key, eval_key, sample_key = jax.random.split(key, 3)
        eval_features, eval_x1 = _make_batch(eval_key, 1, L, n_features, H)
        y = np.asarray(eval_x1[0])

        context = model.apply(
            params, jnp.array(eval_features),
            method=model.encode, deterministic=True,
        )[0]

        ens = generate_ensemble(
            model, params, context, sample_key,
            horizon=H,
            n_samples=config.sampler.n_samples,
            n_steps=config.sampler.n_steps,
            solver=config.sampler.solver,
        )
        crps_vals.append(float(crps_per_lead(np.asarray(ens), y).mean()))

    return float(np.mean(crps_vals)), key


@pytest.fixture(scope="module")
def trained_state():
    """Train for NUM_TRAIN_STEPS; return (state, config, model, losses, key)."""
    config = load_config(CONFIG_PATH)
    L = config.data.context_length
    H = config.data.horizon

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

    key = jax.random.PRNGKey(config.train.seed)
    key, init_key = jax.random.split(key)
    state = create_train_state(
        model=model, key=init_key,
        horizon=H, context_len=L, num_features=NUM_FEATURES,
        learning_rate=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
        warmup_steps=config.train.warmup_steps,
        total_steps=NUM_TRAIN_STEPS * 3,  # keep LR in useful range during short fixture run
    )

    # Use a test-appropriate EMA decay so the EMA converges within NUM_TRAIN_STEPS.
    # config.train.ema_decay (0.999) is designed for ~50k production steps;
    # with 400 steps, 0.999^400 ≈ 0.67, meaning EMA is 67% random init params.
    # decay = 1 - 5/NUM_TRAIN_STEPS gives effective window of ~NUM_TRAIN_STEPS/5 steps.
    test_ema_decay = 1.0 - 5.0 / NUM_TRAIN_STEPS  # ≈ 0.9875 for 400 steps
    ema_update = make_ema_updater(test_ema_decay)
    ema_params = state.params

    losses = []
    for step in range(NUM_TRAIN_STEPS):
        key, data_key, step_key = jax.random.split(key, 3)
        features, x1 = _make_batch(data_key, config.train.batch_size, L, NUM_FEATURES, H)
        state, loss = train_step(state, x1, features, step_key)
        ema_params = ema_update(ema_params, state.params)
        losses.append(float(loss))

    return state, config, model, losses, key, ema_params


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_loss_decreases(trained_state):
    _, _, _, losses, _, _ = trained_state
    initial = np.mean(losses[:10])
    final = np.mean(losses[-10:])
    assert final < initial, (
        f"Loss did not decrease: initial={initial:.4f}, final={final:.4f}"
    )


def test_crps_beats_climatology(trained_state):
    """
    Model CRPS must be < 0.9 × climatological CRPS.

    Climatological CRPS is CRPS of the empirical marginal distribution of
    training targets — the score achievable by a model that uses no
    conditioning signal at all.  The 0.9× threshold requires the model to
    have genuinely learned to use context (a pure-context-ignorant model
    would score ≈ 1.0× climatological).
    """
    state, config, model, _, key, ema_params = trained_state

    clim_crps, key = _climatological_crps(key, config)
    # Evaluate using EMA params for best generalisation
    model_crps, _ = _model_crps(model, state, key, config, ema_params=ema_params)

    assert model_crps < 0.9 * clim_crps, (
        f"Model CRPS={model_crps:.4f} is not below 0.9 × climatological "
        f"CRPS={clim_crps:.4f} (threshold={0.9*clim_crps:.4f}). "
        f"The model has not learned to use the conditioning signal."
    )


def test_ensemble_shape(trained_state):
    state, config, model, _, key, ema_params = trained_state
    L, H, N = config.data.context_length, config.data.horizon, config.sampler.n_samples
    key, eval_key, sample_key = jax.random.split(key, 3)
    eval_features, _ = _make_batch(eval_key, 1, L, NUM_FEATURES, H)
    context = model.apply(
        ema_params, jnp.array(eval_features), method=model.encode, deterministic=True,
    )[0]
    ens = generate_ensemble(
        model, ema_params, context, sample_key,
        horizon=H, n_samples=N,
        n_steps=config.sampler.n_steps, solver=config.sampler.solver,
    )
    assert ens.shape == (N, H)


def test_model_param_count(trained_state):
    state, _, _, _, _, _ = trained_state
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(state.params))
    assert n_params < 1_000_000, f"Too many params for small config: {n_params}"
