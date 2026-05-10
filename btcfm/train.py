from __future__ import annotations

"""
Training entry-point for btcfm.

Supports two data modes:
  --synthetic : conditional Gaussian mixture toy (session-1 / CI default)
  (default)   : real BTC bars from the historical cache

Key features
------------
- AdamW optimiser with cosine LR schedule and linear warmup.
- EMA of parameters (half-life = ema_half_life_frac × num_steps).
- Mixed precision: bf16 forward/backward, fp32 params + EMA (A100 default).
- Async single-worker data prefetch for real-data mode.
- Per-step JSONL log + run_metadata.json for post-hoc analysis.
- Login-node guard: aborts on Slurm login node if GPU config is requested.
- Best-val-loss EMA checkpoint saved to {output_dir}/checkpoint_best.pkl.

Usage::

    # Synthetic smoke-test (also works on login node — CPU only):
    python -m btcfm.train --config configs/small.yaml --synthetic \\
        --output-dir runs/smoke

    # GPU job via sbatch (run-id = $SLURM_JOB_ID):
    python -m btcfm.train --config configs/default.yaml \\
        --run-id $SLURM_JOB_ID \\
        --data-dir $DATA_DIR --output-dir $RUN_DIR
"""

import argparse
import json
import logging
import queue
import subprocess
import threading
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from btcfm.config import load_config, BTCFMConfig
from btcfm.data.dataset import WindowDataset, prepare_datasets, assert_no_normalisation_leakage
from btcfm.features.builders import FEATURE_COLS
from btcfm.logging_utils import get_logger
from btcfm.model.flow_matching import (
    BTCFlowModel,
    FlowTrainState,
    create_model,
    create_train_state,
    make_ema_updater,
    train_step,
    train_step_with_stats,
    eval_loss,
    save_checkpoint,
)
from btcfm.model.sampler import generate_ensemble
from btcfm.model.training import ema_decay_for
from btcfm.verification.crps import crps_per_lead

logger = get_logger(__name__)

NUM_FEATURES = len(FEATURE_COLS)   # 15 for real data


# ---------------------------------------------------------------------------
# JSONL logger
# ---------------------------------------------------------------------------

class _JsonlLogger:
    """
    Line-buffered JSONL logger.  Each record is one JSON object per line.
    The file is opened in append mode so multiple restarts (without resume)
    are safe — older records are preserved for debugging.
    """

    def __init__(self, path: Path, run_id: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = path
        self._run_id = run_id
        self._f = open(path, "a", buffering=1)   # line-buffered

    def log(self, record: dict) -> None:
        record["ts"]     = datetime.now(tz=timezone.utc).isoformat()
        record["run_id"] = self._run_id
        self._f.write(json.dumps(record) + "\n")

    def close(self) -> None:
        self._f.close()


# ---------------------------------------------------------------------------
# Async single-worker prefetch loader (real data only)
# ---------------------------------------------------------------------------

class _PrefetchLoader:
    """
    Single-worker prefetch loader for WindowDataset.

    Submits the next batch fetch on a background thread immediately after the
    current batch is consumed, overlapping CPU data sampling with GPU compute.

    # DESIGN: single worker, queue depth 2. If nvidia-smi shows GPU
    utilisation below 80% during a run, the dataloader is the bottleneck.
    Revisit (increase queue size or switch to a proper DataLoader) before
    trusting throughput numbers.

    The NumPy Generator (``rng``) is accessed exclusively from the background
    thread after construction, so there is no thread-safety issue.
    """

    def __init__(
        self,
        dataset: WindowDataset,
        batch_size: int,
        rng: np.random.Generator,
        maxsize: int = 2,
    ) -> None:
        self._ds = dataset
        self._bs = batch_size
        self._rng = rng
        self._q: queue.Queue = queue.Queue(maxsize=maxsize)
        self._stop = threading.Event()
        self._worker = threading.Thread(
            target=self._run, daemon=True, name="PrefetchLoader"
        )
        self._worker.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                batch = self._ds.sample_batch(self._bs, self._rng)
                self._q.put(batch, timeout=0.5)
            except queue.Full:
                pass  # retry until the main thread consumes

    def next_batch(self) -> tuple[np.ndarray, np.ndarray]:
        return self._q.get()

    def close(self) -> None:
        self._stop.set()
        self._worker.join(timeout=2.0)


# ---------------------------------------------------------------------------
# Synthetic data generator
# ---------------------------------------------------------------------------

def make_synthetic_batch(
    key: jax.Array,
    batch_size: int,
    context_len: int,
    num_features: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Conditional Gaussian mixture toy:
      c[0] ~ Uniform(-1, 1) at last timestep
      x1 | c[0] > 0  ~ N(+0.5, 0.3²) per lead time (iid)
      x1 | c[0] <= 0 ~ N(-0.5, 0.3²) per lead time (iid)
    """
    k1, k2, k3 = jax.random.split(key, 3)
    features = jax.random.normal(k1, (batch_size, context_len, num_features)) * 0.1
    cond = jax.random.uniform(k2, (batch_size,), minval=-1.0, maxval=1.0)
    features = features.at[:, -1, 0].set(cond)
    noise = jax.random.normal(k3, (batch_size, horizon)) * 0.3
    mean = jnp.where(cond[:, None] > 0, 0.5, -0.5) * jnp.ones((batch_size, horizon))
    x1 = mean + noise
    return np.asarray(features), np.asarray(x1)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _get_git_sha() -> str:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        return sha[:12]
    except Exception:
        return "unknown"


def _gpu_mem_gb() -> float:
    """Best-effort GPU memory in use (GB). Returns 0.0 on CPU or if unavailable."""
    try:
        stats = jax.devices()[0].memory_stats()
        if stats and "bytes_in_use" in stats:
            return stats["bytes_in_use"] / 1e9
    except Exception:
        pass
    return 0.0


def _make_model_and_state(
    config: BTCFMConfig,
    num_features: int,
    key: jax.Array,
    total_steps: int,
) -> tuple[BTCFlowModel, FlowTrainState]:
    model = create_model(
        horizon=config.data.horizon,
        num_features=num_features,
        context_len=config.data.context_length,
        encoder_dim=config.model.encoder_dim,
        encoder_heads=config.model.encoder_heads,
        encoder_layers=config.model.encoder_layers,
        encoder_dropout=config.model.encoder_dropout,
        velocity_hidden=config.model.velocity_hidden,
        velocity_layers=config.model.velocity_layers,
        time_emb_dim=config.model.time_emb_dim,
    )
    state = create_train_state(
        model=model, key=key,
        horizon=config.data.horizon,
        context_len=config.data.context_length,
        num_features=num_features,
        learning_rate=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
        warmup_steps=config.train.warmup_steps,
        total_steps=total_steps,
    )
    return model, state


def _make_lr_schedule(config: BTCFMConfig, total_steps: int):
    """Return the same cosine schedule used inside create_train_state."""
    import optax
    return optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=config.train.learning_rate,
        warmup_steps=config.train.warmup_steps,
        decay_steps=total_steps,
        end_value=config.train.learning_rate * 0.01,
    )


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _compute_val_loss(
    model: BTCFlowModel,
    ema_params: Any,
    val_ds: WindowDataset | None,
    rng: np.random.Generator,
    batch_size: int,
    key: jax.Array,
    synthetic: bool,
    config: BTCFMConfig,
    num_features: int,
    compute_dtype,
) -> tuple[float, jax.Array]:
    key, data_key, loss_key = jax.random.split(key, 3)
    if synthetic:
        feat_np, x1_np = make_synthetic_batch(
            data_key, batch_size, config.data.context_length, num_features, config.data.horizon,
        )
    else:
        assert val_ds is not None
        feat_np, x1_np = val_ds.sample_batch(batch_size, rng)

    loss = eval_loss(
        ema_params, model.apply,
        jnp.array(x1_np), jnp.array(feat_np), loss_key,
        compute_dtype=compute_dtype,
    )
    return float(loss), key


def _compute_crps_sample(
    model: BTCFlowModel,
    ema_params: Any,
    feat_np: np.ndarray,
    x1_np: np.ndarray,
    config: BTCFMConfig,
    key: jax.Array,
) -> tuple[float, jax.Array]:
    """Compute mean CRPS over the first item in a batch (representative)."""
    key, sample_key = jax.random.split(key)
    context = model.apply(
        ema_params, jnp.array(feat_np[:1]),
        method=model.encode, deterministic=True,
    )[0]
    ens = generate_ensemble(
        model, ema_params, context, sample_key,
        horizon=config.data.horizon,
        n_samples=config.sampler.n_samples,
        n_steps=config.sampler.n_steps,
        solver=config.sampler.solver,
    )
    crps = float(crps_per_lead(np.asarray(ens), x1_np[0]).mean())
    return crps, key


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(
    config: BTCFMConfig,
    *,
    synthetic: bool = True,
    data_dir: Path | None = None,
    output_dir: Path | None = None,
    run_id: str = "",
    num_features: int = NUM_FEATURES,
) -> tuple[FlowTrainState, Any]:
    """
    Run the training loop.

    Returns
    -------
    (state, ema_params) — the final state and best-val EMA parameters.
    The best-validation-loss EMA checkpoint is saved to
    {output_dir}/checkpoint_best.pkl.
    """
    from btcfm.runtime.preflight import check_not_login_node, gpu_preflight, resolve_precision

    # ------------------------------------------------------------------
    # Login-node guard (must run before JAX materialises GPU context)
    # ------------------------------------------------------------------
    check_not_login_node(config.train.precision)

    cfg_hash = config.config_hash()
    git_sha  = _get_git_sha()
    rng = np.random.default_rng(config.train.seed)

    if not run_id:
        run_id = cfg_hash

    logger.info(
        "=== btcfm training ===  run_id=%s  config_hash=%s  seed=%d  "
        "git_sha=%s  synthetic=%s",
        run_id, cfg_hash, config.train.seed, git_sha, synthetic,
    )

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # GPU pre-flight + precision resolution
    # ------------------------------------------------------------------
    devs = jax.devices()
    has_gpu = any(d.platform == "gpu" for d in devs)
    device_kind: str | None = None

    if has_gpu:
        device_kind = gpu_preflight()
    else:
        logger.info("No GPU detected — running on CPU (synthetic smoke-test mode).")

    compute_dtype = resolve_precision(config.train.precision, device_kind)
    precision_str = {
        jnp.float32:  "fp32",
        jnp.bfloat16: "bf16",
        jnp.float16:  "fp16",
    }.get(compute_dtype, str(compute_dtype))

    logger.info("Config: %s", config)
    logger.info("Compute dtype: %s", precision_str)

    # ------------------------------------------------------------------
    # XLA determinism flag reminder
    # ------------------------------------------------------------------
    import os
    xla_flags = os.environ.get("XLA_FLAGS", "")
    if has_gpu and "--xla_gpu_deterministic_ops=true" not in xla_flags:
        logger.warning(
            "XLA_FLAGS does not include --xla_gpu_deterministic_ops=true. "
            "Results may not be exactly reproducible across runs. "
            "Set this flag in the sbatch script for the reported run."
        )

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    train_ds: WindowDataset | None = None
    val_ds:   WindowDataset | None = None
    test_ds:  WindowDataset | None = None
    data_meta: dict = {}
    num_features_actual = num_features
    loader: _PrefetchLoader | None = None

    if not synthetic:
        if data_dir is None:
            raise ValueError("data_dir must be set when synthetic=False")
        from btcfm.data.coinbase_rest import fetch_bars, most_recent_closed_day
        from datetime import timedelta

        end_date = most_recent_closed_day()
        start_date = end_date - timedelta(days=179)
        logger.info("Loading bars %s → %s", start_date, end_date)

        bars = fetch_bars(
            product="BTC-USD",
            start_date=start_date,
            end_date=end_date,
            cache_dir=Path(data_dir),
        )
        data_meta = {
            "data_start": str(start_date),
            "data_end": str(end_date),
            "n_bars": len(bars),
        }
        logger.info("Bars loaded: %d minutes", len(bars))

        train_ds, val_ds, test_ds, norm_state = prepare_datasets(bars, config)
        assert_no_normalisation_leakage(val_ds, norm_state)

        if output_dir is not None:
            norm_path = output_dir / "norm_state.json"
            norm_state.save(norm_path)
            logger.info("Normalisation state → %s", norm_path)

        num_features_actual = len(FEATURE_COLS)
        logger.info(
            "Datasets: train=%d, val=%d, test=%d windows",
            len(train_ds), len(val_ds), len(test_ds),
        )

        # Async prefetch for the training split
        loader = _PrefetchLoader(train_ds, config.train.batch_size, rng)
    else:
        data_meta = {"data": "synthetic_gaussian_mixture"}

    # ------------------------------------------------------------------
    # Model + state
    # ------------------------------------------------------------------
    key = jax.random.PRNGKey(config.train.seed)
    key, init_key = jax.random.split(key)

    model, state = _make_model_and_state(
        config, num_features_actual, init_key, config.train.num_steps
    )

    ema_decay = ema_decay_for(config.train.num_steps, config.train.ema_half_life_frac)
    ema_update = make_ema_updater(ema_decay)
    ema_params = state.params   # initialise EMA = params

    lr_schedule = _make_lr_schedule(config, config.train.num_steps)

    logger.info(
        "EMA decay = %.8f  (half-life = %d steps, frac=%.2f)",
        ema_decay,
        max(1, int(config.train.ema_half_life_frac * config.train.num_steps)),
        config.train.ema_half_life_frac,
    )

    # ------------------------------------------------------------------
    # run_metadata.json  (written once at startup)
    # ------------------------------------------------------------------
    metadata_base = {
        "run_id":          run_id,
        "config_hash":     cfg_hash,
        "git_sha":         git_sha,
        "seed":            config.train.seed,
        "precision":       precision_str,
        "compute_dtype":   precision_str,
        "batch_size":      config.train.batch_size,
        "num_steps":       config.train.num_steps,
        "ema_decay":       ema_decay,
        "ema_half_life_frac": config.train.ema_half_life_frac,
        "num_features":    num_features_actual,
        "jax_version":     jax.__version__,
        "gpu_device":      device_kind or "cpu",
        "xla_flags":       xla_flags,
        "config":          asdict(config),
        **data_meta,
    }
    try:
        import jaxlib
        metadata_base["jaxlib_version"] = jaxlib.__version__
    except Exception:
        pass

    if output_dir is not None:
        meta_path = output_dir / "run_metadata.json"
        meta_path.write_text(json.dumps(metadata_base, indent=2))
        logger.info("run_metadata.json → %s", meta_path)

    # ------------------------------------------------------------------
    # JSONL logger
    # ------------------------------------------------------------------
    jsonl_logger: _JsonlLogger | None = None
    if output_dir is not None:
        jsonl_logger = _JsonlLogger(output_dir / "train.jsonl", run_id)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    losses: list[float] = []
    val_losses: list[float] = []
    crps_history: list[float] = []
    best_val_loss = float("inf")
    best_ema_params = ema_params
    first_step_done = False
    t0_compile = time.perf_counter()
    t_prev = t0_compile

    for step in range(1, config.train.num_steps + 1):
        key, data_key, step_key = jax.random.split(key, 3)

        # Sample batch
        if synthetic:
            feat_np, x1_np = make_synthetic_batch(
                data_key, config.train.batch_size,
                config.data.context_length, num_features_actual, config.data.horizon,
            )
        else:
            assert loader is not None
            feat_np, x1_np = loader.next_batch()

        # Train step (with stats for production logging)
        state, loss, grad_norm = train_step_with_stats(
            state,
            jnp.array(x1_np),
            jnp.array(feat_np),
            step_key,
            compute_dtype=compute_dtype,
        )
        ema_params = ema_update(ema_params, state.params)

        loss_val     = float(loss)
        grad_norm_val = float(grad_norm)
        losses.append(loss_val)

        # Time the first compiled step
        if not first_step_done:
            loss.block_until_ready()
            t_compile = time.perf_counter() - t0_compile
            logger.info("First compiled step: %.3f s", t_compile)
            first_step_done = True

        # Per-step JSONL record (every step, line-buffered)
        if jsonl_logger is not None:
            t_now = time.perf_counter()
            sps = 1.0 / max(t_now - t_prev, 1e-6)
            t_prev = t_now
            lr_val = float(lr_schedule(step - 1))
            jsonl_logger.log({
                "step":          step,
                "phase":         "train",
                "loss":          round(loss_val, 6),
                "lr":            round(lr_val, 8),
                "grad_norm":     round(grad_norm_val, 6),
                "steps_per_sec": round(sps, 2),
                "gpu_mem_gb":    round(_gpu_mem_gb(), 3),
            })

        # Validation + CRPS logging
        if step % config.train.val_every == 0:
            val_loss, key = _compute_val_loss(
                model, ema_params, val_ds, rng,
                config.train.batch_size, key, synthetic, config,
                num_features_actual, compute_dtype,
            )
            val_losses.append(val_loss)

            key, eval_data_key = jax.random.split(key)
            if synthetic:
                eval_feat, eval_x1 = make_synthetic_batch(
                    eval_data_key, 1,
                    config.data.context_length, num_features_actual, config.data.horizon,
                )
            else:
                assert val_ds is not None
                eval_feat, eval_x1 = val_ds.sample_batch(1, rng)

            crps_val, key = _compute_crps_sample(
                model, ema_params, eval_feat, eval_x1, config, key,
            )
            crps_history.append(crps_val)

            recent_train = float(np.mean(losses[-config.train.val_every:]))
            logger.info(
                "step=%5d  train_loss=%.4f  val_loss=%.4f  CRPS=%.4f",
                step, recent_train, val_loss, crps_val,
            )

            if jsonl_logger is not None:
                jsonl_logger.log({
                    "step":              step,
                    "phase":             "val",
                    "val_loss":          round(val_loss, 6),
                    "val_crps_terminal": round(crps_val, 6),
                })

            # Save best checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_ema_params = ema_params
                if output_dir is not None:
                    _save_full_checkpoint(
                        output_dir / "checkpoint_best.pkl",
                        state, ema_params, key,
                        config, ema_decay, step, val_loss, metadata_base,
                    )

        # Periodic checkpoint (every checkpoint_every steps)
        if output_dir is not None and step % config.train.checkpoint_every == 0:
            _save_full_checkpoint(
                output_dir / f"checkpoint_{step:06d}.pkl",
                state, ema_params, key,
                config, ema_decay, step, None, metadata_base,
            )

    # ------------------------------------------------------------------
    # Final checkpoint
    # ------------------------------------------------------------------
    if output_dir is not None:
        _save_full_checkpoint(
            output_dir / "checkpoint_final.pkl",
            state, ema_params, key,
            config, ema_decay, config.train.num_steps, None, metadata_base,
        )

    # ------------------------------------------------------------------
    # Steady-state timing
    # ------------------------------------------------------------------
    t_ss = time.perf_counter()
    n_timing = 10
    for _ in range(n_timing):
        key, dk, sk = jax.random.split(key, 3)
        if synthetic:
            fp, xp = make_synthetic_batch(dk, config.train.batch_size,
                                          config.data.context_length,
                                          num_features_actual, config.data.horizon)
        else:
            assert loader is not None
            fp, xp = loader.next_batch()
        state, loss, _ = train_step_with_stats(
            state, jnp.array(xp), jnp.array(fp), sk,
            compute_dtype=compute_dtype,
        )
    loss.block_until_ready()
    t_ss_per = (time.perf_counter() - t_ss) / n_timing
    logger.info(
        "Steady-state step: %.4f s/step  (%.0f steps/min)",
        t_ss_per, 60 / t_ss_per,
    )

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    if output_dir is not None and losses:
        _save_training_plots(losses, val_losses, crps_history, config.train.val_every, output_dir)

    if jsonl_logger is not None:
        jsonl_logger.close()

    if loader is not None:
        loader.close()

    logger.info("Training complete.  best_val_loss=%.4f", best_val_loss)
    return state, best_ema_params


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _save_full_checkpoint(
    path: Path,
    state: FlowTrainState,
    ema_params: Any,
    rng_key: jax.Array,
    config: BTCFMConfig,
    ema_decay: float,
    step: int,
    val_loss: float | None,
    metadata: dict,
) -> None:
    """
    Save a self-contained checkpoint loadable by infer.py / run_verification.py.

    Contents
    --------
    params      : current (non-EMA) parameters — kept for completeness
    ema_params  : EMA-averaged parameters (use these for inference)
    opt_state   : optimizer state (stored for future checkpoint-resume; unused now)
    step        : training step at save time
    rng_key     : JAX PRNG key state (for future resume)
    config      : full resolved config dict (with ema_decay inlined)
    ema_decay   : computed EMA decay value
    val_loss    : validation loss at save time (None for periodic checkpoints)
    ...metadata : run_id, config_hash, git_sha, data window, etc.
    """
    from btcfm.model.flow_matching import save_checkpoint
    extra: dict = {
        "opt_state":  state.opt_state,
        "rng_key":    rng_key,
        "config":     asdict(config),
        "ema_decay":  ema_decay,
        "num_features": metadata.get("num_features"),
    }
    if val_loss is not None:
        extra["val_loss"] = val_loss
    save_checkpoint(
        path, state, ema_params,
        {**metadata, **extra},
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _save_training_plots(
    losses: list[float],
    val_losses: list[float],
    crps_history: list[float],
    val_every: int,
    output_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(losses, lw=0.8)
    axes[0].set_xlabel("Step"); axes[0].set_ylabel("FM loss")
    axes[0].set_title("Training loss"); axes[0].set_yscale("log")

    if val_losses:
        val_steps = [i * val_every for i in range(1, len(val_losses) + 1)]
        axes[1].plot(val_steps, val_losses, marker="o", ms=4)
        axes[1].set_xlabel("Step"); axes[1].set_ylabel("Val loss")
        axes[1].set_title("Validation loss (EMA params)")

    if crps_history:
        crps_steps = [i * val_every for i in range(1, len(crps_history) + 1)]
        axes[2].plot(crps_steps, crps_history, marker="o", ms=4)
        axes[2].set_xlabel("Step"); axes[2].set_ylabel("Mean CRPS")
        axes[2].set_title("CRPS vs step")

    plt.tight_layout()
    path = output_dir / "training_curves.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    logger.info("Plots → %s", path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train btcfm flow-matching model")
    parser.add_argument("--config",     default="configs/small.yaml")
    parser.add_argument("--synthetic",  action="store_true", default=False)
    parser.add_argument("--data-dir",   default=None)
    parser.add_argument("--output-dir", default="runs/latest")
    parser.add_argument(
        "--run-id", default="",
        help="Unique run identifier (default: config hash). Set to $SLURM_JOB_ID in sbatch scripts.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    train(
        config,
        synthetic=args.synthetic,
        data_dir=Path(args.data_dir) if args.data_dir else None,
        output_dir=Path(args.output_dir),
        run_id=args.run_id,
    )


if __name__ == "__main__":
    main()
