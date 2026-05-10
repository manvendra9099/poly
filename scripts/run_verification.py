#!/usr/bin/env python3
"""
Test-set verification pipeline.

Loads the best-validation EMA checkpoint, runs ensemble forecasts over the
held-out test window, and produces:
  <output-dir>/
    crps_vs_lead.{png,csv}
    rank_histograms.png
    reliability.{png,csv}
    spread_skill.{png,csv}
    pinball_loss.{png,csv}
    REPORT.md

Baselines for CRPS
------------------
  Climatology : CRPS of the empirical marginal distribution of training
                targets (no conditioning used).
  Persistence : CRPS of N(0, σ_k) where σ_k = sqrt(k) * σ_1min and σ_1min
                is the per-step volatility from training returns.

Usage (preferred — run-dir mode)::

    python scripts/run_verification.py \\
        --run-dir $RUN_DIR \\
        --config configs/default.yaml \\
        --data-dir $DATA_DIR

    # Writes results to $RUN_DIR/outputs/test/

Usage (explicit paths)::

    python scripts/run_verification.py \\
        --checkpoint runs/exp01/checkpoint_best.pkl \\
        --config configs/default.yaml \\
        --data-dir data/cache/coinbase \\
        --output-dir outputs/test
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from btcfm.config import load_config
from btcfm.data.coinbase_rest import fetch_bars, most_recent_closed_day
from btcfm.data.dataset import prepare_datasets
from btcfm.features.builders import FEATURE_COLS, build_features
from btcfm.features.normalise import NormState
from btcfm.logging_utils import get_logger
from btcfm.model.flow_matching import create_model, load_checkpoint
from btcfm.model.sampler import generate_ensemble
from btcfm.verification.crps import crps_per_lead, crps_gaussian_analytic
from btcfm.verification.rank_hist import rank_histogram, flatness_score
from btcfm.verification.reliability import reliability_diagram
from btcfm.verification.spread_skill import (
    spread_skill, pinball_loss, QUANTILE_LEVELS,
)

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def _climatology_crps(
    train_returns: np.ndarray,   # (N_train, H)
    test_returns: np.ndarray,    # (T, H)
) -> np.ndarray:
    """
    CRPS where the forecast is the empirical marginal of training returns.

    Returns array of shape (H,) — mean CRPS per lead time.
    """
    H = train_returns.shape[1]
    T = test_returns.shape[0]
    scores = np.zeros((T, H))
    for h in range(H):
        clim_ens = train_returns[:, h]       # (N_train,) empirical marginal
        for t in range(T):
            scores[t, h] = crps_per_lead(clim_ens[:, None], test_returns[t:t+1, h])[0]
    return scores.mean(axis=0)


def _persistence_crps(
    train_returns: np.ndarray,   # (N_train, H)
    test_returns: np.ndarray,    # (T, H)
) -> np.ndarray:
    """
    Persistence forecast: r_{t0+k} ~ N(0, σ_k²), σ_k = sqrt(k) * σ_1min.

    Returns array of shape (H,) — mean CRPS per lead time.
    """
    sigma_1min = float(train_returns[:, 0].std())
    H = test_returns.shape[1]
    T = test_returns.shape[0]
    scores = np.zeros((T, H))
    for h in range(H):
        sigma_k = sigma_1min * np.sqrt(h + 1)
        for t in range(T):
            scores[t, h] = crps_gaussian_analytic(0.0, sigma_k, float(test_returns[t, h]))
    return scores.mean(axis=0)


# ---------------------------------------------------------------------------
# Forecast loop
# ---------------------------------------------------------------------------

def run_forecast_loop(
    model,
    ema_params,
    test_X: np.ndarray,   # (T, L, F) normalised context
    config,
    key: jax.Array,
    stride: int = 5,
) -> np.ndarray:
    """
    Generate ensemble for every `stride`-th test window.

    # DESIGN: stride=5 (every 5 minutes) to keep CPU runtime tractable.
    At n_samples=1000, 50 Heun steps, one ensemble is several CPU-seconds.
    15 days × 1440 / 5 = 4320 forecasts vs 21600.
    Set stride=1 for GPU runs.

    Returns
    -------
    np.ndarray, shape (T_eff, N, H)  where T_eff = len(range(0, T, stride))
    """
    T = len(test_X)
    indices = list(range(0, T, stride))
    logger.info(
        "Forecast loop: %d issue times (stride=%d, n_samples=%d, n_steps=%d)",
        len(indices), stride, config.sampler.n_samples, config.sampler.n_steps,
    )

    ensembles = []
    t_start = time.perf_counter()

    for ii, i in enumerate(indices):
        key, sk = jax.random.split(key)
        ctx = model.apply(
            ema_params, jnp.array(test_X[i:i+1]),
            method=model.encode, deterministic=True,
        )[0]
        ens = generate_ensemble(
            model, ema_params, ctx, sk,
            horizon=config.data.horizon,
            n_samples=config.sampler.n_samples,
            n_steps=config.sampler.n_steps,
            solver=config.sampler.solver,
        )
        ensembles.append(np.asarray(ens))

        if (ii + 1) % 50 == 0:
            elapsed = time.perf_counter() - t_start
            rate = elapsed / (ii + 1)
            eta = rate * (len(indices) - ii - 1)
            logger.info("  %d/%d  %.2f s/fc  ETA %.0f s", ii+1, len(indices), rate, eta)

    return np.stack(ensembles)  # (T_eff, N, H)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _plot_crps(model_crps, clim_crps, pers_crps, output_dir: Path) -> None:
    H = len(model_crps)
    leads = np.arange(1, H + 1)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(leads, model_crps, "o-", lw=2, ms=4, label="Model (EMA)")
    ax.plot(leads, clim_crps,  "s--", lw=1.5, ms=4, label="Climatology")
    ax.plot(leads, pers_crps,  "^--", lw=1.5, ms=4, label="Persistence N(0,σ√k)")
    ax.set_xlabel("Lead time (minutes)")
    ax.set_ylabel("Mean CRPS")
    ax.set_title("CRPS vs lead time — test set")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_dir / "crps_vs_lead.png", dpi=120)
    plt.close(fig)


def _plot_rank_hists(ensembles, test_y, lead_times, output_dir: Path) -> None:
    lead_times = [lt for lt in lead_times if lt <= ensembles.shape[2]]
    n = len(lead_times)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1: axes = [axes]
    for ax, lt in zip(axes, lead_times):
        h = lt - 1
        hist = rank_histogram(ensembles[:, :, h], test_y[:, h], normalise=True)
        ks = flatness_score(hist)
        ax.bar(np.arange(len(hist)), hist, width=1.0, color="steelblue", alpha=0.8)
        ax.axhline(1 / len(hist), color="red", ls="--", lw=1.5, label="uniform")
        ax.set_title(f"Lead {lt} min  KS={ks:.3f}")
        ax.set_xlabel("Rank")
        if ax is axes[0]: ax.set_ylabel("Frequency")
        ax.legend(fontsize=8)
    plt.suptitle("Rank histograms — test set")
    plt.tight_layout()
    fig.savefig(output_dir / "rank_histograms.png", dpi=120)
    plt.close(fig)


def _plot_reliability(ensembles, test_y, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    events = [
        ("terminal > 0", lambda ens: ens.sum(axis=1) > 0,
         lambda y: y.sum() > 0),
        ("touch +10 bps", lambda ens: np.exp(np.cumsum(ens, axis=1)).max(axis=1) > 1.001,
         lambda y: np.exp(np.cumsum(y)).max() > 1.001),
    ]

    for ax, (title, event_fn, obs_fn) in zip(axes, events):
        obs = np.array([float(obs_fn(test_y[i])) for i in range(len(test_y))])
        obs_arr = obs[:, None] * np.ones((len(test_y), 1))

        rel = reliability_diagram(
            ensembles,
            event_fn,
            obs_arr,
            n_bins=8, n_bootstrap=200, ci_level=0.9,
        )
        valid = ~np.isnan(rel["obs_frequency"])
        ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Perfect")
        ax.fill_between(
            rel["bin_centers"][valid], rel["ci_lower"][valid], rel["ci_upper"][valid],
            alpha=0.25, color="steelblue",
        )
        ax.plot(rel["mean_forecast"][valid], rel["obs_frequency"][valid],
                "o-", color="steelblue", lw=2, ms=6, label="Model")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("Forecast probability")
        ax.set_ylabel("Observed frequency")
        ax.set_title(f"Reliability: {title}")
        ax.legend(fontsize=9)

    plt.suptitle("Reliability diagrams — test set")
    plt.tight_layout()
    fig.savefig(output_dir / "reliability.png", dpi=120)
    plt.close(fig)


def _plot_spread_skill(ensembles, test_y, lead_times, output_dir: Path) -> None:
    lead_times = [lt for lt in lead_times if lt <= ensembles.shape[2]]
    n = len(lead_times)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1: axes = [axes]
    for ax, lt in zip(axes, lead_times):
        h = lt - 1
        from btcfm.verification.spread_skill import spread_skill
        ss = spread_skill(ensembles[:, :, h], test_y[:, h])
        valid = ~np.isnan(ss["bin_spread"])
        lim = max(ss["bin_spread"][valid].max(), ss["bin_rmse"][valid].max()) * 1.1
        ax.plot([0, lim], [0, lim], "k--", lw=1, label="y=x (ideal)")
        ax.scatter(ss["bin_spread"][valid], ss["bin_rmse"][valid],
                   s=40 * ss["counts"][valid] / ss["counts"][valid].max() + 10,
                   color="steelblue", zorder=3)
        ax.set_xlabel("Spread (ensemble std)"); ax.set_ylabel("|Error|")
        ax.set_title(f"Spread–skill  lead {lt} min")
        ax.legend(fontsize=8)
    plt.suptitle("Spread–skill — test set")
    plt.tight_layout()
    fig.savefig(output_dir / "spread_skill.png", dpi=120)
    plt.close(fig)


def _plot_pinball(ensembles, test_y, output_dir: Path) -> None:
    from btcfm.verification.spread_skill import pinball_loss_per_lead
    pb = pinball_loss_per_lead(ensembles, test_y)  # (H, Q)
    H, Q = pb.shape
    leads = np.arange(1, H + 1)
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = [f"q={q:.2f}" for q in QUANTILE_LEVELS]
    for qi in range(Q):
        ax.plot(leads, pb[:, qi], lw=1.5, label=labels[qi])
    ax.set_xlabel("Lead time (minutes)")
    ax.set_ylabel("Pinball loss")
    ax.set_title("Quantile / pinball loss vs lead time — test set")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_dir / "pinball_loss.png", dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# REPORT.md
# ---------------------------------------------------------------------------

def _write_report(
    output_dir: Path,
    ckpt_meta: dict,
    config,
    n_train: int,
    n_test_forecasts: int,
    stride: int,
    model_crps: np.ndarray,
    clim_crps: np.ndarray,
    pers_crps: np.ndarray,
    lead_times: list[int],
) -> None:
    H = len(model_crps)
    beats_clim = np.all(model_crps < clim_crps)
    beats_pers = np.all(model_crps < pers_crps)

    report = f"""# btcfm Calibration Report

## Run metadata
- **Config hash**: `{ckpt_meta.get('config_hash', 'N/A')}`
- **Git SHA**: `{ckpt_meta.get('git_sha', 'N/A')}`
- **Data window**: {ckpt_meta.get('data_start', 'N/A')} → {ckpt_meta.get('data_end', 'N/A')}
- **Best val loss**: {ckpt_meta.get('val_loss', float('nan')):.4f}

## Split
- Train: 150 days ({n_train:,} training windows)
- Val: 15 days
- Test: 15 days ({n_test_forecasts:,} forecast issue times, stride={stride} min)

## Model
- Encoder: {config.model.encoder_layers}L × dim {config.model.encoder_dim}, {config.model.encoder_heads} heads
- Velocity field: {config.model.velocity_layers}L × dim {config.model.velocity_hidden}, SiLU
- Sampler: {config.sampler.solver}, {config.sampler.n_steps} steps, {config.sampler.n_samples} samples

## CRPS vs lead time

| Lead (min) | Model | Climatology | Persistence | Beats clim? | Beats pers? |
|---|---|---|---|---|---|
"""
    for h in range(H):
        lt = h + 1
        bc = "✓" if model_crps[h] < clim_crps[h] else "✗"
        bp = "✓" if model_crps[h] < pers_crps[h] else "✗"
        report += (
            f"| {lt} | {model_crps[h]:.4f} | {clim_crps[h]:.4f} | "
            f"{pers_crps[h]:.4f} | {bc} | {bp} |\n"
        )

    report += f"""
## Summary interpretation

The model {'beats' if beats_clim else 'does NOT beat'} the climatological baseline
at all lead times, and {'beats' if beats_pers else 'does NOT beat'} the persistence baseline
at all lead times.

"""
    if not beats_clim:
        report += (
            "**Finding:** The model fails to beat climatology at some lead times.  "
            "This may indicate insufficient training, underfitting, or that the "
            "conditioning features carry less information than expected.  "
            "Inspect the rank histogram for calibration structure.\n\n"
        )
    if not beats_pers:
        report += (
            "**Finding:** The model does not beat persistence at all leads.  "
            "At very short horizons (1–2 min) persistence is a strong baseline "
            "due to the near-martingale nature of BTC returns; "
            "underperformance at longer horizons warrants investigation.\n\n"
        )

    report += (
        "**Note on vol/count imbalance features:** These features are identically 0 "
        "for REST-loaded data (the candle API does not provide trade direction).  "
        "The vol_imbal and cnt_imbal features carry no signal; "
        "the normaliser collapses them to 0.  "
        "Real directional conditioning requires the live WebSocket tick stream.\n\n"
        "**Note on config:** "
        f"{'These results use small.yaml (CI only). Do NOT cite these as calibration results.' if config.model.encoder_dim < 128 else 'Results use default.yaml (research-grade config).'}\n"
    )

    path = output_dir / "REPORT.md"
    path.write_text(report)
    logger.info("REPORT.md → %s", path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="btcfm test-set verification")
    # Preferred: supply --run-dir and let the script infer checkpoint + output paths
    parser.add_argument(
        "--run-dir", default=None,
        help="Run output directory (e.g. $SCRATCH/btcfm/runs/$SLURM_JOB_ID). "
             "Infers checkpoint_best.pkl and writes results to <run-dir>/outputs/test/.",
    )
    # Legacy / explicit-path mode
    parser.add_argument("--checkpoint", default=None,
                        help="Explicit checkpoint path (overrides --run-dir).")
    parser.add_argument("--config",     required=True)
    parser.add_argument("--data-dir",   required=True)
    parser.add_argument("--output-dir", default=None,
                        help="Explicit output directory (overrides --run-dir).")
    parser.add_argument("--stride",     type=int, default=None,
                        help="Forecast stride in minutes (overrides config)")
    args = parser.parse_args()

    # Resolve checkpoint and output paths from --run-dir if provided
    if args.run_dir is not None:
        run_dir = Path(args.run_dir)
        checkpoint_path = args.checkpoint or str(run_dir / "checkpoint_best.pkl")
        output_dir_root = args.output_dir or str(run_dir / "outputs" / "test")
    elif args.checkpoint is not None:
        checkpoint_path = args.checkpoint
        output_dir_root = args.output_dir or "outputs/test"
    else:
        parser.error("Provide either --run-dir or --checkpoint.")

    config = load_config(args.config)
    stride = args.stride or config.verify.forecast_stride

    ckpt = load_checkpoint(Path(checkpoint_path))
    ema_params = ckpt["ema_params"]
    num_features = ckpt.get("num_features", len(FEATURE_COLS))

    run_id = ckpt.get("config_hash", "run")
    output_dir = Path(output_dir_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output → %s", output_dir)

    # ------------------------------------------------------------------
    # Load data (all 180 days; splits are re-derived)
    # ------------------------------------------------------------------
    from datetime import date
    end_date_str = ckpt.get("data_end")
    start_date_str = ckpt.get("data_start")

    bars = fetch_bars(
        product="BTC-USD",
        start_date=date.fromisoformat(start_date_str) if start_date_str else None,
        end_date=date.fromisoformat(end_date_str) if end_date_str else None,
        cache_dir=Path(args.data_dir),
    )

    norm_path = Path(args.checkpoint).parent / "norm_state.json"
    if not norm_path.exists():
        raise FileNotFoundError(
            f"Normalisation state not found at {norm_path}. "
            "Run training first to generate norm_state.json."
        )
    norm_state = NormState.load(norm_path)

    train_ds, val_ds, test_ds, _ = prepare_datasets(bars, config)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Forecast loop on test set
    # ------------------------------------------------------------------
    key = jax.random.PRNGKey(0)
    ensembles = run_forecast_loop(
        model, ema_params,
        test_ds.X,
        config, key, stride=stride,
    )

    # Align test targets to strided issue times
    test_y = test_ds.y[::stride]   # (T_eff, H)
    T_eff = min(len(ensembles), len(test_y))
    ensembles = ensembles[:T_eff]
    test_y    = test_y[:T_eff]

    logger.info("Ensembles: %s, test_y: %s", ensembles.shape, test_y.shape)

    # ------------------------------------------------------------------
    # Baselines
    # ------------------------------------------------------------------
    # Climatological baseline from training targets
    train_y = train_ds.y   # (N_train, H)
    model_crps = np.mean(
        np.stack([crps_per_lead(ensembles[t], test_y[t]) for t in range(T_eff)]),
        axis=0,
    )
    clim_crps = _climatology_crps(train_y, test_y)
    pers_crps = _persistence_crps(train_y, test_y)

    # ------------------------------------------------------------------
    # Plots + CSVs
    # ------------------------------------------------------------------
    H = config.data.horizon
    lead_times = list(config.verify.lead_times_for_diag)

    _plot_crps(model_crps, clim_crps, pers_crps, output_dir)
    _plot_rank_hists(ensembles, test_y, lead_times, output_dir)
    _plot_reliability(ensembles, test_y, output_dir)
    _plot_spread_skill(ensembles, test_y, lead_times, output_dir)
    _plot_pinball(ensembles, test_y, output_dir)

    # CSV exports
    import polars as pl
    pl.DataFrame({
        "lead_min": list(range(1, H + 1)),
        "model_crps": model_crps.tolist(),
        "clim_crps": clim_crps.tolist(),
        "pers_crps": pers_crps.tolist(),
    }).write_csv(output_dir / "crps_vs_lead.csv")

    _write_report(
        output_dir, ckpt, config,
        n_train=len(train_ds),
        n_test_forecasts=T_eff,
        stride=stride,
        model_crps=model_crps,
        clim_crps=clim_crps,
        pers_crps=pers_crps,
        lead_times=lead_times,
    )

    logger.info("Verification complete → %s", output_dir)


if __name__ == "__main__":
    main()
