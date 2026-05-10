#!/usr/bin/env python3
"""
Post-training diagnostic plots for the synthetic (Gaussian mixture) model.

Produces, for the EMA checkpoint in `--checkpoint`:
  1. Rank histogram at selected lead times.
  2. Reliability diagram for the binary event "terminal log-return > 0".

Both are saved as PNGs to `--output-dir`.

CRPS is necessary but not sufficient for calibration assessment:
a biased and underdispersed ensemble can cancel in CRPS while the rank
histogram exposes the structural problem.  Run this script after every
synthetic smoke-test.

Usage::

    python scripts/plot_synthetic_diagnostics.py \\
        --checkpoint runs/smoke/checkpoint_best.pkl \\
        --config configs/small.yaml \\
        --output-dir outputs/synthetic
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from btcfm.config import load_config
from btcfm.logging_utils import get_logger
from btcfm.model.flow_matching import create_model, load_checkpoint
from btcfm.model.sampler import generate_ensemble
from btcfm.verification.rank_hist import rank_histogram, flatness_score
from btcfm.verification.reliability import reliability_diagram
from btcfm.verification.crps import crps_per_lead

logger = get_logger(__name__)

NUM_FEATURES = 4   # must match the synthetic training setup
N_EVAL = 200       # number of forecast issue times


def _make_batch_np(seed: int, n: int, L: int, F: int, H: int):
    """Deterministic synthetic batch using numpy (no JAX needed for data)."""
    rng = np.random.default_rng(seed)
    cond = rng.uniform(-1, 1, n)
    features = rng.normal(0, 0.1, (n, L, F))
    features[:, -1, 0] = cond
    mean = np.where(cond > 0, 0.5, -0.5)[:, None] * np.ones((n, H))
    x1 = mean + rng.normal(0, 0.3, (n, H))
    return features.astype(np.float32), x1.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic diagnostic plots")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/small.yaml")
    parser.add_argument("--output-dir", default="outputs/synthetic")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)
    ckpt = load_checkpoint(Path(args.checkpoint))
    ema_params = ckpt["ema_params"]
    n_features = ckpt.get("num_features", NUM_FEATURES)

    H = config.data.horizon
    L = config.data.context_length
    N = config.sampler.n_samples

    model = create_model(
        horizon=H, num_features=n_features, context_len=L,
        encoder_dim=config.model.encoder_dim,
        encoder_heads=config.model.encoder_heads,
        encoder_layers=config.model.encoder_layers,
        encoder_dropout=config.model.encoder_dropout,
        velocity_hidden=config.model.velocity_hidden,
        velocity_layers=config.model.velocity_layers,
        time_emb_dim=config.model.time_emb_dim,
    )

    logger.info("Generating %d forecasts…", N_EVAL)
    features, targets = _make_batch_np(seed=7, n=N_EVAL, L=L, F=n_features, H=H)

    ensembles = []
    key = jax.random.PRNGKey(0)
    for i in range(N_EVAL):
        key, sk = jax.random.split(key)
        ctx = model.apply(ema_params, jnp.array(features[i:i+1]),
                          method=model.encode, deterministic=True)[0]
        ens = generate_ensemble(
            model, ema_params, ctx, sk,
            horizon=H, n_samples=N,
            n_steps=config.sampler.n_steps, solver=config.sampler.solver,
        )
        ensembles.append(np.asarray(ens))

    ensembles_np = np.stack(ensembles)  # (N_EVAL, N, H)

    # ------------------------------------------------------------------
    # Rank histogram — at lead times in config or subset up to H
    # ------------------------------------------------------------------
    lead_times = [lt for lt in config.verify.lead_times_for_diag if lt <= H]
    n_leads = len(lead_times)

    fig, axes = plt.subplots(1, n_leads, figsize=(4 * n_leads, 4))
    if n_leads == 1:
        axes = [axes]

    for ax, lt in zip(axes, lead_times):
        h = lt - 1
        ens_h = ensembles_np[:, :, h]   # (N_EVAL, N)
        obs_h = targets[:, h]            # (N_EVAL,)
        hist = rank_histogram(ens_h, obs_h, normalise=True)
        ks = flatness_score(hist)
        ax.bar(np.arange(len(hist)), hist, width=1.0, color="steelblue", alpha=0.8)
        ax.axhline(1 / len(hist), color="red", ls="--", lw=1.5, label="uniform")
        ax.set_title(f"Lead {lt} min\nKS={ks:.3f}")
        ax.set_xlabel("Rank"); ax.set_ylabel("Frequency")
        ax.legend(fontsize=8)

    plt.suptitle("Rank histograms — synthetic toy (small.yaml)\n"
                 "Flat = calibrated,  U-shape = underdispersed,  Dome = overdispersed",
                 fontsize=10)
    plt.tight_layout()
    rank_path = output_dir / "rank_histograms_synthetic.png"
    fig.savefig(rank_path, dpi=120)
    plt.close(fig)
    logger.info("Rank histogram → %s", rank_path)

    # ------------------------------------------------------------------
    # Reliability diagram — terminal log-return > 0
    # ------------------------------------------------------------------
    event_fn = lambda ens: ens.sum(axis=1) > 0   # terminal r̄ > 0

    rel = reliability_diagram(
        ensembles_np, event_fn, targets,
        n_bins=8, n_bootstrap=200, ci_level=0.9,
    )

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Perfect calibration")
    valid = ~np.isnan(rel["obs_frequency"])
    ax.fill_between(
        rel["bin_centers"][valid],
        rel["ci_lower"][valid], rel["ci_upper"][valid],
        alpha=0.25, color="steelblue", label="90 % bootstrap CI",
    )
    ax.plot(rel["mean_forecast"][valid], rel["obs_frequency"][valid],
            "o-", color="steelblue", lw=2, ms=6, label="Model")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Forecast probability"); ax.set_ylabel("Observed frequency")
    ax.set_title("Reliability: terminal log-return > 0\n(synthetic toy, small.yaml)")
    ax.legend()
    plt.tight_layout()
    rel_path = output_dir / "reliability_synthetic.png"
    fig.savefig(rel_path, dpi=120)
    plt.close(fig)
    logger.info("Reliability diagram → %s", rel_path)

    # ------------------------------------------------------------------
    # CRPS summary
    # ------------------------------------------------------------------
    crps_vals = []
    for i in range(N_EVAL):
        crps_vals.append(crps_per_lead(ensembles_np[i], targets[i]).mean())
    logger.info("Mean CRPS (synthetic): %.4f", np.mean(crps_vals))

    print(f"\n[synthetic diagnostics] saved to {output_dir}/")
    print(f"  rank_histograms_synthetic.png")
    print(f"  reliability_synthetic.png")
    print(f"  mean CRPS = {np.mean(crps_vals):.4f}")


if __name__ == "__main__":
    main()
