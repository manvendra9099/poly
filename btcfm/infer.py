from __future__ import annotations

"""
Inference: load a trained model and generate ensemble forecasts.

Usage::

    python -m btcfm.infer \\
        --config configs/small.yaml \\
        --checkpoint runs/latest/checkpoint.pkl \\
        --features path/to/features.parquet \\
        --output path/to/ensemble.parquet
"""

import argparse
import logging
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import polars as pl

from btcfm.config import load_config
from btcfm.logging_utils import get_logger
from btcfm.model.flow_matching import create_model
from btcfm.model.sampler import generate_ensemble

logger = get_logger(__name__)


def load_checkpoint(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def forecast(
    config,
    params,
    features: np.ndarray,
    key: jax.Array,
) -> np.ndarray:
    """
    Generate an ensemble forecast for a single context window.

    Parameters
    ----------
    config   : BTCFMConfig
    params   : model params pytree
    features : (L, F) — feature matrix for the context window
    key      : PRNGKey

    Returns
    -------
    np.ndarray, shape (N, H) — ensemble of log-return paths
    """
    model = create_model(
        horizon=config.data.horizon,
        num_features=features.shape[-1],
        context_len=config.data.context_length,
        encoder_dim=config.model.encoder_dim,
        encoder_heads=config.model.encoder_heads,
        encoder_layers=config.model.encoder_layers,
        encoder_dropout=config.model.encoder_dropout,
        velocity_hidden=config.model.velocity_hidden,
        velocity_layers=config.model.velocity_layers,
        time_emb_dim=config.model.time_emb_dim,
    )

    features_jax = jnp.array(features)[None]  # (1, L, F)
    context = model.apply(
        params, features_jax, method=model.encode, deterministic=True
    )[0]  # (context_dim,)

    ens = generate_ensemble(
        model, params, context, key,
        horizon=config.data.horizon,
        n_samples=config.sampler.n_samples,
        n_steps=config.sampler.n_steps,
        solver=config.sampler.solver,
    )  # (N, H)
    return np.asarray(ens)


def main() -> None:
    parser = argparse.ArgumentParser(description="btcfm inference")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--features", required=True,
                        help="Parquet file with normalised feature matrix (L, F)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    config = load_config(args.config)
    ckpt = load_checkpoint(Path(args.checkpoint))
    params = ckpt["params"]

    features = pl.read_parquet(args.features).to_numpy()
    key = jax.random.PRNGKey(args.seed)
    ens = forecast(config, params, features, key)

    # Save ensemble as Parquet: columns = lead_1, lead_2, ..., lead_H
    horizon = ens.shape[1]
    df = pl.DataFrame(
        {f"lead_{h+1}": ens[:, h].tolist() for h in range(horizon)}
    )
    df.write_parquet(args.output)
    logger.info("Wrote ensemble (%d × %d) → %s", *ens.shape, args.output)


if __name__ == "__main__":
    main()
