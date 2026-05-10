#!/usr/bin/env python3
"""
Post-hoc plots from a btcfm JSONL training log.

Reads ``train.jsonl`` written by ``btcfm.train`` and produces PNGs:
  training_loss.png    — per-step FM loss (log scale)
  val_loss.png         — validation loss at EMA checkpoint steps
  lr_schedule.png      — learning rate vs step
  grad_norm.png        — gradient L2 norm vs step
  steps_per_sec.png    — throughput (steps/s) vs step

Usage::

    python scripts/plot_run.py --log-file $RUN_DIR/train.jsonl \\
        --output-dir $RUN_DIR/plots/

    # Or point to a run directory directly (infers train.jsonl):
    python scripts/plot_run.py --run-dir $RUN_DIR
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load_jsonl(path: Path) -> tuple[list[dict], list[dict]]:
    """
    Load a JSONL log file and split into train and val records.

    Returns
    -------
    (train_records, val_records)
    """
    train_records: list[dict] = []
    val_records:   list[dict] = []
    skipped = 0

    with open(path) as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            phase = rec.get("phase", "train")
            if phase == "train":
                train_records.append(rec)
            elif phase == "val":
                val_records.append(rec)

    if skipped:
        print(f"[plot_run] Warning: skipped {skipped} malformed lines in {path}")

    return train_records, val_records


def _smooth(values: list[float], window: int = 50) -> np.ndarray:
    """Simple box-car smoothing."""
    arr = np.asarray(values, dtype=float)
    if len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    padded = np.pad(arr, (window // 2, window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: len(arr)]


def plot_training_loss(
    train_records: list[dict], output_dir: Path, smooth_window: int = 50
) -> None:
    steps  = [r["step"]  for r in train_records if "loss" in r]
    losses = [r["loss"]  for r in train_records if "loss" in r]
    if not steps:
        print("[plot_run] No training loss records found — skipping.")
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(steps, losses, lw=0.4, alpha=0.4, color="steelblue", label="raw")
    if len(losses) >= smooth_window:
        ax.plot(steps, _smooth(losses, smooth_window), lw=1.5, color="steelblue",
                label=f"smoothed (w={smooth_window})")
    ax.set_xlabel("Step")
    ax.set_ylabel("FM loss")
    ax.set_title("Training loss")
    ax.set_yscale("log")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = output_dir / "training_loss.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[plot_run] → {path}")


def plot_val_loss(val_records: list[dict], output_dir: Path) -> None:
    steps  = [r["step"]      for r in val_records if "val_loss" in r]
    losses = [r["val_loss"]  for r in val_records if "val_loss" in r]
    crps   = [r.get("val_crps_terminal", float("nan")) for r in val_records if "val_loss" in r]
    if not steps:
        print("[plot_run] No validation loss records found — skipping.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(steps, losses, "o-", ms=4, lw=1.5, color="steelblue")
    axes[0].set_xlabel("Step"); axes[0].set_ylabel("Val loss")
    axes[0].set_title("Validation loss (EMA params)")
    axes[0].grid(True, alpha=0.3)

    crps_valid = [c for c in crps if not np.isnan(c)]
    steps_crps = [s for s, c in zip(steps, crps) if not np.isnan(c)]
    if crps_valid:
        axes[1].plot(steps_crps, crps_valid, "o-", ms=4, lw=1.5, color="darkorange")
        axes[1].set_xlabel("Step"); axes[1].set_ylabel("CRPS (one sample)")
        axes[1].set_title("Val CRPS (terminal lead, EMA params)")
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = output_dir / "val_loss.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[plot_run] → {path}")


def plot_lr_schedule(train_records: list[dict], output_dir: Path) -> None:
    steps = [r["step"] for r in train_records if "lr" in r]
    lrs   = [r["lr"]   for r in train_records if "lr" in r]
    if not steps:
        print("[plot_run] No learning rate records found — skipping.")
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(steps, lrs, lw=1.2, color="darkorange")
    ax.set_xlabel("Step"); ax.set_ylabel("Learning rate")
    ax.set_title("LR schedule (warmup + cosine decay)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = output_dir / "lr_schedule.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[plot_run] → {path}")


def plot_grad_norm(
    train_records: list[dict], output_dir: Path, smooth_window: int = 50
) -> None:
    steps = [r["step"]      for r in train_records if "grad_norm" in r]
    norms = [r["grad_norm"] for r in train_records if "grad_norm" in r]
    if not steps:
        print("[plot_run] No grad_norm records found — skipping.")
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(steps, norms, lw=0.4, alpha=0.4, color="firebrick", label="raw")
    if len(norms) >= smooth_window:
        ax.plot(steps, _smooth(norms, smooth_window), lw=1.5, color="firebrick",
                label=f"smoothed (w={smooth_window})")
    ax.set_xlabel("Step"); ax.set_ylabel("Gradient L2 norm")
    ax.set_title("Gradient norm")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = output_dir / "grad_norm.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[plot_run] → {path}")


def plot_steps_per_sec(
    train_records: list[dict], output_dir: Path, smooth_window: int = 50
) -> None:
    # Skip the first step (includes JIT compile time)
    records = [r for r in train_records if "steps_per_sec" in r and r["step"] > 1]
    if not records:
        print("[plot_run] No throughput records found — skipping.")
        return

    steps = [r["step"]          for r in records]
    sps   = [r["steps_per_sec"] for r in records]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(steps, sps, lw=0.4, alpha=0.4, color="seagreen", label="raw")
    if len(sps) >= smooth_window:
        ax.plot(steps, _smooth(sps, smooth_window), lw=1.5, color="seagreen",
                label=f"smoothed (w={smooth_window})")
    ax.set_xlabel("Step"); ax.set_ylabel("Steps / second")
    ax.set_title("Training throughput (step 1 excluded — includes JIT compile)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = output_dir / "steps_per_sec.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[plot_run] → {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot btcfm JSONL training log")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--log-file", help="Path to train.jsonl")
    source.add_argument("--run-dir",  help="Run directory containing train.jsonl")
    parser.add_argument("--output-dir", default=None,
                        help="Directory for output PNGs (default: alongside log file)")
    parser.add_argument("--smooth", type=int, default=50,
                        help="Smoothing window size (default 50)")
    args = parser.parse_args()

    # Resolve log file path
    if args.run_dir:
        log_path = Path(args.run_dir) / "train.jsonl"
        default_out = Path(args.run_dir) / "plots"
    else:
        log_path = Path(args.log_file)
        default_out = log_path.parent / "plots"

    if not log_path.exists():
        print(f"ERROR: JSONL log not found: {log_path}", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else default_out
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[plot_run] Loading: {log_path}")
    train_records, val_records = _load_jsonl(log_path)
    print(f"[plot_run] {len(train_records)} train records, {len(val_records)} val records")
    print(f"[plot_run] Output: {output_dir}/")

    plot_training_loss(train_records, output_dir, smooth_window=args.smooth)
    plot_val_loss(val_records, output_dir)
    plot_lr_schedule(train_records, output_dir)
    plot_grad_norm(train_records, output_dir, smooth_window=args.smooth)
    plot_steps_per_sec(train_records, output_dir, smooth_window=args.smooth)

    print(f"\n[plot_run] All plots written to {output_dir}/")


if __name__ == "__main__":
    main()
