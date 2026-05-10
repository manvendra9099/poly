#!/bin/bash
# =============================================================================
# btcfm — full training run on a single GPU
# Config: default.yaml  |  50 000 steps  |  8-hour walltime
#
# Prerequisites
# -------------
# Run setup_ccub.sh on the login node FIRST:
#   bash scripts/setup_ccub.sh
#
# Then fill in every "FILL IN" line below before submitting:
#   sbatch scripts/sbatch/train_default.sh
#
# After training completes, run verification separately:
#   sbatch scripts/sbatch/verify.sh
# (or see README for the --dependency pattern)
# =============================================================================

#SBATCH --job-name=btcfm-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00

# FILL IN: GPU partition name. Run `sinfo` on the login node to list partitions.
#SBATCH --partition=<FILL IN: e.g. gpu or gpu_p100>

# FILL IN: Account / project code (required on most CCuB allocations).
##SBATCH --account=<FILL IN: your project code>

# FILL IN: Replace with your /work path.
#SBATCH --output=/work/ciad/<FILL IN: lab/user>/btcfm/logs/%x-%j.out
#SBATCH --error=/work/ciad/<FILL IN: lab/user>/btcfm/logs/%x-%j.err

# =============================================================================
# Environment — must match setup_ccub.sh exactly
# =============================================================================
# FILL IN: Your /work path (same as BTCFM_ROOT in setup_ccub.sh).
export BTCFM_ROOT=/work/ciad/<FILL IN: lab/user>/btcfm
export PYTHONUSERBASE=$BTCFM_ROOT

# Subdirectories (do not change these)
export DATA_DIR=$BTCFM_ROOT/data/coinbase
export RUN_DIR=$BTCFM_ROOT/runs/$SLURM_JOB_ID
export LOG_DIR=$BTCFM_ROOT/logs

# XLA determinism — costs ~10–20 % throughput; required for the reported run.
# Remove for exploratory runs where exact bit-reproducibility is not needed.
export XLA_FLAGS="--xla_gpu_deterministic_ops=true"

# =============================================================================
set -euo pipefail

mkdir -p "$RUN_DIR" "$LOG_DIR"

echo "============================================================"
echo "btcfm training job"
echo "  SLURM_JOB_ID : $SLURM_JOB_ID"
echo "  Node         : $(hostname)"
echo "  BTCFM_ROOT   : $BTCFM_ROOT"
echo "  DATA_DIR     : $DATA_DIR"
echo "  RUN_DIR      : $RUN_DIR"
echo "  XLA_FLAGS    : $XLA_FLAGS"
echo "============================================================"
echo ""

cd "$BTCFM_ROOT"

# ---------------------------------------------------------------------------
# Load GPU runtime (provides CUDA 11.7/11.8 via PyTorch 2.0.0)
# Do NOT change these module names — they must match setup_ccub.sh.
# ---------------------------------------------------------------------------
module load tensorflow/2.11.0/gpu
module load pytorch/2.0.0/gpu

# ---------------------------------------------------------------------------
# GPU pre-flight check
# Aborts the job immediately if JAX cannot see the GPU.
# If this fails, check:
#   - pip show jaxlib    (must show a cuda11 build)
#   - module list        (pytorch/2.0.0/gpu must be loaded)
#   See README → Troubleshooting for details.
# ---------------------------------------------------------------------------
echo "[preflight] Running GPU check..."
python -m btcfm.runtime.preflight
echo "[preflight] GPU check passed."
echo ""

# ---------------------------------------------------------------------------
# Training
# --run-id $SLURM_JOB_ID tags every checkpoint and JSONL record.
# --output-dir writes all artefacts to $RUN_DIR (on /work, never $HOME).
# ---------------------------------------------------------------------------
echo "[training] Starting training (50 000 steps, default.yaml)..."
python -m btcfm.train \
    --config configs/default.yaml \
    --run-id "$SLURM_JOB_ID" \
    --data-dir "$DATA_DIR" \
    --output-dir "$RUN_DIR"

echo ""
echo "============================================================"
echo "Training complete."
echo "  Checkpoints : $RUN_DIR/"
echo "  JSONL log   : $RUN_DIR/train.jsonl"
echo "  Metadata    : $RUN_DIR/run_metadata.json"
echo ""
echo "Run verification:"
echo "  TRAIN_JOB_ID=$SLURM_JOB_ID sbatch scripts/sbatch/verify.sh"
echo ""
echo "Or with dependency (auto-submits after this job):"
echo "  sbatch --dependency=afterok:$SLURM_JOB_ID scripts/sbatch/verify.sh"
echo "============================================================"
