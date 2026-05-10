#!/bin/bash
# =============================================================================
# btcfm — GPU smoke-test using small.yaml (30-minute walltime)
#
# Purpose
# -------
# Run this BEFORE train_default.sh to verify:
#   1. JAX sees the GPU (correct CUDA version, correct jaxlib wheel).
#   2. Module loads and PYTHONUSERBASE are wired up correctly.
#   3. The training loop JIT-compiles and runs for 200 steps without error.
#
# small.yaml uses synthetic data (no download required) and runs in
# < 5 minutes on any GPU.  If this script passes, submit train_default.sh.
#
# Usage:
#   sbatch scripts/sbatch/smoketest_gpu.sh
# =============================================================================

#SBATCH --job-name=btcfm-smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00

# FILL IN: Same partition/account as train_default.sh.
#SBATCH --partition=<FILL IN>
##SBATCH --account=<FILL IN>

# FILL IN: Replace with your /work path.
#SBATCH --output=/work/ciad/<FILL IN: lab/user>/btcfm/logs/%x-%j.out
#SBATCH --error=/work/ciad/<FILL IN: lab/user>/btcfm/logs/%x-%j.err

# =============================================================================
# Environment — identical to train_default.sh
# =============================================================================
export BTCFM_ROOT=/work/ciad/<FILL IN: lab/user>/btcfm
export PYTHONUSERBASE=$BTCFM_ROOT
export SMOKE_DIR=$BTCFM_ROOT/smoke/$SLURM_JOB_ID
export LOG_DIR=$BTCFM_ROOT/logs
export XLA_FLAGS="--xla_gpu_deterministic_ops=true"

# =============================================================================
set -euo pipefail

mkdir -p "$SMOKE_DIR" "$LOG_DIR"

echo "============================================================"
echo "btcfm GPU smoke-test (small.yaml, 200 steps)"
echo "  SLURM_JOB_ID : $SLURM_JOB_ID"
echo "  Node         : $(hostname)"
echo "  BTCFM_ROOT   : $BTCFM_ROOT"
echo "  SMOKE_DIR    : $SMOKE_DIR"
echo "============================================================"
echo ""

cd "$BTCFM_ROOT"

module load python/3.11/anaconda/2024.02
module load cuda/12.1
# cuDNN is bundled inside the jax[cuda12_pip] wheel — no cuDNN module needed.

# GPU pre-flight
echo "[preflight] Running GPU check..."
python -m btcfm.runtime.preflight
echo "[preflight] GPU check passed."
echo ""

# Synthetic training for 200 steps
echo "[smoke] Running synthetic training (small.yaml, 200 steps)..."
python -m btcfm.train \
    --config configs/small.yaml \
    --synthetic \
    --run-id "$SLURM_JOB_ID" \
    --output-dir "$SMOKE_DIR"

echo ""
echo "============================================================"
echo "Smoke-test PASSED."
echo "  Artefacts : $SMOKE_DIR/"
echo "  JSONL log : $SMOKE_DIR/train.jsonl"
echo "  GPU info  : $SMOKE_DIR/run_metadata.json"
echo ""
echo "Submit the full training run:"
echo "  sbatch scripts/sbatch/train_default.sh"
echo "============================================================"
