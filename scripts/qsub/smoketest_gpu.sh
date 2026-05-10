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
#   qsub scripts/qsub/smoketest_gpu.sh
# =============================================================================

#$ -N btcfm-smoke          # Job name
#$ -q gpu                  # GPU queue (2 concurrent job slots; provides 1 GPU)
#$ -pe smp 4               # 4 CPU cores (shared-memory parallel environment)
#$ -l h_rt=00:30:00        # Walltime limit (HH:MM:SS)
#$ -l h_vmem=2G            # Memory per slot: 2G × 4 slots = 8G total
#$ -cwd                    # Run from submission directory

# FILL IN: Absolute path to your logs directory (must exist before job starts).
#$ -o /work/ciad/ma4082ja/poly/poly/logs/
#$ -e /work/ciad/ma4082ja/poly/poly/logs/

# =============================================================================
# Environment — identical to train_default.sh
# =============================================================================
export BTCFM_ROOT=/work/ciad/ma4082ja/poly/poly
export PYTHONUSERBASE=$BTCFM_ROOT
export SMOKE_DIR=$BTCFM_ROOT/smoke/$JOB_ID   # $JOB_ID set by Grid Engine
export LOG_DIR=$BTCFM_ROOT/logs
export XLA_FLAGS="--xla_gpu_deterministic_ops=true"

# =============================================================================
set -euo pipefail

mkdir -p "$SMOKE_DIR" "$LOG_DIR"

echo "============================================================"
echo "btcfm GPU smoke-test (small.yaml, 200 steps)"
echo "  JOB_ID       : $JOB_ID"
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
    --run-id "$JOB_ID" \
    --output-dir "$SMOKE_DIR"

echo ""
echo "============================================================"
echo "Smoke-test PASSED."
echo "  Artefacts : $SMOKE_DIR/"
echo "  JSONL log : $SMOKE_DIR/train.jsonl"
echo "  GPU info  : $SMOKE_DIR/run_metadata.json"
echo ""
echo "Submit the full training run:"
echo "  qsub scripts/qsub/train_default.sh"
echo "============================================================"
