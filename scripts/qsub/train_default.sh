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
# Fill in every "FILL IN" line below, then submit:
#   qsub scripts/qsub/train_default.sh
#
# After training completes, run verification separately:
#   TRAIN_JOB_ID=<job_id> qsub scripts/qsub/verify.sh
# (or see the completion message at the end of this script for the
#  -hold_jid one-liner that chains the jobs automatically)
# =============================================================================

#$ -N btcfm-train          # Job name
#$ -q gpu                  # GPU queue — -q gpu is all that is needed to get a GPU.
                           # 8h walltime exceeds webern07's 2h cap, so the scheduler
                           # will route this to an A100 node automatically.
#$ -pe smp 8               # 8 CPU cores (shared-memory parallel environment)
#$ -l h_rt=08:00:00        # Walltime limit (HH:MM:SS)
#$ -l h_vmem=4G            # Memory per slot: 4G × 8 slots = 32G total
#$ -cwd                    # Run from submission directory

# FILL IN: Absolute path to your logs directory (must exist before job starts).
#$ -o /work/ciad/ma4082ja/poly/poly/logs/
#$ -e /work/ciad/ma4082ja/poly/poly/logs/

# =============================================================================
# Environment — must match setup_ccub.sh exactly
# =============================================================================
export BTCFM_ROOT=/work/ciad/ma4082ja/poly/poly
export PYTHONUSERBASE=$BTCFM_ROOT

# Subdirectories (do not change these)
export DATA_DIR=$BTCFM_ROOT/data/coinbase
export RUN_DIR=$BTCFM_ROOT/runs/$JOB_ID   # $JOB_ID set by Grid Engine
export LOG_DIR=$BTCFM_ROOT/logs

# XLA determinism — costs ~10–20 % throughput; required for the reported run.
# Remove for exploratory runs where exact bit-reproducibility is not needed.
export XLA_FLAGS="--xla_gpu_deterministic_ops=true"

# =============================================================================
set -euo pipefail

mkdir -p "$RUN_DIR" "$LOG_DIR"

echo "============================================================"
echo "btcfm training job"
echo "  JOB_ID       : $JOB_ID"
echo "  Node         : $(hostname)"
echo "  BTCFM_ROOT   : $BTCFM_ROOT"
echo "  DATA_DIR     : $DATA_DIR"
echo "  RUN_DIR      : $RUN_DIR"
echo "  XLA_FLAGS    : $XLA_FLAGS"
echo "============================================================"
echo ""

cd "$BTCFM_ROOT"

module load python/3.11/anaconda/2024.02
module load cuda/12.1
# cuDNN is bundled inside the jax[cuda12_pip] wheel — no cuDNN module needed.

echo "[preflight] Running GPU check..."
python -m btcfm.runtime.preflight
echo "[preflight] GPU check passed."
echo ""

echo "[training] Starting training (50 000 steps, default.yaml)..."
python -m btcfm.train \
    --config configs/default.yaml \
    --run-id "$JOB_ID" \
    --data-dir "$DATA_DIR" \
    --output-dir "$RUN_DIR"

echo ""
echo "============================================================"
echo "Training complete."
echo "  Checkpoints : $RUN_DIR/"
echo "  JSONL log   : $RUN_DIR/train.jsonl"
echo "  Metadata    : $RUN_DIR/run_metadata.json"
echo ""
echo "Run verification (manual):"
echo "  TRAIN_JOB_ID=$JOB_ID qsub scripts/qsub/verify.sh"
echo ""
echo "Or chain automatically (submits verify now, runs after this job):"
echo "  qsub -hold_jid $JOB_ID -v TRAIN_JOB_ID=$JOB_ID scripts/qsub/verify.sh"
echo "============================================================"
