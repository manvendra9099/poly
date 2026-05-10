#!/bin/bash
# =============================================================================
# btcfm — test-set verification (decoupled from training)
#
# DESIGN: Verification is a separate qsub job rather than appended to the
# training script tail.
#   - A 50k-step training run uses 6–7 h of an 8-h walltime window.
#     Running verification in the same job risks OOM or walltime expiry.
#   - Decoupling lets you iterate on diagnostic code (reliability bins,
#     CRPS horizons) and re-run verification without retraining.
#
# Usage (auto-chain — submits now, runs only after train job finishes):
#   qsub -hold_jid <TRAIN_JOB_ID> -v TRAIN_JOB_ID=<TRAIN_JOB_ID> \
#        scripts/qsub/verify.sh
#
# Usage (manual, after training completes):
#   TRAIN_JOB_ID=<job_id> qsub scripts/qsub/verify.sh
#
# TRAIN_JOB_ID selects which run directory to load the checkpoint from.
# =============================================================================

#$ -N btcfm-verify         # Job name
#$ -q gpu                  # GPU queue (for faster ensemble generation)
#$ -pe smp 4               # 4 CPU cores
#$ -l h_rt=01:00:00        # Walltime limit (HH:MM:SS)
#$ -l h_vmem=4G            # Memory per slot: 4G × 4 slots = 16G total
#$ -cwd                    # Run from submission directory

# FILL IN: Absolute path to your logs directory (must exist before job starts).
#$ -o /work/ciad/ma4082ja/poly/poly/logs/
#$ -e /work/ciad/ma4082ja/poly/poly/logs/

# =============================================================================
# Environment
# =============================================================================
export BTCFM_ROOT=/work/ciad/ma4082ja/poly/poly
export PYTHONUSERBASE=$BTCFM_ROOT
export DATA_DIR=$BTCFM_ROOT/data/coinbase
export LOG_DIR=$BTCFM_ROOT/logs
export XLA_FLAGS="--xla_gpu_deterministic_ops=true"

# TRAIN_JOB_ID: passed via -v flag (qsub -v TRAIN_JOB_ID=...) or set in env.
# If not set, the script fails with a clear error.
TRAIN_JOB_ID="${TRAIN_JOB_ID:-}"

# =============================================================================
set -euo pipefail

if [ -z "$TRAIN_JOB_ID" ]; then
    echo "ERROR: TRAIN_JOB_ID is not set." >&2
    echo "Usage:" >&2
    echo "  TRAIN_JOB_ID=<job_id> qsub scripts/qsub/verify.sh" >&2
    echo "  qsub -hold_jid <job_id> -v TRAIN_JOB_ID=<job_id> scripts/qsub/verify.sh" >&2
    exit 1
fi

export RUN_DIR=$BTCFM_ROOT/runs/$TRAIN_JOB_ID
export VERIFY_DIR=$RUN_DIR/outputs/test

if [ ! -f "$RUN_DIR/checkpoint_best.pkl" ]; then
    echo "ERROR: checkpoint_best.pkl not found in $RUN_DIR" >&2
    echo "Did training job $TRAIN_JOB_ID complete successfully?" >&2
    exit 1
fi

mkdir -p "$VERIFY_DIR" "$LOG_DIR"

echo "============================================================"
echo "btcfm test-set verification"
echo "  JOB_ID        : $JOB_ID"
echo "  TRAIN_JOB_ID  : $TRAIN_JOB_ID"
echo "  Node          : $(hostname)"
echo "  RUN_DIR       : $RUN_DIR"
echo "  VERIFY_DIR    : $VERIFY_DIR"
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

echo "[verify] Running test-set verification..."
python scripts/run_verification.py \
    --run-dir "$RUN_DIR" \
    --config configs/default.yaml \
    --data-dir "$DATA_DIR" \
    --stride 1

echo ""
echo "============================================================"
echo "Verification complete."
echo "  Report      : $VERIFY_DIR/REPORT.md"
echo "  CRPS plot   : $VERIFY_DIR/crps_vs_lead.png"
echo "  Rank hists  : $VERIFY_DIR/rank_histograms.png"
echo "  Reliability : $VERIFY_DIR/reliability.png"
echo "  Spread-skill: $VERIFY_DIR/spread_skill.png"
echo "  Pinball     : $VERIFY_DIR/pinball_loss.png"
echo "============================================================"
