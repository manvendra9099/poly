#!/bin/bash
# =============================================================================
# btcfm — test-set verification (decoupled from training)
#
# # DESIGN: Verification is a separate sbatch job rather than appended to the
# training script tail.
#   - A 50k-step training run uses 6–7 h of an 8-h walltime window.
#     Running verification in the same job risks OOM or walltime expiry after
#     a long training run.
#   - Decoupling lets the user iterate on diagnostic code (reliability bins,
#     CRPS horizons) and re-run verification without retraining.
#   - The job can be submitted with --dependency=afterok:<train_job_id> to
#     chain it automatically, or run manually after inspecting training logs.
#
# Usage (auto-submit after training):
#   sbatch --dependency=afterok:<TRAIN_JOB_ID> scripts/sbatch/verify.sh
#
# Usage (manual, after training completes):
#   TRAIN_JOB_ID=<job_id> sbatch scripts/sbatch/verify.sh
#
# The TRAIN_JOB_ID variable selects which run directory to load the
# checkpoint from.  It defaults to the most recent run if not set.
# =============================================================================

#SBATCH --job-name=btcfm-verify
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1          # GPU for faster ensemble generation
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00

# FILL IN: Same partition/account as train_default.sh.
#SBATCH --partition=<FILL IN>
##SBATCH --account=<FILL IN>

# FILL IN: Replace with your /work path.
#SBATCH --output=/work/ciad/<FILL IN: lab/user>/btcfm/logs/%x-%j.out
#SBATCH --error=/work/ciad/<FILL IN: lab/user>/btcfm/logs/%x-%j.err

# =============================================================================
# Environment
# =============================================================================
export BTCFM_ROOT=/work/ciad/<FILL IN: lab/user>/btcfm
export PYTHONUSERBASE=$BTCFM_ROOT
export DATA_DIR=$BTCFM_ROOT/data/coinbase
export LOG_DIR=$BTCFM_ROOT/logs
export XLA_FLAGS="--xla_gpu_deterministic_ops=true"

# TRAIN_JOB_ID: set externally or hardcode the training job's SLURM_JOB_ID.
# If not set, the script will fail with a clear error.
TRAIN_JOB_ID="${TRAIN_JOB_ID:-}"

# =============================================================================
set -euo pipefail

if [ -z "$TRAIN_JOB_ID" ]; then
    echo "ERROR: TRAIN_JOB_ID is not set." >&2
    echo "Usage:" >&2
    echo "  TRAIN_JOB_ID=<job_id> sbatch scripts/sbatch/verify.sh" >&2
    echo "  sbatch --dependency=afterok:<job_id> scripts/sbatch/verify.sh" >&2
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
echo "  SLURM_JOB_ID  : $SLURM_JOB_ID"
echo "  TRAIN_JOB_ID  : $TRAIN_JOB_ID"
echo "  Node          : $(hostname)"
echo "  RUN_DIR       : $RUN_DIR"
echo "  VERIFY_DIR    : $VERIFY_DIR"
echo "============================================================"
echo ""

cd "$BTCFM_ROOT"

module load tensorflow/2.11.0/gpu
module load pytorch/2.0.0/gpu

# GPU pre-flight (verification also uses GPU for ensemble generation)
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
