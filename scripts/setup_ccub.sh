#!/bin/bash
# =============================================================================
# btcfm — one-shot environment setup for CCuB
#
# Run ONCE on the login node before submitting any sbatch jobs.
# Idempotent: safe to run again after updating requirements.
#
# Usage:
#   1. Edit BTCFM_ROOT below to match your /work path.
#   2. bash scripts/setup_ccub.sh
#
# Module stack (no tensorflow or pytorch module required):
#   python/3.11/anaconda/2024.02   — Python 3.11
#   cuda/11.4.4                    — CUDA 11.4 runtime
#   cudnn/8.2.4.15-11.4            — cuDNN 8.2 (paired with CUDA 11.4)
#
# What this script does
# ---------------------
# 1. Loads Python 3.11 + CUDA 11.4 + cuDNN 8.2.
# 2. Installs JAX 0.4.30 with the cuda11.cudnn82 jaxlib wheel — the last JAX
#    release with CUDA 11 wheels (0.4.31+ is CUDA 12 only).
# 3. Installs all remaining runtime deps from requirements.txt.
# 4. Installs the btcfm package in editable mode (--no-deps to protect the
#    cuda11 jaxlib from being overwritten by a CPU-only build).
# 5. Sanity-imports JAX to confirm the install is clean.
#    GPU is NOT visible on the login node — full GPU check happens in
#    smoketest_gpu.sh.
#
# PYTHONUSERBASE
# --------------
# Setting PYTHONUSERBASE to your /work directory redirects pip's user-site
# to a project-specific location, avoiding conflicts with your ~/.local.
# This variable must also be exported in every sbatch script.
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# FILL IN: Set your /work path.
# ---------------------------------------------------------------------------
export BTCFM_ROOT=/work/ciad/ma4082ja/poly/poly 

# All Python packages land here (pip --user redirect)
export PYTHONUSERBASE=$BTCFM_ROOT

echo "============================================================"
echo "btcfm CCuB environment setup"
echo "  BTCFM_ROOT     : $BTCFM_ROOT"
echo "  PYTHONUSERBASE : $PYTHONUSERBASE"
echo "============================================================"
echo ""

# Confirm we are in the project root
if [ ! -f "pyproject.toml" ]; then
    echo "ERROR: run this script from the btcfm project root." >&2
    echo "  cd $BTCFM_ROOT && bash scripts/setup_ccub.sh" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# 1. Load Python 3.11 + CUDA 11.4 + cuDNN 8.2
#
#    We do NOT load tensorflow/2.11.0/gpu or pytorch/2.0.0/gpu — those
#    modules force Python 3.10 and we don't need them.  JAX only needs the
#    CUDA runtime from the module system.
# ---------------------------------------------------------------------------
module load python/3.11/anaconda/2024.02
module load cuda/11.4.4
module load cudnn/8.2.4.15-11.4
echo "[setup] Modules loaded."
echo "        Python : $(python --version)"
echo "        CUDA   : $(nvcc --version 2>/dev/null | grep 'release' || echo 'nvcc not on PATH (OK — runtime is loaded)')"

# ---------------------------------------------------------------------------
# 2. Install JAX 0.4.30 + cuda11.cudnn82 jaxlib
#
#    The suffix +cuda11.cudnn82 matches the cuDNN 8.2 module loaded above.
#    Do NOT use jax[cuda11_pip] — that extra was removed from JAX >= 0.4.15.
#    Do NOT omit the -f flag — the CUDA wheels are not on PyPI.
# ---------------------------------------------------------------------------
echo "[setup] Installing JAX 0.4.30 + cuda11.cudnn82 jaxlib..."
pip install --user jax==0.4.30 \
    "jaxlib==0.4.30+cuda11.cudnn82" \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# ---------------------------------------------------------------------------
# 3. Install remaining runtime dependencies
#    requirements.txt intentionally excludes jax/jaxlib to avoid overwriting
#    the cuda11 build with a CPU-only wheel.
# ---------------------------------------------------------------------------
echo "[setup] Installing runtime dependencies..."
pip install --user -r requirements.txt

# ---------------------------------------------------------------------------
# 4. Install the btcfm package in editable mode
#    --no-deps: do NOT let pip resolve jax/jaxlib from pyproject.toml and
#    overwrite the cuda11 jaxlib that was installed in step 2.
# ---------------------------------------------------------------------------
echo "[setup] Installing btcfm (editable, no-deps)..."
pip install --user --no-deps -e .

# ---------------------------------------------------------------------------
# 5. Sanity check: confirm JAX imports cleanly
#    GPU is not visible on the login node — this only tests the install.
# ---------------------------------------------------------------------------
echo "[setup] Sanity import check..."
python -c "
import jax
print(f'JAX {jax.__version__} imported OK')
try:
    import jaxlib
    print(f'jaxlib {jaxlib.__version__}')
    if 'cuda11' not in jaxlib.__version__:
        print('WARNING: jaxlib version does not contain cuda11 — GPU may not work')
except Exception as e:
    print(f'jaxlib import warning: {e}')
"

echo ""
echo "============================================================"
echo "Setup complete."
echo ""
echo "Next steps:"
echo "  1. Edit the FILL IN fields in the sbatch scripts:"
echo "       nano scripts/sbatch/smoketest_gpu.sh"
echo "       nano scripts/sbatch/train_default.sh"
echo ""
echo "  2. Run the 30-min GPU smoke-test (sanity check before full run):"
echo "       sbatch scripts/sbatch/smoketest_gpu.sh"
echo ""
echo "  3. Once the smoke-test passes, submit the full training run:"
echo "       sbatch scripts/sbatch/train_default.sh"
echo "============================================================"
