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
# What this script does
# ---------------------
# 1. Loads the PyTorch 2.0.0 module (which provides the CUDA 11.7/11.8 stack).
# 2. Installs JAX 0.4.30 with the cuda11_pip extra — the last JAX release
#    with CUDA 11 wheels. JAX 0.4.31+ is CUDA 12 only.
# 3. Installs all remaining runtime deps from requirements.txt.
# 4. Installs the btcfm package itself in editable mode (--no-deps so pip
#    does not overwrite the cuda11 jaxlib with a CPU-only build).
# 5. Runs a sanity import to confirm JAX loads without error.
#    Note: GPU is NOT visible on the login node — the sanity check only
#    confirms the package imports cleanly.  Full GPU verification happens
#    in smoketest_gpu.sh.
#
# PYTHONUSERBASE
# --------------
# Setting PYTHONUSERBASE to your /work directory redirects pip's user-site
# to a project-specific location, avoiding conflicts with your ~/.local.
# This variable must also be exported in every sbatch script that uses the
# same installation.
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# FILL IN: Set your /work path.
# ---------------------------------------------------------------------------
export BTCFM_ROOT=/work/ciad/<lab>/<user>/btcfm   # FILL IN

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
# 1. Load GPU framework modules
#    - pytorch/2.0.0/gpu provides the CUDA 11.7/11.8 runtime libraries.
#    - tensorflow/2.11.0/gpu is kept to match the user's established workflow.
# ---------------------------------------------------------------------------
module load tensorflow/2.11.0/gpu
module load pytorch/2.0.0/gpu
echo "[setup] Modules loaded."

# ---------------------------------------------------------------------------
# 2. Install JAX 0.4.30 with CUDA 11 support
#
#    cuda11_pip   — installs JAX-side CUDA libraries alongside the wheel.
#                   The GPU CUDA runtime itself comes from the pytorch module.
#
#    Do NOT use cuda12 (incompatible with PyTorch 2.0.0's CUDA 11 stack).
#    Do NOT use cuda11_local (assumes standalone nvcc on PATH, not needed).
#    Do NOT omit the -f flag (the cuda wheels are not on PyPI).
# ---------------------------------------------------------------------------
echo "[setup] Installing JAX 0.4.30 (cuda11_pip)..."
pip install --user \
    "jax[cuda11_pip]==0.4.30" \
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
#    --no-deps: do NOT let pip pull jax/jaxlib from PyPI and overwrite the
#    cuda11 jaxlib that was installed in step 2.
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
