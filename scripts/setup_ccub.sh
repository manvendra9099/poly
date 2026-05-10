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
# Module stack:
#   python/3.11/anaconda/2024.02   — Python 3.11
#   cuda/12.1                      — CUDA 12 runtime + driver headers
#
# No separate cuDNN module is needed: jax[cuda12_pip] bundles its own cuDNN
# alongside the wheel.  Only the NVIDIA GPU driver (libcuda.so) must be
# present at runtime — that is always the case on GPU compute nodes.
#
# What this script does
# ---------------------
# 1. Loads Python 3.11 + CUDA 12.1.
# 2. Installs JAX with the cuda12_pip extra, which bundles cuDNN and the
#    CUDA runtime libraries into pip packages (no system cuDNN required).
# 3. Installs all remaining runtime deps from requirements.txt.
# 4. Installs the btcfm package in editable mode (--no-deps to protect
#    the cuda12 jaxlib from being overwritten by a CPU-only build).
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
# 1. Load Python 3.11 + CUDA 12
# ---------------------------------------------------------------------------
module load python/3.11/anaconda/2024.02
module load cuda/12.1
echo "[setup] Modules loaded."
echo "        Python : $(python --version)"

# ---------------------------------------------------------------------------
# 2. Install JAX 0.4.30 with CUDA 12 bundled (includes cuDNN via pip packages)
#
#    We pin to JAX 0.4.30 for two reasons:
#      a) The cuda12_pip extra (which bundles cuDNN alongside jaxlib) was
#         present in the 0.4.x series.  In JAX 0.5.0+ the extra was removed
#         and the CUDA install mechanism changed.
#      b) JAX 0.5.0+ pulls flax 0.10+ → orbax-checkpoint ≥ 0.5.0 →
#         tensorstore, which requires Bazel 8+ and GLIBC 2.25.  CentOS 7
#         only has GLIBC 2.17, so tensorstore cannot be installed on this
#         cluster.  JAX 0.4.30 + flax 0.8.x avoid this entirely.
#
#    cuda12_pip bundles the CUDA 12 and cuDNN libraries as pip packages
#    alongside jaxlib — no separate cuDNN module is required.
#    Only the NVIDIA GPU driver (libcuda.so) must exist at runtime, which
#    is always present on GPU compute nodes.
# ---------------------------------------------------------------------------
echo "[setup] Removing any previously installed jax/jaxlib..."
pip uninstall -y jax jaxlib 2>/dev/null || true

# ---------------------------------------------------------------------------
# 3. Install JAX + all runtime deps in ONE pip invocation
#
#    This is critical.  If jax[cuda12_pip]==0.4.30 is installed first and
#    then the rest are installed separately, pip re-resolves from scratch and
#    sees optax 0.2.8 requiring jaxlib>=0.5.3, which forces an upgrade of
#    jaxlib to the CPU 0.7.0 build, overwriting the CUDA wheel.
#
#    With a single pip call, the resolver sees jax==0.4.30's constraint
#    (jaxlib<=0.4.30,>=0.4.27) from the start and is forced to pick
#    optax/flax/orbax versions compatible with that bound.
#
#    orbax-checkpoint<0.5.0  — versions ≥0.5.0 require tensorstore, which
#    needs Bazel 8 / GLIBC 2.25.  CentOS 7 has GLIBC 2.17.
#
#    The -f flag is required: CUDA jaxlib wheels are not on PyPI.
# ---------------------------------------------------------------------------
echo "[setup] Installing JAX 0.4.30 (cuda12_pip) + all runtime deps..."
pip install --user \
    "jax[cuda12_pip]==0.4.30" \
    "flax>=0.8.0,<0.9.0" \
    "orbax-checkpoint<0.5.0" \
    "optax>=0.2.0" \
    "numpy>=1.26" \
    "polars>=1.40,<2" \
    "pyarrow>=14.0,<18.0" \
    "requests>=2.31" \
    "websockets>=12.0" \
    "pyyaml>=6.0" \
    "pydantic>=2.0" \
    "matplotlib>=3.8" \
    "pytest>=8.0" \
    "pytest-asyncio>=0.23" \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# ---------------------------------------------------------------------------
# 4. Install the btcfm package in editable mode
#    --no-deps: do NOT let pip resolve jax/jaxlib from pyproject.toml and
#    overwrite the cuda12 jaxlib that was installed in step 2.
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
    v = jaxlib.__version__
    print(f'jaxlib {v}')
    if 'cuda12' not in v:
        print('WARNING: jaxlib version does not contain cuda12 — GPU may not work')
    else:
        print('cuda12 confirmed in jaxlib version string.')
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
