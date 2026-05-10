from __future__ import annotations

"""
Pre-flight checks for GPU training on CCuB (and any Slurm site).

Run this module BEFORE any model code at the top of every GPU sbatch job:

    python -m btcfm.runtime.preflight

Failure exits with a non-zero code and a human-readable message so the
Slurm .err log immediately identifies the problem.  The training loop in
``btcfm.train`` also calls these functions internally; running the module
standalone in the sbatch script adds an extra checkpoint that fires before
any heavy imports.

CCuB-specific context
---------------------
CUDA is NOT a standalone module at CCuB.  The GPU CUDA libraries are
provided by loading the PyTorch framework module:

    module load pytorch/2.0.0/gpu

That module bundles CUDA 11.7/11.8.  JAX must therefore be pinned to
0.4.30 (the last release with CUDA 11 wheels) and installed with the
``cuda11_pip`` extra:

    pip install --user "jax[cuda11_pip]==0.4.30" \\
        -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

If the pre-flight assertion fires inside an sbatch job, the most common
cause is a CUDA-version mismatch: the job's LD_LIBRARY_PATH contains CUDA
11 libraries (from the PyTorch module) but the installed jaxlib is a
generic CPU-only build that was accidentally installed after the cuda11_pip
wheel.  Fix: re-run ``scripts/setup_ccub.sh`` on the login node.
"""

import logging
import os
import sys

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Login-node guard
# ---------------------------------------------------------------------------

def check_not_login_node(precision: str = "auto") -> None:
    """
    Abort if on a Slurm LOGIN node with a GPU-targeting precision mode.

    Detection: ``SLURM_CLUSTER_NAME`` or ``SLURM_CONF`` set (we are on a
    Slurm cluster) AND ``SLURM_JOB_ID`` not set (no job allocation, i.e.
    login node).

    CPU-only smoke-tests are permitted on login nodes (precision="fp32"
    with no GPU is fine).  The guard only fires when precision is "auto",
    "bf16", or "fp16" — modes that will attempt GPU computation.
    """
    if os.environ.get("SLURM_JOB_ID", ""):
        return  # Inside a compute job — OK

    on_slurm = bool(
        os.environ.get("SLURM_CLUSTER_NAME")
        or os.environ.get("SLURM_CONF")
        or os.environ.get("SLURM_VERSION")
    )
    if not on_slurm:
        return  # Not a Slurm environment at all (dev machine, CI) — OK

    # On a Slurm login node.
    if precision not in ("fp32",):
        raise SystemExit(
            "\n"
            "ERROR: Running on a Slurm LOGIN node without a job allocation.\n"
            "       GPU training must be submitted via sbatch:\n\n"
            "         sbatch scripts/sbatch/train_default.sh\n\n"
            "       Sanity-check the GPU environment first:\n"
            "         sbatch scripts/sbatch/smoketest_gpu.sh\n\n"
            "       CPU-only smoke-test directly on the login node:\n"
            "         python -m btcfm.train --config configs/small.yaml \\\n"
            "             --synthetic --output-dir runs/smoke\n"
        )


# ---------------------------------------------------------------------------
# GPU assertion and version logging
# ---------------------------------------------------------------------------

def gpu_preflight() -> str:
    """
    Assert that at least one GPU is visible to JAX and log version info.

    Must run BEFORE any model import so the job aborts immediately with a
    clear message rather than silently falling back to CPU.

    Returns
    -------
    str
        ``device_kind`` of the primary GPU, e.g. "NVIDIA V100-SXM2-32GB".
    """
    import jax

    devs = jax.devices()
    gpu_devs = [d for d in devs if d.platform == "gpu"]

    if not gpu_devs:
        raise SystemExit(
            "\n"
            f"ERROR: No GPU visible to JAX.  Devices reported: {devs}\n\n"
            "Most likely causes on CCuB:\n"
            "  1. Module not loaded — add to your sbatch script:\n"
            "         module load pytorch/2.0.0/gpu\n"
            "  2. Wrong JAX wheel — generic CPU jaxlib overwrote the cuda11 build.\n"
            "     Check with:  pip show jaxlib\n"
            "     If the version does not contain 'cuda', re-run setup_ccub.sh.\n"
            "  3. JAX version mismatch — must use 0.4.30 with cuda11_pip:\n"
            "         pip install --user 'jax[cuda11_pip]==0.4.30' \\\n"
            "             -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html\n"
            "  4. LD_LIBRARY_PATH not set — the module load must happen\n"
            "     BEFORE Python starts (i.e. in the sbatch script, not inside Python).\n"
        )

    primary = gpu_devs[0]
    device_kind = getattr(primary, "device_kind", str(primary))

    logger.info("JAX devices    : %s", devs)
    logger.info("JAX version    : %s", jax.__version__)
    logger.info("Primary GPU    : %s", device_kind)

    try:
        import jaxlib
        logger.info("jaxlib version : %s", jaxlib.__version__)
    except Exception:
        pass

    # CUDA platform version string (e.g. "cuda 11.8")
    try:
        cuda_ver = jax.lib.xla_bridge.get_backend().platform_version
        logger.info("CUDA visible   : %s", cuda_ver)
        print(f"CUDA visible: {cuda_ver}", flush=True)
    except Exception:
        pass

    try:
        mem = primary.memory_stats()
        if mem and "bytes_limit" in mem:
            gb = mem["bytes_limit"] / 1e9
            logger.info("GPU memory     : %.1f GB", gb)
    except Exception:
        pass

    return device_kind


# ---------------------------------------------------------------------------
# Precision resolution
# ---------------------------------------------------------------------------

def resolve_precision(precision: str, device_kind: str | None = None):
    """
    Map a precision config string to a JAX dtype.

    Rules
    -----
    "fp32" → jnp.float32  (always)
    "bf16" → jnp.bfloat16 (always)
    "fp16" → jnp.float16  (experimental — see README)
    "auto" → bf16 on Ampere/Hopper (A100, H100, …); fp32 on V100/CPU

    Note: CCuB's GPU nodes are typically V100 (Volta).  Volta does NOT have
    native bf16 throughput, so "auto" will resolve to fp32 on V100.
    bf16 is available on A100/A30/H100 if CCuB gains Ampere nodes later.
    """
    import jax.numpy as jnp

    if precision == "fp32":
        return jnp.float32
    if precision == "bf16":
        return jnp.bfloat16
    if precision == "fp16":
        logger.warning(
            "fp16 is experimental. Flow-matching MSE losses are numerically "
            "stable in fp32/bf16; fp16 on V100 with loss scaling has known "
            "failure modes on long runs. Do not use for the first production run."
        )
        return jnp.float16
    if precision == "auto":
        if device_kind is None:
            return jnp.float32  # CPU or unknown
        bf16_markers = ("A100", "A10", "H100", "H800", "L40", "L4",
                        "RTX 30", "RTX 40", "A30", "A40")
        if any(m in device_kind for m in bf16_markers):
            logger.info(
                "precision=auto → bfloat16 (Ampere/Hopper GPU: %s)", device_kind
            )
            return jnp.bfloat16
        logger.info(
            "precision=auto → float32 (no native bf16 on %s)", device_kind
        )
        return jnp.float32
    raise ValueError(
        f"Unknown precision '{precision}'. Valid: auto, fp32, bf16, fp16."
    )


# ---------------------------------------------------------------------------
# Standalone entry-point  (python -m btcfm.runtime.preflight)
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Run the GPU pre-flight check and print results to stdout.

    Called from sbatch scripts before starting training:
        python -m btcfm.runtime.preflight
    """
    import jax

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    print("=" * 60, flush=True)
    print("btcfm GPU pre-flight check", flush=True)
    print("=" * 60, flush=True)

    print(f"JAX version  : {jax.__version__}", flush=True)
    print(f"Python       : {sys.version.split()[0]}", flush=True)

    device_kind = gpu_preflight()

    print(f"Primary GPU  : {device_kind}", flush=True)
    print("Pre-flight check PASSED.", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
