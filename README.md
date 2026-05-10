# btcfm — Conditional Flow-Matching for Intraday BTC

Probabilistic intraday forecaster for BTC/USD that emits a **full predictive
distribution over the log-return path** at horizons of minutes to a few
hours, calibrated for pricing binary and range contracts.

---

## Mathematical setup

Let $S_t$ be BTC mid-price on a 1-minute grid. Define log-returns
$r_t = \log S_t - \log S_{t-1}$. For a forecast at time $t_0$ with horizon $H$,
the target is the path

$$x := (r_{t_0+1}, \ldots, r_{t_0+H}) \in \mathbb{R}^H$$

conditioned on a context window $c$ of length $L$ minutes containing
multivariate features. The model learns the conditional density $p(x \mid c)$
via **flow matching**.

### Flow-matching objective (linear / rectified interpolant)

Given source $x_0 \sim \mathcal{N}(0, I_H)$ and data sample $x_1 \sim p(x \mid c)$:

$$x_\tau = (1-\tau)\,x_0 + \tau\,x_1, \qquad \tau \sim \mathrm{Unif}(0,1)$$

The conditional vector field is $u_\tau = x_1 - x_0$.  We train
$v_\theta(x_\tau, \tau, c)$ to minimise

$$\mathcal{L}(\theta) = \mathbb{E}_{\tau,\, x_0,\, (x_1,c)}\bigl\|v_\theta(x_\tau, \tau, c) - (x_1 - x_0)\bigr\|^2$$

At inference, integrate $dx/d\tau = v_\theta(x, \tau, c)$ from $\tau=0$ to
$\tau=1$ using Heun's method (50 steps default) and draw $N$ independent
trajectories to form the ensemble.

### Why path-level

A path-valued forecast prices *any* contract (terminal, touch, range) from
the same samples via Monte Carlo. Terminal-only models cannot price
path-dependent contracts.

### Architecture

| Component | Details |
|-----------|---------|
| Context encoder | 4-layer Transformer, dim 128, 4 heads; mean-pool → 128-d context vector |
| Velocity field $v_\theta$ | Sinusoidal $\tau$ embedding → 6-layer MLP, width 256, SiLU |
| Sampler | Heun's method, 50 steps, vmapped over $N$ samples, fully `jit`-compiled |

---

## Repository layout

```
btcfm/
  btcfm/
    data/         coinbase_ws.py, bars.py, historical.py, schema.py
    features/     builders.py, normalise.py
    model/        encoder.py, velocity.py, flow_matching.py, sampler.py
    verification/ crps.py, rank_hist.py, reliability.py, spread_skill.py
    markets/      polymarket.py  (stub — stage 2)
    train.py
    infer.py
    config.py
    logging_utils.py
  configs/        default.yaml, small.yaml
  scripts/        run_live_ingest.py, run_train.py, run_backtest.py
  tests/          test_bars.py, test_features.py,
                  test_flow_matching.py, test_verification.py
  pyproject.toml
```

---

## Quick-start

### 1. Install

```bash
pip install -e ".[dev]"
```

### 2. Run the synthetic smoke-test (no data required)

Trains for 200 steps on a conditional Gaussian mixture.  Full pipeline runs
in < 2 minutes on CPU with the `small.yaml` config.

```bash
python scripts/run_train.py --config configs/small.yaml --synthetic --output-dir runs/smoke
```

Produces `runs/smoke/training_curves.png` with loss and CRPS-vs-step plots.

### 3. Run tests

```bash
pytest                          # all tests
pytest tests/test_bars.py       # bar aggregator only
pytest tests/test_verification.py -v   # verification diagnostics
pytest tests/test_flow_matching.py -v  # end-to-end FM test
```

### 4. Live BTC tick ingest

```bash
python scripts/run_live_ingest.py --output-dir /data/btc/ticks
```

### 5. Train on real data (after populating the cache)

```bash
# Build historical bar cache from downloaded klines, then:
python scripts/run_train.py \
    --config configs/default.yaml \
    --data-dir /data/btc/bars \
    --output-dir runs/exp01
```

---

## Verification diagnostics

All metrics are implemented in `btcfm/verification/`, follow
`properscoring` / `xskillscore` conventions, are pure NumPy functions,
and are unit-tested against analytic Gaussian cases.

| Metric | Function | Notes |
|--------|----------|-------|
| CRPS per lead | `crps_per_lead` | Sample-based, $O(N \log N)$ sorted formula |
| Rank histogram | `rank_histogram_per_lead` | Flatness = calibration |
| Reliability diagram | `reliability_diagram` | Bootstrap confidence bands |
| Spread–skill | `spread_skill_per_lead` | Calibrated ↔ $y = x$ |
| Pinball loss | `pinball_loss_per_lead` | 5/25/50/75/95th quantiles |

---

## Configuration

Two configs are provided:

| Config | Context $L$ | Horizon $H$ | Encoder dim | MLP width | Steps | Use |
|--------|------------|------------|-------------|-----------|-------|-----|
| `small.yaml` | 60 | 30 | 32 | 64 | 200 | CPU smoke-test |
| `default.yaml` | 240 | 60 | 128 | 256 | 50 000 | GPU training |

---

## Running on CCuB (Centre de Calcul de l'Université de Bourgogne)

CCuB uses a **PYTHONUSERBASE + `--user`** install model (no venv/conda),
GPU CUDA libraries provided by the **PyTorch 2.0.0/gpu framework module**
(CUDA 11.7/11.8), and **`/work`-based persistent storage** (no `$SCRATCH`).
All three constraints are reflected in the scripts below — do not substitute
generic cluster patterns.

### 1. One-time environment setup (login node)

SSH into CCuB, `cd` to your project root, then run the setup script once:

```bash
ssh <user>@ccub.u-bourgogne.fr
cd /work/ciad/<lab>/<user>/btcfm

# Edit BTCFM_ROOT in the setup script to match your /work path:
nano scripts/setup_ccub.sh

# Then run it (takes a few minutes on a slow pip mirror):
bash scripts/setup_ccub.sh
```

`setup_ccub.sh` is **idempotent** — safe to re-run after updating
`requirements.txt` or after a JAX wheel issue.  What it does:

1. Loads `tensorflow/2.11.0/gpu` and `pytorch/2.0.0/gpu` (the CUDA 11 stack).
2. Installs **JAX 0.4.30** with the `cuda11_pip` extra.
   JAX 0.4.30 is the last release with CUDA 11 wheels; everything ≥ 0.4.31
   is CUDA 12 only.  The `pytorch/2.0.0/gpu` module bundles CUDA 11.7/11.8,
   so the JAX and CUDA versions must match.
3. Installs all remaining deps from `requirements.txt` (excludes jax/jaxlib).
4. Installs the `btcfm` package with `--no-deps` (protects the cuda11 jaxlib).
5. Confirms JAX imports cleanly (CPU-only; GPU check happens in the sbatch job).

### 2. Edit the `FILL IN` placeholders

The sbatch scripts have clearly marked `# FILL IN:` lines for your
`/work` path, partition name, and account.  Edit them before submitting:

```bash
nano scripts/sbatch/smoketest_gpu.sh
nano scripts/sbatch/train_default.sh
nano scripts/sbatch/verify.sh
```

The `BTCFM_ROOT`, partition, and account must be the same in all three scripts.

### 3. GPU smoke-test (30 minutes, do this first)

```bash
sbatch scripts/sbatch/smoketest_gpu.sh
squeue -u $USER          # confirm it starts
```

Runs `small.yaml` for 200 steps on synthetic data (no download needed).
On success, inspect `$BTCFM_ROOT/smoke/<job_id>/run_metadata.json` to
confirm the GPU type and JAX/CUDA versions.  If it fails, see **Troubleshooting**.

### 4. Full training run (8 hours)

```bash
sbatch scripts/sbatch/train_default.sh
```

Runs `default.yaml` for 50 000 steps.  Artefacts land under `$BTCFM_ROOT`:

```
$BTCFM_ROOT/                          # /work/ciad/<lab>/<user>/btcfm/
  data/coinbase/                      — Parquet bar cache (fetched by training)
  runs/<SLURM_JOB_ID>/
    checkpoint_best.pkl               — best-val EMA checkpoint (use for inference)
    checkpoint_<step>.pkl             — periodic checkpoints every 5 000 steps
    checkpoint_final.pkl              — end-of-training snapshot
    run_metadata.json                 — config, ema_decay, JAX version, GPU, XLA_FLAGS
    norm_state.json                   — normalisation stats (required for inference)
    train.jsonl                       — per-step JSONL log (line-buffered, readable live)
    training_curves.png               — quick loss + CRPS overview
  smoke/<SLURM_JOB_ID>/              — smoke-test artefacts
  logs/<job_name>-<job_id>.{out,err} — Slurm stdout/stderr
```

### 5. Verification (separate job, after training)

Verification is decoupled from training — it runs as its own sbatch job.
This prevents walltime OOM (training uses 6–7 h of the 8-h window), and
lets you re-run verification after iterating on diagnostic code without
retraining.

```bash
# Auto-submit after training completes:
sbatch --dependency=afterok:<TRAIN_JOB_ID> scripts/sbatch/verify.sh

# Or run manually after confirming training succeeded:
TRAIN_JOB_ID=<job_id> sbatch scripts/sbatch/verify.sh
```

Verification reads `checkpoint_best.pkl` from the training run and writes
to `$BTCFM_ROOT/runs/<TRAIN_JOB_ID>/outputs/test/`:

```
outputs/test/
  REPORT.md               — calibration report (model vs climatology vs persistence)
  crps_vs_lead.{png,csv}
  rank_histograms.png
  reliability.png
  spread_skill.png
  pinball_loss.png
```

### 6. Plotting the JSONL training log

The `train.jsonl` file is line-buffered and readable while the job runs:

```bash
# After (or during) training:
python scripts/plot_run.py \
    --run-dir $BTCFM_ROOT/runs/<SLURM_JOB_ID>

# Plots land in $BTCFM_ROOT/runs/<SLURM_JOB_ID>/plots/
```

Produces: `training_loss.png`, `val_loss.png`, `lr_schedule.png`,
`grad_norm.png`, `steps_per_sec.png`.

> **Throughput note.** If `nvidia-smi` shows GPU utilisation below 80 %
> during a real-data run, the single-worker data prefetch loader is the
> bottleneck. Revisit before citing throughput numbers.

### Mixed precision

CCuB GPU nodes are typically V100 (Volta). The `precision: auto` setting
in `default.yaml` maps hardware to dtype at runtime:

| GPU  | `precision: auto` | Notes |
|------|--------------------|-------|
| V100 | fp32 | Volta has no native bf16 throughput; fp16 + loss scaling is unreliable on long runs |
| A100 | bf16 | Ampere — native bf16 with no loss scaling needed |
| CPU  | fp32 | Smoke-tests only |

Override by setting `train.precision: fp32 | bf16 | auto` in the YAML.
`fp16` is accepted by the schema but flagged as experimental — do not use
it for the first production run.

### Determinism

All sbatch scripts export:
```bash
export XLA_FLAGS="--xla_gpu_deterministic_ops=true"
```

This costs ≈ 10–20 % throughput. Leave it on for the reported run.
Remove it for exploratory runs where exact reproducibility is not required.

### Config note

**`small.yaml` results are not citable as calibration evidence.**
The smoke-test model is undertrained and underdispersed by design.
All calibration claims must reference `default.yaml` numbers from a full
GPU run with the held-out test window (days 165–180 of the 180-day window).
See `outputs/baselines/synthetic_small/BASELINE.md` for the synthetic-toy floor.

### Troubleshooting

**Pre-flight fails: "No GPU visible to JAX"**

The most common cause is a JAX/CUDA version mismatch.

1. Check what jaxlib is installed:
   ```bash
   module load pytorch/2.0.0/gpu
   pip show jaxlib
   ```
   The version string must contain `cuda11` (e.g. `0.4.30+cuda11.cudnn86`).
   If it shows a CPU-only version (`0.4.30`), the cuda11 jaxlib was
   overwritten — re-run `setup_ccub.sh`.

2. Confirm the module is loaded in the sbatch script:
   ```bash
   module load pytorch/2.0.0/gpu   # must precede python calls
   ```

3. Confirm `PYTHONUSERBASE` is exported before `python -m btcfm.runtime.preflight`.
   Without it, Python looks in `~/.local` instead of `$BTCFM_ROOT`.

**JAX installed with wrong CUDA version**

Symptoms: import error mentioning `libcuda.so` or `libcudart.so`.

Fix: re-run `setup_ccub.sh` and confirm it uses `cuda11_pip`, not `cuda12`:
```bash
pip install --user "jax[cuda11_pip]==0.4.30" \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

If pip pulled `cuda12` (e.g. you ran `pip install jax` without the pin),
the CUDA 12 jaxlib will fail to find the CUDA 11 libraries from the
PyTorch module.

**GPU utilisation below 80 %**

The single-worker prefetch loader is the bottleneck.  For real-data runs,
inspect whether `WindowDataset.sample_batch` is slow (check with `cProfile`).
Increasing the queue depth in `_PrefetchLoader(maxsize=4)` in `btcfm/train.py`
may help before investing in a more sophisticated loader.

---

## Stage 2 (not yet built)

`btcfm/markets/polymarket.py` is a stub.  It will pull live Polymarket BTC
markets, compute model-implied probabilities by Monte Carlo over ensemble
paths, and return a `(market_id, polymarket_prob, model_prob, edge_bps)` table.
