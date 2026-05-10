# Synthetic small-config baseline

Generated from `runs/smoke/checkpoint_best.pkl` (200 steps, `small.yaml`)
using `scripts/plot_synthetic_diagnostics.py`.

These numbers are the **floor** against which real-data results will be compared.
An underdispersed and undertrained model is expected at this scale; this is intentional.

## Numbers (session 2, 2026-05-10)

| Metric | Value | Note |
|--------|-------|------|
| Mean CRPS (200 synthetic forecasts) | 0.3696 | vs climatological ≈ 0.34 |
| Rank-histogram KS range | ~0.14 – 0.21 | U-shape: underdispersed |
| Reliability (terminal r > 0) | pinned near 0.5 | no calibration structure |

## Interpretation

The model has learned to condition on the mixture signal (beats climatology
at step 400 in the CI test) but is heavily underdispersed: the rank histogram
shows a U-shape, meaning the observed value falls outside the ensemble more
often than expected. The reliability diagram is flat near 0.5 because the
50-step Heun sampler hasn't had enough training to sharpen the distribution.

A real-data GPU run with `default.yaml` (50 000 steps, encoder dim 128, 6-layer
velocity MLP, ensemble size 1000) should produce a substantially flatter rank
histogram and tighter CRPS.

## Config note

`small.yaml` results are **not citable as calibration evidence**.
All calibration claims must reference `default.yaml` numbers from a full GPU run.
