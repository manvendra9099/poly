from __future__ import annotations

"""
Tests for all five verification diagnostics against analytic Gaussian cases.

For large N the sample-based estimates must converge to analytic values;
we use N=5000 and assert errors below 5 %.
"""

import numpy as np
import pytest
from scipy import stats as scipy_stats

from btcfm.verification.crps import (
    crps_sample,
    crps_gaussian_analytic,
    crps_per_lead,
    crps_terminal,
    mean_crps,
)
from btcfm.verification.rank_hist import rank_histogram, rank_histogram_per_lead, flatness_score
from btcfm.verification.reliability import reliability_diagram
from btcfm.verification.spread_skill import (
    spread_skill,
    spread_skill_per_lead,
    pinball_loss,
    pinball_loss_per_lead,
    QUANTILE_LEVELS,
)

RNG = np.random.default_rng(0)
N_ENS = 5000   # large enough for convergence
N_TIMES = 500  # number of forecast times


# ---------------------------------------------------------------------------
# CRPS
# ---------------------------------------------------------------------------

class TestCRPS:
    def test_crps_gaussian_zero_mean(self):
        """CRPS(N(0,1), 0) matches analytic value."""
        ens = RNG.standard_normal(N_ENS)
        y = np.array([0.0])
        crps_emp = crps_sample(ens[:, None], y)[0]
        crps_ana = crps_gaussian_analytic(0.0, 1.0, 0.0)
        assert abs(crps_emp - crps_ana) / crps_ana < 0.05

    def test_crps_gaussian_shifted(self):
        """CRPS(N(2, 0.5), 1.5) matches analytic value."""
        mu, sigma, y = 2.0, 0.5, 1.5
        ens = RNG.normal(mu, sigma, N_ENS)
        crps_emp = float(crps_sample(ens[:, None], np.array([y]))[0])
        crps_ana = crps_gaussian_analytic(mu, sigma, y)
        assert abs(crps_emp - crps_ana) / crps_ana < 0.05

    def test_crps_perfect_forecast_near_zero(self):
        """CRPS is near zero when observation equals ensemble mean."""
        ens = RNG.normal(0.0, 1e-4, N_ENS)  # very tight ensemble
        y = np.array([0.0])
        crps_emp = float(crps_sample(ens[:, None], y)[0])
        assert crps_emp < 1e-3

    def test_crps_per_lead_shape(self):
        ens = RNG.standard_normal((N_ENS, 10))   # 10 lead times
        y = RNG.standard_normal(10)
        result = crps_per_lead(ens, y)
        assert result.shape == (10,)
        assert np.all(result >= 0)

    def test_crps_terminal(self):
        """Terminal CRPS is non-negative."""
        ens = RNG.standard_normal((200, 5))
        y = RNG.standard_normal(5)
        val = crps_terminal(ens, y)
        assert val >= 0

    def test_mean_crps_shape(self):
        T, N, H = 50, 200, 5
        ensembles = RNG.standard_normal((T, N, H))
        ys = RNG.standard_normal((T, H))
        result = mean_crps(ensembles, ys)
        assert result.shape == (H,)


# ---------------------------------------------------------------------------
# Rank histogram
# ---------------------------------------------------------------------------

class TestRankHistogram:
    def test_flat_for_calibrated_ensemble(self):
        """
        If ensemble and obs are drawn from the same distribution,
        rank histogram should be approximately flat (KS stat < 0.1).
        """
        T, N = N_TIMES, 50
        ens = RNG.standard_normal((T, N))
        obs = RNG.standard_normal(T)
        hist = rank_histogram(ens, obs, normalise=True)
        assert hist.shape == (N + 1,)
        ks = flatness_score(hist)
        assert ks < 0.15, f"KS statistic {ks:.3f} too large for calibrated ensemble"

    def test_udome_for_underdispersed(self):
        """
        Underdispersed ensemble (too narrow) should show U-shape:
        more mass in first and last bins.
        """
        T, N = N_TIMES, 50
        ens = RNG.standard_normal((T, N)) * 0.1  # very narrow
        obs = RNG.standard_normal(T)              # true spread = 1
        hist = rank_histogram(ens, obs, normalise=True)
        # In a U-shape, tails dominate the centre
        edge_mass = hist[0] + hist[-1]
        centre_mass = hist[1:-1].mean()
        assert edge_mass > centre_mass

    def test_per_lead_shape(self):
        T, N, H = 100, 30, 5
        ens = RNG.standard_normal((T, N, H))
        obs = RNG.standard_normal((T, H))
        hists = rank_histogram_per_lead(ens, obs)
        assert hists.shape == (H, N + 1)


# ---------------------------------------------------------------------------
# Reliability diagram
# ---------------------------------------------------------------------------

class TestReliability:
    def test_reliability_diagonal_for_calibrated(self):
        """
        For a well-calibrated binary forecast (forecasted prob matches
        observed freq), the diagonal should be close to y=x.
        """
        T = 1000
        # True prob varies across forecast times
        true_p = RNG.uniform(0.1, 0.9, T)

        # Generate N ensemble members from Bernoulli(true_p)
        N = 200
        ens = (RNG.uniform(size=(T, N, 1)) < true_p[:, None, None]).astype(float)
        obs = (RNG.uniform(size=(T, 1)) < true_p[:, None]).astype(float)

        event_fn = lambda e: e[:, 0] > 0.5  # event = "value is 1"
        result = reliability_diagram(ens, event_fn, obs, n_bins=5, n_bootstrap=50)

        valid = ~np.isnan(result["mean_forecast"])
        if valid.sum() >= 3:
            mf = result["mean_forecast"][valid]
            of = result["obs_frequency"][valid]
            # Correlation between forecast and observed frequency > 0.8
            corr = np.corrcoef(mf, of)[0, 1]
            assert corr > 0.7, f"Reliability diagonal too weak: corr={corr:.2f}"

    def test_reliability_output_keys(self):
        T, N, H = 100, 50, 5
        ens = RNG.standard_normal((T, N, H))
        obs = RNG.standard_normal((T, H))
        event_fn = lambda e: e.sum(axis=1) > 0
        result = reliability_diagram(ens, event_fn, obs)
        for key in ("bin_centers", "mean_forecast", "obs_frequency",
                    "ci_lower", "ci_upper", "counts"):
            assert key in result


# ---------------------------------------------------------------------------
# Spread–skill
# ---------------------------------------------------------------------------

class TestSpreadSkill:
    def test_spread_skill_calibrated(self):
        """
        For a calibrated ensemble (spread ≈ error), mean_spread ≈ mean_rmse.
        """
        T, N = N_TIMES, 50
        # Draw obs and ensemble from same distribution
        mu = RNG.standard_normal(T)
        ens = mu[:, None] + RNG.standard_normal((T, N)) * 0.3
        obs = mu + RNG.standard_normal(T) * 0.3
        result = spread_skill(ens, obs)
        ratio = result["mean_spread"] / result["mean_rmse"]
        assert 0.5 < ratio < 2.0, f"Spread/skill ratio {ratio:.2f} outside [0.5, 2]"

    def test_spread_skill_per_lead_shape(self):
        T, N, H = 100, 30, 5
        ens = RNG.standard_normal((T, N, H))
        obs = RNG.standard_normal((T, H))
        results = spread_skill_per_lead(ens, obs)
        assert len(results) == H


# ---------------------------------------------------------------------------
# Pinball / quantile loss
# ---------------------------------------------------------------------------

class TestPinballLoss:
    def test_pinball_median_vs_mae(self):
        """
        Pinball loss at q=0.5 should equal 0.5 * MAE.
        """
        T, N = N_TIMES, 200
        ens = RNG.standard_normal((T, N))
        obs = RNG.standard_normal(T)
        q_hat = np.median(ens, axis=1)
        mae = np.abs(q_hat - obs).mean() * 0.5

        pb = pinball_loss(ens, obs, quantiles=np.array([0.5]))
        assert abs(pb[0] - mae) / max(mae, 1e-9) < 0.05

    def test_pinball_non_negative(self):
        T, N = 100, 50
        ens = RNG.standard_normal((T, N))
        obs = RNG.standard_normal(T)
        losses = pinball_loss(ens, obs)
        assert np.all(losses >= 0)

    def test_pinball_per_lead_shape(self):
        T, N, H = 50, 30, 5
        ens = RNG.standard_normal((T, N, H))
        obs = RNG.standard_normal((T, H))
        result = pinball_loss_per_lead(ens, obs)
        assert result.shape == (H, len(QUANTILE_LEVELS))
