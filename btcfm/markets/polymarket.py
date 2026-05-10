from __future__ import annotations

"""
Polymarket BTC market comparison — STAGE 2 STUB, not implemented.

Intended interface (to be implemented in stage 2):

  fetch_btc_markets() -> pl.DataFrame
    Pull live BTC binary/range markets from the Polymarket Gamma public API
    (https://gamma-api.polymarket.com/markets, no auth required).
    Returns columns: market_id, question, end_time, polymarket_prob.

  compute_model_probs(markets, ensemble_paths) -> pl.DataFrame
    For each market row, evaluate the contract condition (terminal > threshold,
    touch +k bps, range [lo, hi]) against Monte-Carlo ensemble paths from
    btcfm.model.sampler.generate_ensemble.
    Returns: market_id, model_prob.

  compare(markets, model_probs) -> pl.DataFrame
    Join the two frames and compute edge_bps = (model_prob - polymarket_prob)
    × 10 000.  Returns columns:
      market_id, question, polymarket_prob, model_prob, edge_bps.

Notes
-----
- Paths are in log-return space; convert to price relatives before applying
  threshold conditions: S_{t0+H} / S_{t0} = exp(Σ r_i).
- Touch contracts require checking the running maximum/minimum of exp(cumsum(r)),
  not just the terminal value.
- Range contracts check both lower and upper bounds.
- Bootstrap standard errors on model_prob should be reported.
"""
