"""Option chains, package construction, synthetic chains (T3), lifecycle,
strategy-based margin, parity/box utilities.  See docs/OPTIONS.md.

Modules:
  chain       canonical-chain prep, quality filters, expiry/delta/ATM/wing selection
  structures  Package + builders for every S126-S172 structure; valuation, payoff, breakevens
  listing     historical listed-expiry calendar, strike grids, tick sizes
  synthetic   synthetic chain generator from Cboe vol indices (T3_MODELED)
  calibrate   calibration of the synthetic model on DoltHub chains
  lifecycle   expiration / exercise / assignment / settlement / adjustments
  margin      Cboe strategy-based margin + stress scenarios
  parity      conversion/reversal edges, box-spread rates
  plugins     adapters for atlasbt.portfolio hooks (OptionLifecycle, OptionMarginModel)
"""
