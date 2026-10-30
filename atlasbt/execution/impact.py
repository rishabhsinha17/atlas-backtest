"""Square-root market impact.

    impact (fraction of price) = k * sigma_daily * sqrt(Q / ADV)

with Q the traded quantity (shares) of the fill and ADV the 20-day average
daily volume, both measured in DOLLARS here (dollar volume is split
invariant; Q$/ADV$ == Q/ADV in shares on a non-split day).

Default k = 0.1. Justification:
* Almgren, Thum, Hauptmann & Li (2005, "Direct estimation of equity market
  impact", Risk) fit temporary impact eta*sigma*(X/(V*T))^0.6 with eta=0.142
  and permanent impact gamma*sigma*(X/V) with gamma=0.314. For a one-day
  execution the expected cost is ~ temporary + half permanent:
     Q/V = 0.1%: 0.0024 sigma   (k=0.1 sqrt law: 0.0032 sigma)
     Q/V = 1%  : 0.0105 sigma   (k=0.1 sqrt law: 0.0100 sigma)
     Q/V = 10% : 0.0514 sigma   (k=0.1 sqrt law: 0.0316 sigma)
  so k ~= 0.1 reproduces their average *cost* within a factor ~1.5 over the
  range our strategies trade (Q/V << 10%).
* Toth et al. (2011, "Anomalous price impact and the critical nature of
  liquidity in financial markets", PRX) report Y of order 1 (0.5-1) for the
  PEAK impact of metaorders; peak impact partially reverts, so the realized
  cost coefficient is lower. Use k=0.5 as a stress setting.
* Frazzini, Israel & Moskowitz (2018, "Trading Costs") find realized
  institutional impact well below standard model estimates.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class SquareRootImpact:
    k: float = 0.1
    max_impact: float = 0.05          # cap per fill (5%) to avoid absurd values on tiny ADV

    def impact_fraction(self, notional: float, adv_dollar: float, sigma_daily: float) -> float:
        if not (notional > 0 and adv_dollar > 0 and sigma_daily >= 0):
            return 0.0
        if not (math.isfinite(adv_dollar) and math.isfinite(sigma_daily)):
            return 0.0
        return min(self.k * sigma_daily * math.sqrt(notional / adv_dollar), self.max_impact)


@dataclass
class NoImpact(SquareRootImpact):
    k: float = 0.0
