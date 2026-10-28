"""Strategy-based option margin (Cboe Margin Manual, Nov-2021 edition; Cboe Rule
10.3 / FINRA Rule 4210(f)(2)) and a predefined stress-scenario loss calculator.

Requirement conventions follow the Cboe Margin Manual "Margin Requirement"
line, i.e. the requirement INCLUDES 100% of short option proceeds/market value
and the full cost of long options:
  * long option (<= 9 months): paid in full (100% of premium); > 9 months
    (listed, equity-based, margin account): 75% of premium.
  * naked short call: 100% premium + pct*S - OTM amount, min premium + 10%*S.
  * naked short put : 100% premium + pct*S - OTM amount, min premium + 10%*K.
    pct = 20% equity / narrow index, 15% broad-based index & broad-based ETF;
    both percentages are multiplied by the leverage factor of leveraged ETFs.
  * spreads (long expires with/after short): maximum potential loss from the
    intrinsic-value payoff evaluated at every strike, plus longs paid in full.
    A whole same-expiry group (iron condors, butterflies incl. broken wing,
    boxes, ratio spreads with surplus longs) is evaluated together, so iron
    condors/flies are charged the riskier side only.
  * short straddle / strangle: greater naked requirement + premium of the
    other side.
  * short call covered by long stock / short put covered by short stock:
    no option requirement (stock margined by Reg T in atlasbt.portfolio).
  * cash account: short put must be cash-secured (aggregate exercise price);
    short call must be covered by stock; spreads only European cash-settled.

How the portfolio uses it: ``MarginResult.net_requirement`` =
requirement - short option market value.  The account is in compliance iff
    NAV >= stock_reg_t_requirement + net_requirement
(NAV counting options at mark).  This reproduces every "SMA Debit or Margin
Call" figure in the manual (see tests/options/test_margin.py).

Pairing (strategy decomposition) is a documented greedy heuristic:
  1. per expiry, short calls beyond long calls (and short puts beyond long
     puts) are "excess"; the least risky are chosen as excess (highest strike
     calls / lowest strike puts);
  2. excess shorts are paired with SURPLUS longs of later expiries
     (calendars / diagonals; loss = max(0, K_long-K_short) for calls,
     max(0, K_short-K_long) for puts);
  3. each expiry's remaining legs form a bounded group charged its max loss;
  4. remaining short calls covered by long stock (100 sh/contract), remaining
     short puts by short stock;
  5. remaining naked calls & puts paired as straddles/strangles (largest with
     largest); leftovers charged naked.
Brokers may charge more (house margin); calendars of European options get no
special add-on here (Cboe manual notes brokers add one).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass
class OptionPosition:
    strike: float
    right: str                 # 'C' | 'P'
    expiry: pd.Timestamp
    qty: float                 # signed contracts (+ long, - short)
    price: float               # premium per share (sale proceeds at entry / current mark for maintenance)
    multiplier: float = 100.0
    exercise: str = "AMERICAN"
    settlement: str = "PHYSICAL"
    iv: float = float("nan")   # used by the stress calculator only

    @property
    def is_call(self) -> bool:
        return self.right.upper()[0] == "C"

    @staticmethod
    def from_leg(leg, qty: float, price: float | None = None) -> "OptionPosition":
        return OptionPosition(leg.strike, leg.right, pd.Timestamp(leg.expiry), qty,
                              leg.mid if price is None else price, leg.multiplier, leg.exercise, leg.settlement, leg.iv)


@dataclass
class MarginResult:
    requirement: float                  # Cboe "Margin Requirement" (USD)
    short_market_value: float           # proceeds / short option market value (USD)
    long_market_value: float
    net_requirement: float              # requirement - short_market_value (= SMA debit / margin call)
    components: list = field(default_factory=list)   # (description, USD)
    cash_account_ok: Optional[bool] = None
    notes: list = field(default_factory=list)


UNDERLYING_CLASSES = {"equity": 0.20, "narrow_index": 0.20, "broad_index": 0.15, "broad_etf": 0.15,
                      "volatility_index": 0.20}


def naked_requirement(right: str, strike: float, price: float, S: float, underlying_class: str = "equity",
                      leverage: float = 1.0, multiplier: float = 100.0, qty: float = 1.0) -> float:
    """Cboe naked short requirement per |qty| contracts (USD, incl. premium)."""
    pct = UNDERLYING_CLASSES[underlying_class] * leverage
    minpct = 0.10 * leverage
    c = right.upper()[0] == "C"
    otm = max(strike - S, 0.0) if c else max(S - strike, 0.0)
    basic = price + pct * S - otm
    minimum = price + minpct * (S if c else strike)
    return max(basic, minimum) * multiplier * abs(qty)


def _months(now, expiry) -> float:
    if now is None:
        return 0.0
    n = pd.Timestamp(now)
    if n.tzinfo is not None:
        n = n.tz_convert("America/New_York").tz_localize(None)
    return (pd.Timestamp(expiry).normalize() - n.normalize()).days / 30.4375


def long_requirement(p: OptionPosition, now=None, account: str = "margin", equity_based: bool = True) -> float:
    cost = p.price * p.multiplier * abs(p.qty)
    if account == "margin" and equity_based and _months(now, p.expiry) > 9:
        return 0.75 * cost
    return cost


def _units(positions: Sequence[OptionPosition]):
    """Expand into single-contract units (qty must be integral)."""
    out = []
    for i, p in enumerate(positions):
        n = int(round(abs(p.qty)))
        if not np.isclose(n, abs(p.qty)):
            raise ValueError("option quantities must be integral contracts")
        for _ in range(n):
            out.append(dict(i=i, K=float(p.strike), c=p.is_call, exp=pd.Timestamp(p.expiry),
                            long=p.qty > 0, px=float(p.price), m=float(p.multiplier), pos=p))
    return out


def _group_max_loss(units: list[dict]) -> float:
    """Max potential loss (USD, >= 0) of a bounded group, intrinsic values at
    every strike (Cboe "maximum potential loss computation")."""
    if not units:
        return 0.0
    ks = np.unique([u["K"] for u in units] + [0.0])
    pay = np.zeros_like(ks)
    for u in units:
        intr = np.maximum(ks - u["K"], 0) if u["c"] else np.maximum(u["K"] - ks, 0)
        pay += (1 if u["long"] else -1) * intr * u["m"]
    return float(max(0.0, -pay.min()))


def strategy_margin(positions: Sequence[OptionPosition], underlying_price: float, stock_shares: float = 0.0,
                    underlying_class: str = "equity", leverage: float = 1.0, account: str = "margin",
                    now=None, apply_proceeds_in_cash_account: bool = False) -> MarginResult:
    """Strategy-based requirement for all option positions on ONE underlying.

    ``underlying_class``: 'equity' | 'narrow_index' | 'broad_index' |
    'broad_etf' | 'volatility_index'.  ``stock_shares`` signed shares held (used
    for covering).  ``account``: 'margin' | 'cash'.  ``now`` (for the 9-month
    long-option rule).
    """
    S = float(underlying_price)
    units = _units(positions)
    comps: list = []
    notes: list = []
    short_mv = sum(u["px"] * u["m"] for u in units if not u["long"])
    long_mv = sum(u["px"] * u["m"] for u in units if u["long"])

    # ---- 1. excess shorts per expiry
    exps = sorted({u["exp"] for u in units})
    groups: dict = {e: [u for u in units if u["exp"] == e] for e in exps}
    excess: list = []
    surplus: dict = {}
    for e in exps:
        g = groups[e]
        for is_c in (True, False):
            sh = [u for u in g if u["c"] == is_c and not u["long"]]
            lg = [u for u in g if u["c"] == is_c and u["long"]]
            n_ex = len(sh) - len(lg)
            if n_ex > 0:
                # least risky shorts become excess (calls: highest K; puts: lowest K)
                sh_sorted = sorted(sh, key=lambda u: (-u["K"] if is_c else u["K"]))
                ex = sh_sorted[:n_ex]
                excess += ex
                for u in ex:
                    g.remove(u)
            elif n_ex < 0:
                # surplus longs available to cover nearer-dated shorts (calls: lowest K; puts: highest K)
                lg_sorted = sorted(lg, key=lambda u: (u["K"] if is_c else -u["K"]))
                surplus.setdefault((e, is_c), []).extend(lg_sorted[:(-n_ex)])
    # ---- 2. calendars / diagonals with later-expiring surplus longs
    still: list = []
    for u in sorted(excess, key=lambda u: (u["exp"], u["K"] if u["c"] else -u["K"])):
        best, best_loss, best_key = None, None, None
        for (e, is_c), lst in surplus.items():
            if is_c != u["c"] or e < u["exp"] or not lst:
                continue
            for cand in lst:
                loss = max(0.0, cand["K"] - u["K"]) if u["c"] else max(0.0, u["K"] - cand["K"])
                if best is None or loss < best_loss - 1e-12:
                    best, best_loss, best_key = cand, loss, (e, is_c)
        if best is not None:
            surplus[best_key].remove(best)
            groups[best["exp"]].remove(best)
            loss_usd = best_loss * u["m"]
            comps.append((f"time spread short {u['exp']:%Y-%m-%d} {u['K']:g}{'C' if u['c'] else 'P'} vs long "
                          f"{best['exp']:%Y-%m-%d} {best['K']:g}: max loss", loss_usd))
            comps.append(("time spread long paid", long_requirement(best["pos"], now, account) / abs(best["pos"].qty)))
            if account == "cash":
                notes.append("cash account: calendar/diagonal spreads not permitted (long treated as paid, short must be covered)")
        else:
            still.append(u)
    # ---- 3. bounded same-expiry groups
    for e in exps:
        g = groups[e]
        if not g:
            continue
        loss = _group_max_loss(g)
        longs = [u for u in g if u["long"]]
        if any(not u["long"] for u in g):
            comps.append((f"spread group {e:%Y-%m-%d}: max potential loss", loss))
            if account == "cash":
                eur_cash = all(u["pos"].exercise.upper() == "EUROPEAN" and u["pos"].settlement.upper().startswith("CASH")
                               for u in g)
                if not eur_cash:
                    notes.append(f"cash account: spread group {e:%Y-%m-%d} not European cash-settled -> not permitted")
        for u in longs:
            comps.append((f"long {e:%Y-%m-%d} {u['K']:g}{'C' if u['c'] else 'P'} paid",
                          long_requirement(u["pos"], now, account) / abs(u["pos"].qty)))
    # ---- 4. cover by stock
    shares = float(stock_shares)
    naked_c, naked_p = [], []
    for u in sorted(still, key=lambda u: u["K"]):  # cover the most ITM calls first
        if u["c"] and shares >= u["m"] - 1e-9:
            shares -= u["m"]
            comps.append((f"covered call {u['K']:g} (stock)", 0.0))
        elif u["c"]:
            naked_c.append(u)
    sh_short = -float(stock_shares) if stock_shares < 0 else 0.0
    for u in sorted([u for u in still if not u["c"]], key=lambda u: -u["K"]):
        if sh_short >= u["m"] - 1e-9:
            sh_short -= u["m"]
            comps.append((f"covered put {u['K']:g} (short stock)", 0.0))
        else:
            naked_p.append(u)
    # ---- 5. naked: cash account / straddle pairing
    cash_ok = True
    if account == "cash":
        if naked_c:
            cash_ok = False
            notes.append("cash account: uncovered short calls not permitted")
        for u in naked_p:
            # net_requirement = requirement - proceeds; by default proceeds are NOT applied
            # (Cboe: deposit cash equal to the aggregate exercise price), so add them back.
            extra = 0.0 if apply_proceeds_in_cash_account else u["px"] * u["m"]
            comps.append((f"cash-secured put {u['K']:g}", u["K"] * u["m"] + extra))
        for u in naked_c:
            comps.append((f"naked call {u['K']:g} (not permitted in cash account)",
                          naked_requirement("C", u["K"], u["px"], S, underlying_class, leverage, u["m"])))
    else:
        rc = sorted([(naked_requirement("C", u["K"], u["px"], S, underlying_class, leverage, u["m"]), u) for u in naked_c],
                    key=lambda t: -t[0])
        rp = sorted([(naked_requirement("P", u["K"], u["px"], S, underlying_class, leverage, u["m"]), u) for u in naked_p],
                    key=lambda t: -t[0])
        for (qc, uc), (qp, up) in zip(rc, rp):
            if qc >= qp:
                comps.append((f"short straddle/strangle {up['K']:g}P/{uc['K']:g}C: call req + put premium", qc + up["px"] * up["m"]))
            else:
                comps.append((f"short straddle/strangle {up['K']:g}P/{uc['K']:g}C: put req + call premium", qp + uc["px"] * uc["m"]))
        n = min(len(rc), len(rp))
        for q, u in rc[n:]:
            comps.append((f"naked call {u['K']:g}", q))
        for q, u in rp[n:]:
            comps.append((f"naked put {u['K']:g}", q))
    # spread groups/time spreads: the Cboe requirement also counts short proceeds applied, i.e.
    # requirement = losses + longs + (naked parts incl. their premium).  Short premium inside
    # spreads is NOT added to the requirement (manual: "Margin Requirement: $500 + $640").
    req = float(sum(c[1] for c in comps))
    return MarginResult(requirement=req, short_market_value=float(short_mv), long_market_value=float(long_mv),
                        net_requirement=req - float(short_mv), components=comps,
                        cash_account_ok=(cash_ok and not any("not permitted" in n for n in notes)) if account == "cash" else None,
                        notes=notes)


def package_margin(pkg, units: float, underlying_price: float, stock_shares: float = 0.0,
                   underlying_class: str = "equity", entry_prices: Optional[Sequence[float]] = None,
                   **kw) -> MarginResult:
    """Convenience: margin for ``units`` of an ``atlasbt.options.structures.Package``
    priced at natural fills by default (buy at ask / sell at bid)."""
    pos = []
    for i, l in enumerate(pkg.legs):
        px = entry_prices[i] if entry_prices is not None else (l.ask if l.ratio > 0 else l.bid)
        pos.append(OptionPosition.from_leg(l, l.ratio * units, px))
    return strategy_margin(pos, underlying_price, stock_shares, underlying_class, **kw)


# --------------------------------------------------------------------------
# Stress scenarios (for OPT_NAKED sizing)
# --------------------------------------------------------------------------
DEFAULT_SCENARIOS = (
    (-0.25, +0.20), (-0.15, +0.10), (-0.25, 0.0), (-0.15, 0.0),
    (+0.15, 0.0), (+0.25, 0.0), (+0.15, -0.05), (+0.25, -0.05),
)


def stress_loss(positions: Sequence[OptionPosition], S: float, now, r: float = 0.0, q: float = 0.0,
                stock_shares: float = 0.0, scenarios=DEFAULT_SCENARIOS, horizon_days: float = 1.0,
                iv_default: float = 0.25, pricer: str = "bsm",
                iv_fn: Optional[Callable[[OptionPosition], float]] = None) -> dict:
    """Worst-case P&L (USD) over predefined (spot %, vol-points) shocks, after
    ``horizon_days`` of time decay.  Each leg is repriced with BSM (or
    Bjerksund-Stensland 2002 for 'american') at shocked spot and IV + shock
    (floored at 1%).  Returns {'worst_loss', 'table'} where worst_loss >= 0.
    Unbounded-risk structures are captured only up to the largest shock —
    the caller should treat the result as a lower bound for naked calls."""
    from atlasbt.pricing.american import bs2002_price
    from atlasbt.pricing.bsm import bsm_price
    from atlasbt.pricing.time import year_fraction
    now = pd.Timestamp(now)
    rows = []
    base_vals = []
    legs = []
    for p in positions:
        T = year_fraction(now, p.expiry, p.settlement) if now.tzinfo is not None else max((pd.Timestamp(p.expiry) - now).days, 0) / 365.0
        iv = iv_fn(p) if iv_fn else (p.iv if np.isfinite(p.iv) else iv_default)
        legs.append((p, T, iv))
        base_vals.append(p.price)
    f = (lambda *a: bs2002_price(*a)) if pricer == "american" else (lambda *a: bsm_price(*a))
    for ds, dv in scenarios:
        S1 = S * (1 + ds)
        pnl = stock_shares * (S1 - S)
        for (p, T, iv), v0 in zip(legs, base_vals):
            T1 = max(T - horizon_days / 365.0, 0.0)
            v1 = float(f(S1, p.strike, T1, r, q, max(iv + dv, 0.01), p.is_call))
            pnl += (v1 - v0) * p.qty * p.multiplier
        rows.append(dict(spot_shock=ds, vol_shock=dv, pnl=pnl))
    table = pd.DataFrame(rows)
    return {"worst_loss": float(max(0.0, -table["pnl"].min())), "table": table}
