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
  * spreads (long settles with/after short, compared by settlement INSTANT so
    AM vs PM series differ): the LESSER of the maximum potential loss (intrinsic
    payoff at every strike) and the shorts' naked requirement, plus longs paid
    in full (the 75% loan value applies only to stand-alone longs > 9 months).
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
    rule_version: str = "current"


# Effective dates of the strategy-based rules (Cboe Margin Manual, Nov-2021, p.4 table and notes).
RULE_DATES = {
    "short_option_pcts": pd.Timestamp("1988-06-06"),     # 20%/10% equity & narrow index, 15%/10% broad index
    "put_min_on_strike": pd.Timestamp("1997-06-02"),     # put minimum 10% of EXERCISE price (before: of underlying)
    "long_over_9m_75pct": pd.Timestamp("1999-08-23"),    # loan value on listed longs > 9 months
    "limited_risk_spreads": pd.Timestamp("1999-08-23"),  # butterflies/boxes, cash-account spreads, hedged maintenance
    "all_spreads_max_loss": pd.Timestamp("2012-08-29"),  # "Spreads: All Types - Maximum Potential Loss Computation"
}


def _asof(as_of) -> pd.Timestamp | None:
    if as_of is None:
        return None
    t = pd.Timestamp(as_of)
    if t.tzinfo is not None:
        t = t.tz_convert("America/New_York").tz_localize(None)
    return t.normalize()


def rule_version(as_of=None) -> str:
    """'current' (>= 2012-08-29), 'side_groups' (1999-08-23 .. 2012-08-28),
    'pairwise' (1988-06-06 .. 1999-08-22), 'pre1988' (not modeled; 1988 rules applied)."""
    d = _asof(as_of)
    if d is None or d >= RULE_DATES["all_spreads_max_loss"]:
        return "current"
    if d >= RULE_DATES["limited_risk_spreads"]:
        return "side_groups"
    if d >= RULE_DATES["short_option_pcts"]:
        return "pairwise"
    return "pre1988"


UNDERLYING_CLASSES = {"equity": 0.20, "narrow_index": 0.20, "broad_index": 0.15, "broad_etf": 0.15,
                      "volatility_index": 0.20}


def naked_requirement(right: str, strike: float, price: float, S: float, underlying_class: str = "equity",
                      leverage: float = 1.0, multiplier: float = 100.0, qty: float = 1.0, as_of=None) -> float:
    """Cboe naked short requirement per |qty| contracts (USD, incl. premium)."""
    pct = UNDERLYING_CLASSES[underlying_class] * leverage
    minpct = 0.10 * leverage
    c = right.upper()[0] == "C"
    otm = max(strike - S, 0.0) if c else max(S - strike, 0.0)
    basic = price + pct * S - otm
    d = _asof(as_of)
    put_base = strike if (d is None or d >= RULE_DATES["put_min_on_strike"]) else S
    minimum = price + minpct * (S if c else put_base)
    return max(basic, minimum) * multiplier * abs(qty)


def _months(now, expiry) -> float:
    if now is None:
        return 0.0
    n = pd.Timestamp(now)
    if n.tzinfo is not None:
        n = n.tz_convert("America/New_York").tz_localize(None)
    return (pd.Timestamp(expiry).normalize() - n.normalize()).days / 30.4375


def long_requirement(p: OptionPosition, now=None, account: str = "margin", equity_based: bool = True,
                     as_of=None) -> float:
    cost = p.price * p.multiplier * abs(p.qty)
    d = _asof(as_of if as_of is not None else now)
    loan_ok = d is None or d >= RULE_DATES["long_over_9m_75pct"]
    if account == "margin" and equity_based and loan_ok and _months(now, p.expiry) > 9:
        return 0.75 * cost
    return cost


def _units(positions: Sequence[OptionPosition]):
    """Expand into single-contract units (qty must be integral)."""
    out = []
    for i, p in enumerate(positions):
        n = int(round(abs(p.qty)))
        if not np.isclose(n, abs(p.qty)):
            raise ValueError("option quantities must be integral contracts")
        ek = _settle_key(p)
        for _ in range(n):
            out.append(dict(i=i, K=float(p.strike), c=p.is_call, exp=pd.Timestamp(p.expiry), ek=ek,
                            long=p.qty > 0, px=float(p.price), m=float(p.multiplier), pos=p))
    return out


def _settle_key(p: "OptionPosition") -> pd.Timestamp:
    """Settlement instant (expiry date AND settlement type): an SPXW Thursday PM
    long settles before an SPX AM monthly short with the same expiry date."""
    from atlasbt.pricing.time import settlement_instant
    try:
        return settlement_instant(pd.Timestamp(p.expiry), p.settlement)
    except Exception:
        return pd.Timestamp(p.expiry).tz_localize("UTC") + pd.Timedelta(hours=21)


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


def _pairwise_loss(units: list[dict]) -> float:
    """Pre-1999 treatment: only two-leg verticals are recognized; each short is paired
    with one long of the same right (least adverse strike first) and charged
    max(0, adverse strike difference); losses are summed (no butterfly/condor netting)."""
    total = 0.0
    for is_c in (True, False):
        sh = sorted([u for u in units if u["c"] == is_c and not u["long"]], key=lambda u: (u["K"] if is_c else -u["K"]))
        lg = [u for u in units if u["c"] == is_c and u["long"]]
        for u in sh:
            if not lg:
                break
            best = min(lg, key=lambda v: (max(0.0, v["K"] - u["K"]) if is_c else max(0.0, u["K"] - v["K"])))
            lg.remove(best)
            total += (max(0.0, best["K"] - u["K"]) if is_c else max(0.0, u["K"] - best["K"])) * u["m"]
    return total


def strategy_margin(positions: Sequence[OptionPosition], underlying_price: float, stock_shares: float = 0.0,
                    underlying_class: str = "equity", leverage: float = 1.0, account: str = "margin",
                    now=None, apply_proceeds_in_cash_account: bool = False, as_of=None) -> MarginResult:
    """Strategy-based requirement for all option positions on ONE underlying.

    ``as_of`` (default ``now``): rules in force on that date (``rule_version``):
    current >= 2012-08-29 (max-loss over whole same-expiry groups); 1999-08-23..2012-08-28
    call and put sides grouped separately and summed (butterflies/boxes recognized, no
    cross-side netting -- conservative assumption); 1988-06-06..1999-08-22 two-leg verticals
    only; puts' 10% minimum on the underlying before 1997-06-02; 75% loan value on
    longs > 9 months only from 1999-08-23.  None -> current rules.
    ``underlying_class``: 'equity' | 'narrow_index' | 'broad_index' |
    'broad_etf' | 'volatility_index'.  ``stock_shares`` signed shares held (used
    for covering).  ``account``: 'margin' | 'cash'.  ``now`` (for the 9-month
    long-option rule).
    """
    S = float(underlying_price)
    as_of = as_of if as_of is not None else now
    regime = rule_version(as_of)
    units = _units(positions)
    comps: list = []
    notes: list = [f"rule version: {regime}"]
    if regime == "pre1988":
        notes.append("pre-1988-06-06 margin rules not modeled: 1988 percentages applied")

    def naked_req(c, K, px, m):
        return naked_requirement("C" if c else "P", K, px, S, underlying_class, leverage, m, as_of=as_of)
    short_mv = sum(u["px"] * u["m"] for u in units if not u["long"])
    long_mv = sum(u["px"] * u["m"] for u in units if u["long"])

    # ---- 1. excess shorts per expiry SERIES (keyed by settlement instant)
    exps = sorted({u["ek"] for u in units})
    groups: dict = {e: [u for u in units if u["ek"] == e] for e in exps}
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
    for u in sorted(excess, key=lambda u: (u["ek"], u["K"] if u["c"] else -u["K"])):
        best, best_loss, best_key = None, None, None
        for (e, is_c), lst in surplus.items():
            if is_c != u["c"] or e < u["ek"] or not lst:
                continue
            for cand in lst:
                loss = max(0.0, cand["K"] - u["K"]) if u["c"] else max(0.0, u["K"] - cand["K"])
                if best is None or loss < best_loss - 1e-12:
                    best, best_loss, best_key = cand, loss, (e, is_c)
        if best is not None:
            surplus[best_key].remove(best)
            groups[best["ek"]].remove(best)
            loss_usd = best_loss * u["m"]
            naked = naked_req(u["c"], u["K"], u["px"], u["m"])
            lbl = (f"time spread short {u['exp']:%Y-%m-%d} {u['K']:g}{'C' if u['c'] else 'P'} vs long "
                   f"{best['exp']:%Y-%m-%d} {best['K']:g}")
            if naked < loss_usd:
                comps.append((lbl + ": short naked requirement (< max loss)", naked))
            else:
                comps.append((lbl + ": max loss", loss_usd))
            comps.append(("time spread long paid in full", best["px"] * best["m"]))
            if account == "cash":
                notes.append("cash account: calendar/diagonal spreads not permitted (long treated as paid, short must be covered)")
        else:
            still.append(u)
    # ---- 3. bounded same-expiry groups: lesser of max potential loss and the shorts' naked
    #         requirement; longs in a spread are paid in full (75% loan value only for
    #         stand-alone longs > 9 months)
    for e in exps:
        g = groups[e]
        if not g:
            continue
        longs = [u for u in g if u["long"]]
        shorts = [u for u in g if not u["long"]]
        d0 = g[0]["exp"]
        if shorts:
            if regime == "current":
                loss = _group_max_loss(g)                                    # whole group, riskier side
            elif regime == "side_groups":
                loss = sum(_group_max_loss([u for u in g if u["c"] == side]) for side in (True, False))
            else:
                loss = _pairwise_loss(g)                                     # two-leg verticals only
            naked = sum(naked_req(u["c"], u["K"], u["px"], u["m"]) for u in shorts)
            if naked < loss:
                comps.append((f"spread group {d0:%Y-%m-%d}: shorts' naked requirement (< max loss)", naked))
            else:
                comps.append((f"spread group {d0:%Y-%m-%d}: max potential loss", loss))
            if account == "cash":
                eur_cash = all(u["pos"].exercise.upper() == "EUROPEAN" and u["pos"].settlement.upper().startswith("CASH")
                               for u in g)
                if not eur_cash:
                    notes.append(f"cash account: spread group {d0:%Y-%m-%d} not European cash-settled -> not permitted")
            for u in longs:
                comps.append((f"long {d0:%Y-%m-%d} {u['K']:g}{'C' if u['c'] else 'P'} paid in full (spread)",
                              u["px"] * u["m"]))
        else:
            for u in longs:
                comps.append((f"long {d0:%Y-%m-%d} {u['K']:g}{'C' if u['c'] else 'P'} paid",
                              long_requirement(u["pos"], now, account, as_of=as_of) / abs(u["pos"].qty)))
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
                          naked_requirement("C", u["K"], u["px"], S, underlying_class, leverage, u["m"], as_of=as_of)))
    else:
        rc = sorted([(naked_requirement("C", u["K"], u["px"], S, underlying_class, leverage, u["m"], as_of=as_of), u) for u in naked_c],
                    key=lambda t: -t[0])
        rp = sorted([(naked_requirement("P", u["K"], u["px"], S, underlying_class, leverage, u["m"], as_of=as_of), u) for u in naked_p],
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
                        notes=notes, rule_version=regime)


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
