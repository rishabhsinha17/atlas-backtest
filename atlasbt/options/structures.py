"""Multi-leg option packages: builders for every structure in S126-S172,
valuation at mid / natural, expiry payoff, max theoretical profit/loss and
breakevens, and conversion to core ``Leg``/``Order`` objects.

Sign conventions:
  * ``ratio`` > 0 long, < 0 short, per 1 package unit.
  * Package "price" is a NET DEBIT per unit in option-price units (per share):
    positive = you pay, negative = you receive a credit.
    ``natural_open`` buys at ask / sells at bid; ``natural_close`` the reverse.
  * Dollar P&L = units * multiplier * (value_at_exit - entry_price).
  * Stock legs (covered call, collar, conversion...) are expressed in SHARES per
    unit (e.g. +100) and priced at the underlying bid/ask/mid supplied.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from atlasbt.core.types import (ExerciseStyle, Instrument, Leg, Order, OrderType, Right, SettlementType,
                                TimeInForce)

from .chain import SelectionError, get_row, nearest_atm, select_by_delta, wing_by_points


def _num(v, default: float) -> float:
    try:
        f = float(v)
        return f if np.isfinite(f) and f > 0 else default
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class OptLeg:
    underlying: str
    expiry: pd.Timestamp
    strike: float
    right: str                 # 'C' | 'P'
    ratio: int
    bid: float = float("nan")
    ask: float = float("nan")
    mid: float = float("nan")
    delta: float = float("nan")
    iv: float = float("nan")
    gamma: float = float("nan")
    vega: float = float("nan")         # per 1 vol point, per share
    theta: float = float("nan")        # per calendar day, per share
    multiplier: float = 100.0
    exercise: str = "AMERICAN"
    settlement: str = "PHYSICAL"
    root: Optional[str] = None
    symbol: Optional[str] = None
    deliverable_shares: float = 100.0  # OCC-adjusted deliverable (non-standard after splits)
    tick_size: float = 0.01

    @staticmethod
    def from_row(row: pd.Series, ratio: int) -> "OptLeg":
        g = row.get
        mid = g("mid", np.nan)
        if not np.isfinite(mid if mid is not None else np.nan):
            mid = 0.5 * (float(g("bid", np.nan)) + float(g("ask", np.nan)))
        return OptLeg(underlying=str(g("underlying", "")), expiry=pd.Timestamp(g("expiry")), strike=float(g("strike")),
                      right=str(g("right")).upper()[0], ratio=int(ratio), bid=float(g("bid", np.nan)),
                      ask=float(g("ask", np.nan)), mid=float(mid),
                      delta=float(g("delta_used", g("delta", np.nan))), iv=float(g("iv_used", g("iv", np.nan))),
                      gamma=float(g("gamma", np.nan)), vega=float(g("vega", np.nan)), theta=float(g("theta", np.nan)),
                      multiplier=float(g("multiplier", 100.0) or 100.0), exercise=str(g("exercise", "AMERICAN") or "AMERICAN"),
                      settlement=str(g("settlement", "PHYSICAL") or "PHYSICAL"),
                      root=(str(g("root")) if g("root") is not None and str(g("root")) != "nan" else None),
                      deliverable_shares=_num(g("deliverable_shares", None), 100.0),
                      tick_size=_num(g("tick_size", None), 0.01),
                      symbol=(str(g("symbol")) if g("symbol") is not None and str(g("symbol")) != "nan" else None))

    def instrument(self) -> Instrument:
        """core Instrument with the chain row's multiplier, deliverable, tick, root,
        settlement and exercise (non-standard contracts keep their terms)."""
        from dataclasses import replace
        ins = Instrument.option(self.underlying, self.expiry, self.strike, Right(self.right),
                                exercise=ExerciseStyle(self.exercise.upper()),
                                settlement=SettlementType(self.settlement.upper()),
                                multiplier=self.multiplier, root=self.root, tick_size=self.tick_size)
        if self.deliverable_shares != ins.deliverable_shares:
            ins = replace(ins, deliverable_shares=float(self.deliverable_shares))
        return ins

    def intrinsic(self, S):
        """Per price unit; uses the effective underlying deliverable*S/multiplier."""
        S = np.asarray(S, float) * self.deliverable_shares / self.multiplier
        return np.maximum(S - self.strike, 0.0) if self.right == "C" else np.maximum(self.strike - S, 0.0)


@dataclass(frozen=True)
class StockLeg:
    symbol: str
    shares: float              # per package unit, signed
    bid: float = float("nan")
    ask: float = float("nan")

    @property
    def mid(self) -> float:
        return 0.5 * (self.bid + self.ask)


@dataclass
class Package:
    name: str
    legs: list[OptLeg]
    stock: list[StockLeg] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    # ---- structure facts
    @property
    def multiplier(self) -> float:
        return self.legs[0].multiplier if self.legs else 1.0

    @property
    def same_expiry(self) -> bool:
        """All legs settle at the same instant (expiry date AND settlement type:
        an SPXW Thursday PM leg and an SPX AM monthly leg are different)."""
        from atlasbt.pricing.time import settlement_instant
        return len({settlement_instant(l.expiry, l.settlement) for l in self.legs}) <= 1

    @property
    def expiry(self) -> pd.Timestamp:
        return min(l.expiry for l in self.legs)

    def strikes(self) -> np.ndarray:
        return np.array(sorted({l.strike for l in self.legs}))

    # ---- valuation (per unit, option-price units; stock per share / multiplier)
    def _stock_px(self, side: str) -> float:
        tot = 0.0
        for s in self.stock:
            px = s.mid if side == "mid" else (s.ask if (s.shares > 0) == (side == "open") else s.bid)
            tot += s.shares * px / self.multiplier
        return tot

    def mid_value(self) -> float:
        return float(sum(l.ratio * l.mid for l in self.legs)) + self._stock_px("mid")

    def natural_open(self) -> float:
        """Net debit (+) / credit (-) to OPEN at natural prices."""
        return float(sum(l.ratio * (l.ask if l.ratio > 0 else l.bid) for l in self.legs)) + self._stock_px("open")

    def natural_close(self) -> float:
        """Net value received (+) when CLOSING at natural prices
        (sell longs at bid, buy back shorts at ask)."""
        return float(sum(l.ratio * (l.bid if l.ratio > 0 else l.ask) for l in self.legs)) + self._stock_px("close")

    def price_with_improvement(self, frac: float = 0.0) -> float:
        """Opening price moved ``frac`` of the way from natural toward mid."""
        nat, mid = self.natural_open(), self.mid_value()
        return nat + frac * (mid - nat)

    def greeks(self) -> dict:
        """Per package unit, per share (x multiplier for $ per 1-point / per-day
        moves).  Stock legs add delta only.  NaN leg greeks count as 0."""
        def tot(attr):
            return float(sum(l.ratio * getattr(l, attr) for l in self.legs if np.isfinite(getattr(l, attr))))
        d = tot("delta") + sum(s.shares for s in self.stock) / self.multiplier
        return {"delta": d, "gamma": tot("gamma"), "vega": tot("vega"), "theta": tot("theta")}

    def dollar_greeks(self, units: float = 1.0) -> dict:
        """Share-equivalent delta, $ vega per vol point, $ theta per day."""
        g = self.greeks()
        m = self.multiplier * units
        return {"delta_shares": g["delta"] * m, "gamma_shares": g["gamma"] * m, "vega_usd": g["vega"] * m,
                "theta_usd": g["theta"] * m}

    # ---- expiry analysis (valid for same-expiry packages)
    def payoff(self, S) -> np.ndarray:
        """Terminal value per unit at common expiry (option-price units),
        excluding the entry price; stock legs valued at S."""
        S = np.asarray(S, float)
        v = sum(l.ratio * l.intrinsic(S) for l in self.legs)
        v = v + sum(s.shares * S for s in self.stock) / self.multiplier
        return v

    def pnl_at_expiry(self, S, entry_price: float) -> np.ndarray:
        return self.payoff(S) - entry_price

    def _slopes(self):
        """Payoff slope (per $1 of S) below lowest and above highest strike."""
        lo = sum(-l.ratio for l in self.legs if l.right == "P") + sum(s.shares for s in self.stock) / self.multiplier
        hi = sum(l.ratio for l in self.legs if l.right == "C") + sum(s.shares for s in self.stock) / self.multiplier
        return lo, hi

    def max_profit_loss(self, entry_price: float | None = None) -> dict:
        """Max theoretical profit / loss per unit (option-price units) at expiry.
        ``inf`` when unbounded.  Requires a same-expiry package."""
        if not self.same_expiry:
            raise ValueError("max profit/loss at expiry needs a same-expiry package (use scenario analysis)")
        e = self.natural_open() if entry_price is None else entry_price
        ks = self.strikes()
        pts = np.unique(np.r_[0.0, ks])
        pnl = self.pnl_at_expiry(pts, e)
        lo_slope, hi_slope = self._slopes()
        max_p = float(np.max(pnl))
        min_p = float(np.min(pnl))
        if hi_slope > 1e-12:
            max_p = np.inf
        if hi_slope < -1e-12:
            min_p = -np.inf
        # below lowest strike slope applies toward S=0 (finite, included via S=0 point)
        return {"max_profit": max_p, "max_loss": -min_p, "entry_price": e,
                "max_profit_usd": max_p * self.multiplier, "max_loss_usd": -min_p * self.multiplier}

    def breakevens(self, entry_price: float | None = None) -> list[float]:
        e = self.natural_open() if entry_price is None else entry_price
        ks = self.strikes()
        top = (ks.max() if ks.size else 1.0) * 3 + 1
        pts = np.unique(np.r_[0.0, ks, top])
        pnl = self.pnl_at_expiry(pts, e)
        out = []
        for a, b, pa, pb in zip(pts[:-1], pts[1:], pnl[:-1], pnl[1:]):
            if pa == 0:
                out.append(float(a))
            elif pa * pb < 0:
                out.append(float(a + (b - a) * (-pa) / (pb - pa)))
        if pnl[-1] == 0:
            out.append(float(pts[-1]))
        return sorted(set(round(x, 8) for x in out))

    # ---- engine objects
    def core_legs(self) -> tuple[Leg, ...]:
        return tuple(Leg(l.instrument(), int(l.ratio)) for l in self.legs)

    def to_order(self, units: float, limit_price: float | None = None, tag: str = "",
                 tif: TimeInForce = TimeInForce.DAY) -> Order:
        """COMBO order (+units buys the package as defined).  Stock legs are not
        part of the option combo; submit them separately (documented)."""
        return Order(instrument=None, quantity=float(units),
                     type=OrderType.COMBO, limit_price=limit_price,
                     tif=tif, legs=self.core_legs(), tag=tag or self.name)

    def describe(self) -> str:
        parts = [f"{'+' if l.ratio > 0 else ''}{l.ratio} {l.expiry:%Y-%m-%d} {l.strike:g}{l.right}" for l in self.legs]
        parts += [f"{s.shares:+g} sh {s.symbol}" for s in self.stock]
        return f"{self.name}: " + ", ".join(parts)


# ==========================================================================
# Builders.  ``x`` is a prepared single-expiry slice (atlasbt.options.chain.
# prepare_chain + slice_expiry); ``usable`` optional quality mask.
# ==========================================================================
def _ref(x: pd.DataFrame, ref_price):
    if ref_price is not None:
        return float(ref_price)
    return float(np.nanmedian(x["spot"])) if "spot" in x.columns else float(np.nanmedian(x["underlying_price"]))


def _leg(x, strike, right, ratio):
    return OptLeg.from_row(get_row(x, strike, right), ratio)


def single(x, target_delta: float, ratio: int = 1, usable=None, name: str | None = None) -> Package:
    """Long/short single option by delta (S126/S127/S130 put; S129 alt delta)."""
    r = select_by_delta(x, target_delta, usable=usable)
    return Package(name or ("long_" if ratio > 0 else "short_") + ("call" if target_delta > 0 else "put"),
                   [OptLeg.from_row(r, ratio)])


def atm_single(x, right: str, ratio: int = 1, ref_price=None, usable=None) -> Package:
    k = nearest_atm(x, _ref(x, ref_price), right, usable)
    return Package(("long_" if ratio > 0 else "short_") + ("atm_call" if right.upper()[0] == "C" else "atm_put"),
                   [_leg(x, k, right, ratio)])


def vertical(x, long_delta: float, short_delta: float, usable=None, name: str | None = None) -> Package:
    """Same-expiry vertical by deltas (S134-S137, S166, S169).  Both deltas
    must have the same sign (calls >0, puts <0).  Distinct strikes enforced."""
    lr = select_by_delta(x, long_delta, usable=usable)
    sr = select_by_delta(x, short_delta, usable=usable)
    if np.isclose(lr["strike"], sr["strike"]):
        raise SelectionError("vertical legs collapsed onto the same strike")
    return Package(name or "vertical", [OptLeg.from_row(lr, 1), OptLeg.from_row(sr, -1)])


def vertical_by_points(x, short_delta: float, width: float, usable=None, policy: str = "outward") -> Package:
    """Credit vertical: short by delta, long wing ``width`` points further OTM
    on an actually listed strike (S136 '5/10 points')."""
    sr = select_by_delta(x, short_delta, usable=usable)
    right = sr["right"]
    k = wing_by_points(x, sr["strike"], width, right, policy, usable)
    return Package("credit_vertical", [OptLeg.from_row(sr, -1), _leg(x, k, right, 1)])


def straddle(x, ratio: int = 1, ref_price=None, usable=None) -> Package:
    """ATM straddle on a strike listed for both rights (S143/S146/S147/S158/S164)."""
    k = nearest_atm(x, _ref(x, ref_price), usable=usable, both_rights=True)
    return Package("straddle" if ratio > 0 else "short_straddle", [_leg(x, k, "C", ratio), _leg(x, k, "P", ratio)])


def strangle(x, call_delta: float = 0.25, put_delta: float = -0.25, ratio: int = 1, usable=None) -> Package:
    c = select_by_delta(x, abs(call_delta), "C", usable)
    p = select_by_delta(x, -abs(put_delta), "P", usable)
    if p["strike"] > c["strike"]:
        raise SelectionError("strangle put strike above call strike")
    return Package("strangle" if ratio > 0 else "short_strangle", [OptLeg.from_row(c, ratio), OptLeg.from_row(p, ratio)])


def iron_condor(x, short_delta: float = 0.20, wing_delta: float | None = 0.05, wing_points: float | None = None,
                usable=None, policy: str = "outward") -> Package:
    """Short iron condor (S138/S159/S161/S165): short +/-short_delta, long wings
    by delta or by points (actual listed strikes)."""
    sc = select_by_delta(x, abs(short_delta), "C", usable)
    sp = select_by_delta(x, -abs(short_delta), "P", usable)
    if wing_points is not None:
        kc = wing_by_points(x, sc["strike"], wing_points, "C", policy, usable)
        kp = wing_by_points(x, sp["strike"], wing_points, "P", policy, usable)
        lc, lp = get_row(x, kc, "C"), get_row(x, kp, "P")
    else:
        lc = select_by_delta(x, abs(wing_delta), "C", usable)
        lp = select_by_delta(x, -abs(wing_delta), "P", usable)
    if not (lp["strike"] < sp["strike"] <= sc["strike"] < lc["strike"]):
        raise SelectionError("iron condor strikes not ordered (wings collapsed onto shorts)")
    return Package("iron_condor", [OptLeg.from_row(lp, 1), OptLeg.from_row(sp, -1),
                                   OptLeg.from_row(sc, -1), OptLeg.from_row(lc, 1)])


def iron_butterfly(x, width: float, short: bool = True, ref_price=None, usable=None, policy: str = "exact") -> Package:
    """Iron fly: ATM straddle + wings at +/- width on listed strikes, symmetric
    (policy 'exact' -> SelectionError if a symmetric wing is not listed).
    S139 short (credit) / S140 long (debit)."""
    k = nearest_atm(x, _ref(x, ref_price), usable=usable, both_rights=True)
    kc = wing_by_points(x, k, width, "C", policy, usable)
    kp = wing_by_points(x, k, width, "P", policy, usable)
    if policy != "exact" and not np.isclose(kc - k, k - kp):
        raise SelectionError("asymmetric iron butterfly wings")
    s = -1 if short else 1
    return Package("short_iron_butterfly" if short else "long_iron_butterfly",
                   [_leg(x, kp, "P", -s), _leg(x, k, "P", s), _leg(x, k, "C", s), _leg(x, kc, "C", -s)])


def butterfly(x, right: str, width: float, far_multiple: float = 1.0, ref_price=None, usable=None,
              center_strike: float | None = None) -> Package:
    """Long butterfly centred ATM (S141 calls, S142 puts).  ``far_multiple`` != 1
    builds a broken-wing fly (S155: calls, K3-K2 = 1.5 (K2-K1)).  For calls the
    near wing is below the body; for puts the near wing is above it."""
    right = right.upper()[0]
    k2 = center_strike if center_strike is not None else nearest_atm(x, _ref(x, ref_price), right, usable)
    if right == "C":
        k1, k3 = k2 - width, k2 + width * far_multiple
    else:
        k3, k1 = k2 + width, k2 - width * far_multiple
    for k in (k1, k3):
        get_row(x, k, right)  # raises if not listed
    name = "butterfly" if far_multiple == 1.0 else "broken_wing_butterfly"
    return Package(f"long_{'call' if right == 'C' else 'put'}_{name}",
                   [_leg(x, k1, right, 1), _leg(x, k2, right, -2), _leg(x, k3, right, 1)])


def ratio_backspread(x, short_delta: float, long_delta: float, ratio: int = 2, usable=None) -> Package:
    """Sell 1 at short_delta, buy ``ratio`` at long_delta (S151 calls, S152 puts)."""
    sr = select_by_delta(x, short_delta, usable=usable)
    lr = select_by_delta(x, long_delta, usable=usable)
    if np.isclose(sr["strike"], lr["strike"]):
        raise SelectionError("backspread legs collapsed")
    return Package("ratio_backspread", [OptLeg.from_row(sr, -1), OptLeg.from_row(lr, ratio)])


def risk_reversal(x, delta: float = 0.25, bullish: bool = True, usable=None) -> Package:
    """Bullish: sell put -delta, buy call +delta (S153/S167).  Bearish reverse (S154)."""
    c = select_by_delta(x, abs(delta), "C", usable)
    p = select_by_delta(x, -abs(delta), "P", usable)
    s = 1 if bullish else -1
    return Package("bullish_risk_reversal" if bullish else "bearish_risk_reversal",
                   [OptLeg.from_row(c, s), OptLeg.from_row(p, -s)])


def jade_lizard(x, put_delta: float = 0.30, call_delta: float = 0.30, wing_delta: float = 0.15, usable=None) -> Package:
    """Short put + short call vertical (S156).  ``meta['no_upside_risk']`` is True
    iff total credit (natural) >= call-vertical width."""
    p = select_by_delta(x, -abs(put_delta), "P", usable)
    sc = select_by_delta(x, abs(call_delta), "C", usable)
    lc = select_by_delta(x, abs(wing_delta), "C", usable)
    if lc["strike"] <= sc["strike"]:
        raise SelectionError("jade lizard call wing not above short call")
    pkg = Package("jade_lizard", [OptLeg.from_row(p, -1), OptLeg.from_row(sc, -1), OptLeg.from_row(lc, 1)])
    credit = -pkg.natural_open()
    pkg.meta["no_upside_risk"] = bool(credit >= (lc["strike"] - sc["strike"]))
    pkg.meta["credit_natural"] = credit
    return pkg


def covered_call(x, stock_symbol: str, call_delta: float = 0.30, stock_bid=np.nan, stock_ask=np.nan,
                 usable=None, include_stock: bool = False) -> Package:
    """Short call against 100 shares (S128); stock leg optional (existing shares)."""
    c = select_by_delta(x, abs(call_delta), "C", usable)
    mult = float(c.get("multiplier", 100.0) or 100.0)
    st = [StockLeg(stock_symbol, mult, stock_bid, stock_ask)] if include_stock else []
    return Package("covered_call", [OptLeg.from_row(c, -1)], st, meta={"covered_shares": mult})


def cash_secured_put(x, put_delta: float = -0.30, usable=None, atm: bool = False, ref_price=None) -> Package:
    """S129 (ATM) / S130 / S131 cash state.  meta['collateral'] = strike*multiplier."""
    if atm:
        pkg = atm_single(x, "P", -1, ref_price, usable)
    else:
        pkg = Package("cash_secured_put", [OptLeg.from_row(select_by_delta(x, -abs(put_delta), "P", usable), -1)])
    l = pkg.legs[0]
    pkg.name = "cash_secured_put"
    pkg.meta["collateral"] = l.strike * l.multiplier
    return pkg


def protective_put(x, put_delta: float = -0.25, usable=None) -> Package:
    return Package("protective_put", [OptLeg.from_row(select_by_delta(x, -abs(put_delta), "P", usable), 1)])


def collar(x, put_delta: float = -0.25, call_delta: float = 0.25, usable=None) -> Package:
    """Long put + short call same expiry (S133); stock held separately."""
    p = select_by_delta(x, -abs(put_delta), "P", usable)
    c = select_by_delta(x, abs(call_delta), "C", usable)
    if p["strike"] >= c["strike"]:
        raise SelectionError("collar put strike >= call strike")
    return Package("collar", [OptLeg.from_row(p, 1), OptLeg.from_row(c, -1)])


def covered_strangle(x, call_delta: float = 0.20, put_delta: float = -0.20, usable=None) -> Package:
    """Short call (covered by shares) + short put (cash-secured) (S157)."""
    c = select_by_delta(x, abs(call_delta), "C", usable)
    p = select_by_delta(x, -abs(put_delta), "P", usable)
    pkg = Package("covered_strangle", [OptLeg.from_row(c, -1), OptLeg.from_row(p, -1)])
    pkg.meta["collateral"] = float(p["strike"]) * float(p.get("multiplier", 100.0) or 100.0)
    return pkg


def calendar(near: pd.DataFrame, far: pd.DataFrame, right: str, strike: float | None = None, ref_price=None,
             usable=None) -> Package:
    """Sell near / buy far at the same (ATM) strike listed in BOTH expiries
    (S148/S149/S168)."""
    right = right.upper()[0]
    ref = _ref(near, ref_price)
    if strike is None:
        common = sorted(set(near[near["right"] == right]["strike"]) & set(far[far["right"] == right]["strike"]))
        if not common:
            raise SelectionError("no common strike")
        common = np.array(common)
        dist = np.round(np.abs(common - ref), 10)
        strike = float(common[np.lexsort((common, dist))][0])
    n, f = _leg(near, strike, right, -1), _leg(far, strike, right, 1)
    if f.expiry <= n.expiry:
        raise SelectionError("far leg must expire after near leg")
    return Package(f"{'call' if right == 'C' else 'put'}_calendar", [n, f])


def diagonal(near: pd.DataFrame, far: pd.DataFrame, long_delta: float = 0.80, short_delta: float = 0.30,
             usable_near=None, usable_far=None) -> Package:
    """Long far-dated call (~0.80 delta) + short near call (~0.30) (S150)."""
    lr = select_by_delta(far, long_delta, usable=usable_far)
    sr = select_by_delta(near, short_delta, usable=usable_near)
    if pd.Timestamp(lr["expiry"]) <= pd.Timestamp(sr["expiry"]):
        raise SelectionError("diagonal long must expire after short")
    return Package("diagonal", [OptLeg.from_row(sr, -1), OptLeg.from_row(lr, 1)])


def box(x, k1: float, k2: float) -> Package:
    """Long box: +C(K1) -C(K2) +P(K2) -P(K1); terminal value (K2-K1) (S172)."""
    if not k1 < k2:
        raise SelectionError("box needs K1 < K2")
    return Package("long_box", [_leg(x, k1, "C", 1), _leg(x, k2, "C", -1), _leg(x, k2, "P", 1), _leg(x, k1, "P", -1)],
                   meta={"width": k2 - k1})


def conversion(x, strike: float, stock_symbol: str, stock_bid: float, stock_ask: float, reverse: bool = False) -> Package:
    """Conversion: +100 sh, -C(K), +P(K).  Reversal: -100 sh, +C(K), -P(K) (S171)."""
    c, p = get_row(x, strike, "C"), get_row(x, strike, "P")
    mult = float(c.get("multiplier", 100.0) or 100.0)
    s = -1 if reverse else 1
    return Package("reversal" if reverse else "conversion",
                   [OptLeg.from_row(c, -s), OptLeg.from_row(p, s)],
                   [StockLeg(stock_symbol, s * mult, stock_bid, stock_ask)])


# ==========================================================================
# hedging / matching helpers
# ==========================================================================
def hedge_shares(option_delta_shares: float, current_shares: float = 0.0, band_shares: float = 0.0,
                 round_lot: float = 1.0) -> float:
    """Shares to TRADE so that option delta (share-equivalent) + stock = 0,
    only if |net delta| exceeds ``band_shares`` (S146: 0.05 x 100 per
    option-equivalent; S147: 0.10 threshold).  Rounded to ``round_lot``."""
    net = option_delta_shares + current_shares
    if abs(net) <= band_shares:
        return 0.0
    return float(-np.round(net / round_lot) * round_lot)


def vega_match_units(target_vega_usd: float, pkg: "Package", ratio: float = 1.0) -> float:
    """Package units whose $ vega equals ``ratio`` x ``target_vega_usd``
    (S170 dispersion: index straddle vs sector straddles).  Fractional; round
    to whole contracts at order time."""
    v = pkg.dollar_greeks(1.0)["vega_usd"]
    if not np.isfinite(v) or v == 0:
        raise SelectionError("package has no vega")
    return float(ratio * target_vega_usd / v)
