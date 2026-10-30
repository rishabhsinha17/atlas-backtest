"""Put-call parity screens (S171) and box-spread financing (S172).

All prices per share; dividends as (time_in_years, amount) pairs with ex-dates
before expiry; rates continuously compounded.  Fees are per share (pass
fee_per_contract / multiplier).

Conversion  = long stock (ask) + long put (ask) + short call (bid): locks in K
at expiry plus dividends received.  Reversal = short stock (bid) + short put
(bid) + long call (ask): pays K at expiry plus dividends owed plus borrow.

American early exercise: the short call of a conversion can be assigned just
before an ex-date (dividend then lost but K received early); the short put of a
reversal can be assigned whenever deep ITM (K paid early, no further dividends /
borrow).  ``worst_case`` edges take the minimum over these stopping times
{each ex-date (cum day), expiry; and t=0 for reversals}.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _pv(divs, r, upto=np.inf):
    return float(sum(a * np.exp(-r * t) for t, a in divs if 0 < t <= upto))


def american_parity_bounds(S, K, T, r, divs=()):
    """Hull: S - PV(D) - K <= C - P <= S - K e^{-rT} (American, discrete dividends)."""
    return S - _pv(divs, r, T) - K, S - K * np.exp(-r * T)


def european_parity(S, K, T, r, divs=(), q=0.0):
    """C - P = S e^{-qT} - PV(D) - K e^{-rT}."""
    return S * np.exp(-q * T) - _pv(divs, r, T) - K * np.exp(-r * T)


@dataclass(frozen=True)
class ParityEdge:
    kind: str                 # 'conversion' | 'reversal'
    hold_to_expiry: float     # PV profit per share if held to expiry
    worst_case: float         # min over early-assignment stopping times
    outlay: float             # net cash paid (+) / received (-) at entry per share
    edge_bp_of_spot: float    # worst_case / S in bp


def conversion_edge(S_ask, C_bid, P_ask, K, T, r, divs=(), fee_per_share: float = 0.0,
                    funding_rate: float | None = None, american: bool = True, S_ref: float | None = None) -> ParityEdge:
    rf = r if funding_rate is None else funding_rate
    outlay = S_ask + P_ask - C_bid + fee_per_share
    hold = K * np.exp(-rf * T) + _pv(divs, rf, T) - outlay
    cands = [hold]
    if american:
        for t, _ in sorted(divs):
            if 0 < t <= T:  # assigned on the cum-dividend day: receive K at t, dividends before t only
                cands.append(K * np.exp(-rf * t) + _pv([(tt, a) for tt, a in divs if tt < t], rf) - outlay)
    worst = min(cands)
    S0 = S_ref if S_ref is not None else S_ask
    return ParityEdge("conversion", float(hold), float(worst), float(outlay), float(worst / S0 * 1e4))


def reversal_edge(S_bid, C_ask, P_bid, K, T, r, divs=(), borrow_fee: float = 0.0, fee_per_share: float = 0.0,
                  rebate_rate: float | None = None, american: bool = True, S_ref: float | None = None) -> ParityEdge:
    """Short-sale proceeds earn ``rebate_rate`` (default r - borrow_fee, i.e. the
    borrow fee is charged as a spread on the rebate); dividends owed are paid."""
    rr = (r - borrow_fee) if rebate_rate is None else rebate_rate
    proceeds = S_bid + P_bid - C_ask - fee_per_share

    def value(t_stop):
        # receive proceeds now, invest at rr until t_stop, pay K at t_stop, pay dividends with ex <= t_stop
        grow = proceeds * np.exp(rr * t_stop)
        pay = K + sum(a * np.exp(rr * (t_stop - t)) for t, a in divs if 0 < t <= t_stop)
        return (grow - pay) * np.exp(-r * t_stop)

    hold = value(T)
    cands = [hold]
    if american:
        cands.append(value(0.0))
        for t, _ in divs:
            if 0 < t <= T:
                cands.append(value(t))
    worst = min(cands)
    S0 = S_ref if S_ref is not None else S_bid
    return ParityEdge("reversal", float(hold), float(worst), float(-proceeds), float(worst / S0 * 1e4))


def parity_screen(S_bid, S_ask, C_bid, C_ask, P_bid, P_ask, K, T, r, divs=(), borrow_fee=0.0,
                  fee_per_share=0.0, min_edge_bp: float = 10.0, american: bool = True) -> dict:
    """Both directions; ``signal`` is 'conversion'/'reversal' only if the
    worst-case edge after costs >= ``min_edge_bp`` of spot."""
    conv = conversion_edge(S_ask, C_bid, P_ask, K, T, r, divs, fee_per_share, american=american,
                           S_ref=0.5 * (S_bid + S_ask))
    rev = reversal_edge(S_bid, C_ask, P_bid, K, T, r, divs, borrow_fee, fee_per_share, american=american,
                        S_ref=0.5 * (S_bid + S_ask))
    sig = None
    if conv.edge_bp_of_spot >= min_edge_bp:
        sig = "conversion"
    elif rev.edge_bp_of_spot >= min_edge_bp:
        sig = "reversal"
    return {"conversion": conv, "reversal": rev, "signal": sig}


# --------------------------------------------------------------------------
# Box spreads
# --------------------------------------------------------------------------
def box_prices(c1_bid, c1_ask, c2_bid, c2_ask, p1_bid, p1_ask, p2_bid, p2_ask) -> dict:
    """Long box (+C1 -C2 +P2 -P1) natural debit, short box natural credit, mid."""
    buy = c1_ask - c2_bid + p2_ask - p1_bid
    sell = c1_bid - c2_ask + p2_bid - p1_ask
    mid = 0.5 * (c1_bid + c1_ask) - 0.5 * (c2_bid + c2_ask) + 0.5 * (p2_bid + p2_ask) - 0.5 * (p1_bid + p1_ask)
    return {"buy_debit": buy, "sell_credit": sell, "mid": mid}


def box_implied_rate(price: float, width: float, T: float, fees_per_share: float = 0.0, side: str = "lend") -> dict:
    """Financing rate implied by a box traded at ``price`` (per share) with
    terminal value ``width``.  side='lend' (buy box: pay price + fees) or
    'borrow' (sell box: receive price - fees).  Returns cc and simple ACT/365."""
    eff = price + fees_per_share if side == "lend" else price - fees_per_share
    if eff <= 0 or T <= 0:
        return {"cc": float("nan"), "simple": float("nan"), "effective_price": eff}
    return {"cc": float(np.log(width / eff) / T), "simple": float((width / eff - 1) / T), "effective_price": eff}


def box_vs_tbill(quotes: dict, k1: float, k2: float, T: float, tbill_cc: float, fee_per_contract: float = 0.0,
                 multiplier: float = 100.0, legs: int = 4) -> dict:
    """S172 analysis: lending/borrowing rates of the natural box vs the T-bill
    (cc) rate, all fees included (``fee_per_contract`` x ``legs``)."""
    px = box_prices(**quotes)
    fps = fee_per_contract * legs / multiplier
    w = k2 - k1
    lend = box_implied_rate(px["buy_debit"], w, T, fps, "lend")
    borrow = box_implied_rate(px["sell_credit"], w, T, fps, "borrow")
    mid = box_implied_rate(px["mid"], w, T, 0.0, "lend")
    return {"prices": px, "lend_rate_cc": lend["cc"], "borrow_rate_cc": borrow["cc"], "mid_rate_cc": mid["cc"],
            "lend_minus_tbill_bp": (lend["cc"] - tbill_cc) * 1e4, "borrow_minus_tbill_bp": (borrow["cc"] - tbill_cc) * 1e4,
            "terminal_value_usd": w * multiplier}
