"""Proxy variants: first-class runs of a strategy with an opt-in, documented proxy (docs/RUNNING.md §2b).

Some strategies are NOT_TESTABLE at their faithful resolution (the sheet's 15:45/15:50/09:35 decision
times need 1-minute bars the free data lacks for the universe) but ship an opt-in proxy param that the
strategy itself documents and flags ``Fidelity.T2_PROXY``. A *proxy variant* runs the base strategy with
that param set, under its own id ``<SID>@<tag>``, with its own plan row (``variant_of`` = base id), its
own results directory ``runs/<SID>@<tag>/`` and its own report. Variants are never mixed with faithful
results: ``atlas`` lists them in their own table, excludes them from the primary leaderboard, the primary
DSR trial count and SPA, and deflates them with N = primary trials + every planned-testable variant grid.

Registry rule (what qualifies). A variant is registered only when ALL hold:
  1. the proxy is the strategy's own opt-in param, documented in its ``deviations``, and the strategy
     flags it ``T2_PROXY`` at run time (sheet-faithful in spirit: same signal, proxied timing/input);
  2. the base strategy is NOT_TESTABLE on free data (a proxy of a testable strategy adds trials and
     nothing else);
  3. the proxy is runnable on data we actually have (otherwise it would only add a NOT_TESTABLE row).
     Exception: a strategy family that declares the proxy uniformly but refuses it for specific members
     keeps those members registered, so that the plan records the strategy's own refusal (S032/S033).
  4. it never sets ``allow_synthetic=True`` for strategies whose SIGNALS would come from synthetic
     chains (docs/OPTIONS.md: synthetic smiles must not drive signals).
Candidates that fail a rule are listed in ``EXCLUDED`` with the reason (documentation; never run).
"""
from __future__ import annotations

from dataclasses import dataclass, field

SEP = "@"


@dataclass(frozen=True)
class Variant:
    vid: str
    base: str
    params: dict = field(default_factory=dict)
    reason: str = ""

    @property
    def tag(self) -> str:
        return self.vid.split(SEP, 1)[1]

    def info(self) -> dict:
        return {"variant_id": self.vid, "variant_of": self.base, "variant_params": dict(self.params),
                "variant_reason": self.reason, "fidelity_floor": "T2_PROXY",
                "label": "PROXY VARIANT - T2, not faithful to the sheet's timing"}


_TIMED = ("daily proxy of the sheet's intraday timing (calendar_events module convention): daily bars, "
          "'15:50' actions -> market-on-close at the official close, '09:35' actions -> market-on-open at the "
          "official open; signals use completed daily closes only; flagged T2_PROXY by the strategy")
_REFUSED = ("strategy refuses the daily proxy (signal needs the 15:45 intraday price; a daily version would "
            "signal and fill on the same closing print) -> stays NOT_TESTABLE; registered so the plan records it")


def _v(vid: str, params: dict, reason: str) -> Variant:
    return Variant(vid, vid.split(SEP, 1)[0], dict(params), reason)


#: variant_id -> Variant. Order = plan / report order.
VARIANTS: dict[str, Variant] = {v.vid: v for v in (
    _v("S031@proxy", {"proxy": True}, _TIMED + "; 10-session momentum gate on closes through the prior session"),
    _v("S032@proxy", {"proxy": True}, _REFUSED),
    _v("S033@proxy", {"proxy": True}, _REFUSED),
    _v("S034@proxy", {"proxy": True}, _TIMED + "; turn-of-month calendar rule is unchanged"),
    _v("S035@proxy", {"proxy": True}, _TIMED + "; month-end calendar rule is unchanged"),
    _v("S036@proxy", {"proxy": True}, _TIMED + "; SPY>SMA200 gate on the previous week's last close (as the sheet)"),
    _v("S037@proxy", {"proxy": True}, _TIMED + "; pre-holiday calendar rule is unchanged"),
    _v("S015@mid_proxy", {"mid_proxy": True},
       "15:25 quote midpoint proxied by the close of the 15:24 1-minute bar (no historical quotes); timing "
       "unchanged; flagged T2_PROXY by the strategy; needs ETF_CORE 1-minute bars (short Yahoo window)"),
)}

#: candidates reviewed and NOT registered (documentation; see module docstring for the rules)
EXCLUDED: dict[str, str] = {
    "S005@pm_mid_proxy": "rule 3: still needs 1-minute PRE bars for SP500_TOP50_DV single stocks (not in the free data)",
    "S006@pm_mid_proxy": "rule 3: same as S005 (1-minute PRE bars for single stocks)",
    "S042@proxy": ("rules 2/3: base is testable on 1-minute ETF bars; a daily run also needs decl_proxy=True, i.e. "
                   "a portal decorator that re-stamps dividend declaration dates (a second, stacked proxy) and "
                   "SP500_TOP50_DV stocks"),
    "S038@posthoc_consensus": "rule 2: base is testable; post-hoc consensus is a data-quality opt-in, not an unlock",
    "S130@earnings_proxy": ("rule 2: base (SPY) is testable; the proxy only matters with use_megacap=True, which "
                            "changes the universe"),
    "S160@timing_proxy": ("rule 3: the close-of-day decision still needs 5-minute S&P 500 stock bars for the "
                          "30-minute opening range; the cache has them only from 2026-08-31 (~40 sessions), "
                          "while the plan would window the run from 2018 (a symbols=('*') bars requirement is "
                          "not windowed) -> a ~95% empty run (LEADING_GAP), not an evaluation"),
    "S165@vix_proxy": ("rule 2: base is testable on real chains from 2019; the VIX proxy would only extend history "
                       "onto synthetic (T3) fills"),
    "S158-S172@allow_synthetic": ("rule 4: synthetic smiles must not drive signals (docs/OPTIONS.md); "
                                  "allow_synthetic=True would make T3 chains the signal source"),
}


# ----------------------------------------------------------------- helpers
def is_variant(sid: str) -> bool:
    return SEP in str(sid)


def get(sid: str) -> Variant | None:
    return VARIANTS.get(str(sid))


def variant_ids() -> list[str]:
    return list(VARIANTS)


def base_of(sid: str) -> str:
    return str(sid).split(SEP, 1)[0]


def normalize_id(tok: str) -> str:
    """CLI token -> id: base part upper-cased, tag matched case-insensitively against the registry."""
    tok = str(tok).strip()
    if SEP not in tok:
        return tok.upper()
    b, t = tok.split(SEP, 1)
    want = f"{b.upper()}{SEP}{t}"
    for vid in VARIANTS:
        if vid.lower() == want.lower():
            return vid
    return want


def require(sid: str) -> Variant:
    v = get(sid)
    if v is None:
        raise KeyError(f"unknown proxy variant {sid!r} (registered: {', '.join(VARIANTS)})")
    return v


def strategy_class(sid: str):
    from atlasbt.strategies import registry as R

    if is_variant(sid):
        return R.get(require(sid).base)
    return R.get(sid)


def default_params(sid: str) -> dict:
    """Sheet params of the base strategy, overlaid with the variant's proxy params."""
    cls = strategy_class(sid)
    p = dict(cls.params)
    if is_variant(sid):
        p.update(require(sid).params)
    return p


def instantiate(sid: str, **over):
    return strategy_class(sid)(**{**default_params(sid), **over})


def expand_grid(sid: str) -> list[dict]:
    """The base strategy's grid restricted to the variant's params: combos whose value for a variant key
    differs are dropped (keys outside the grid are fixed by ``default_params``)."""
    from atlasbt.strategies import registry as R

    cls = strategy_class(sid)
    combos = R.expand_grid(cls.grid)
    if not is_variant(sid):
        return combos
    fixed = require(sid).params
    return [c for c in combos if all(c.get(k, v) == v for k, v in fixed.items())]


def info(sid: str) -> dict:
    v = get(sid)
    return v.info() if v else {}


def trials_total(primary_n: int, testable_variants=None) -> int:
    """DSR trial count for the PROXY VARIANT table: the primary atlas N plus every (planned-testable)
    variant's restricted grid. The primary leaderboard keeps ``primary_n`` (variants never inflate or
    dilute it)."""
    ids = variant_ids() if testable_variants is None else [v for v in testable_variants if v in VARIANTS]
    extra = 0
    for v in ids:
        try:
            extra += max(len(expand_grid(v)), 1)
        except Exception:
            extra += 1
    return int(primary_n) + extra
