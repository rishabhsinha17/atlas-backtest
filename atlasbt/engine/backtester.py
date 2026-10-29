"""The backtest event loop (docs/ARCHITECTURE.md §4, docs/ENGINE.md).

Master timeline = bar END times at the strategy's base frequency. At each
step t (end of bar [t-Δ, t)):

1. session start (first step of a session): dividends due, corporate
   actions with ex_date in (prev session, session] (incl. order adjustment /
   cancellation and OCC option adjustment), AM-settled option settlements,
   ``strategy.on_session_start`` (ctx.now = session start);
2. fills: every working order is offered the data in (prev_t, t] whose bars
   START at/after the order's ``eligible_at`` (never a price from before the
   order existed);
3. ``on_fill`` / ``on_order_event`` callbacks (same step, ctx.now = t);
4. mark-to-market at t, session rules, then ``strategy.on_bar``;
5. session end (after the last step): ``on_session_end``, option expiry,
   option marks, financing accrual, DAY-order expiry, Reg-T margin check
   (margin calls -> liquidation orders for the next open), NAV record.
"""
from __future__ import annotations

import copy
import math
import time
import traceback
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from ..core.clock import ET, et, nyse, to_et
from ..core.types import (
    AssetClass, BacktestResult, CorporateAction, CostBreakdown, Fidelity, Fill, Instrument, Leg,
    LookaheadError, NotTestableError, Order, OrderStatus, OrderType, RunStatus, TimeInForce,
)
from ..execution.constraints import bar_is_halted, round_limit_for_side, round_stop_for_side, valid_tick
from ..execution.fills import (
    AuctionFillModel, BarFillModel, BarView, ComboFillModel, FillDecision, LegQuote, QuoteFillModel,
    QuoteView, SymbolMarket,
)
from ..execution.spread import SpreadModel
from ..portfolio.account import Account, AccountType
from ..portfolio.corporate_actions import CorporateActionProcessor, as_session_date
from ..portfolio.financing import accrue_financing
from ..portfolio.hooks import OptionEvent
from ..portfolio.ledger import Reason
from ..portfolio.margin import RegTMarginModel
from .config import BacktestConfig
from .context import EngineContext
from .market import MarketData
from .orders import ACTIVE, ChildTemplate, OrderExtras, OrderManager
from .results import FidelityLedger, TradeTracker, fills_frame
from .strategy import Strategy
from .timeutil import canonical_freq, freq_to_timedelta, is_daily, ts_utc

NS_MIN = 60_000_000_000


class Backtester:
    def __init__(self, strategy: Strategy, portal, config: Optional[BacktestConfig] = None, calendar=None):
        self.strategy = strategy
        self.portal = portal
        self.cfg = config or BacktestConfig()
        self.config_overrides = self._strategy_overrides(strategy)
        if self.config_overrides:
            from dataclasses import replace as _replace
            self.cfg = _replace(self.cfg, **self.config_overrides)
        self.cal = calendar or getattr(portal, "calendar", None) or nyse()
        self.base_freq = canonical_freq(strategy.bar_freq)
        self.daily = is_daily(self.base_freq)
        self.step_td = freq_to_timedelta(self.base_freq)
        self.timeline_sessions = tuple(strategy.sessions) if not self.daily else ("DAY",)
        data_sess = set(self.timeline_sessions) if not self.daily else {"RTH"}
        for r in strategy.requires:
            if r.kind == "bars" and (r.freq is None or canonical_freq(r.freq) == self.base_freq):
                data_sess |= set(r.sessions)
        order_ = {"PRE": 0, "RTH": 1, "POST": 2}
        self.data_sessions = tuple(sorted((s for s in data_sess if s in order_), key=order_.get)) or ("RTH",)
        self._reset_state()

    #: strategies may only choose model plug-ins (never costs, fees, fills)
    OVERRIDABLE = {"option_lifecycle", "option_margin"}

    @classmethod
    def _strategy_overrides(cls, strategy) -> dict:
        ov = dict(getattr(type(strategy), "config_overrides", None) or {})
        bad = set(ov) - cls.OVERRIDABLE
        if bad:
            raise ValueError(f"{strategy.id}: config_overrides may only set {sorted(cls.OVERRIDABLE)}, "
                             f"not {sorted(bad)}")
        return ov

    # ================================================================= state
    def _reset_state(self) -> None:
        cfg = self.cfg
        self._verified: dict[str, tuple[bool, str]] = {}
        self._cov_log: dict[tuple[str, str], tuple[int, int, list]] = {}
        self._cov_flagged: set = set()
        self._coverage_fail: Optional[str] = None
        self.account = Account(cfg.initial_cash, cfg.account_type)
        self.om = OrderManager()
        self.fees = copy.deepcopy(cfg.make_fees())
        self.spread_model = SpreadModel(cfg.spread)
        self.bar_model = BarFillModel(cfg.bar_fill, self.spread_model, cfg.impact)
        self.quote_model = QuoteFillModel(cfg.impact, cfg.bar_fill.ssr_mode, cfg.bar_fill.allow_fractional)
        self.auction_model = AuctionFillModel(cfg.auction, cfg.impact, cfg.bar_fill.allow_fractional)
        self.combo_model = ComboFillModel(cfg.combo)
        self.margin = RegTMarginModel(option_model=cfg.margin_model())
        self.ca = CorporateActionProcessor(cfg.corporate_actions, roll_to_session=self._roll_to_session)
        self.lifecycle = cfg.lifecycle(self.portal, lambda: self.now)
        self.trades = TradeTracker()
        self.fid = FidelityLedger()
        self.fill_rows: list[dict] = []
        self.equity: dict[pd.Timestamp, float] = {}
        self.rf_daily: dict[pd.Timestamp, float] = {}
        self.intraday_equity: list[tuple] = []
        self.pos_rows: list[dict] = []
        self.logs: list[dict] = []
        self.diag: dict[str, Any] = {
            "margin_calls": [], "warnings": [], "corporate_actions": [], "option_events": [],
            "financing": {"credit": 0.0, "debit": 0.0, "borrow": 0.0, "rebate": 0.0},
            "sim": {"halted_bars": 0, "nan_volume_bars": 0, "ambiguous_bars": 0, "partial_fills": 0,
                    "ssr_blocked": 0, "resolved_bars": 0},
            "invariant_checks": 0, "data_access": {}, "cash_account_negative_cash_days": 0,
            "order_events": [],
        }
        self._instruments: dict[str, Instrument] = {}
        self._pending_on_fill: list[Fill] = []
        self._pending_events: list[dict] = []
        self._sparse_quote_note = False
        self._state_ver = 0                     # bumped on fills / submissions (reduce-only clamp cache)
        self._clamp_seen: dict[str, tuple] = {}
        self._oc_cache: dict[int, tuple[int, int]] = {}
        self._stale_flagged: set[str] = set()
        self._rf_days_total = 0
        self._flat_step_ns: Optional[int] = None
        self._flat_moc_mode = False
        self._quotes_mode: dict[str, bool] = {}
        self._halts: dict[str, list[tuple[int, int]]] = {}
        self._halts_loaded: set[str] = set()
        self._borrow_cache: dict[tuple[str, int], tuple[bool, float, Fidelity]] = {}
        self._flattened_session = False
        self._session_rules = self.strategy.session_rules()
        self._flat_by_ns: Optional[int] = None
        self._no_entry_ns: Optional[int] = None
        self.prior_nav = self.cfg.initial_cash
        self.now = pd.Timestamp(0, tz="UTC")
        self.now_ns = 0
        self.session = pd.Timestamp(0)
        self.prev_session: Optional[pd.Timestamp] = None
        self.session_open_close = (self.now, self.now)
        self.prev_step_ns = 0
        self.md: Optional[MarketData] = None
        self.execution_state = EngineExecutionState(self)

    # ================================================================= helpers
    def _set_now(self, t_ns: int) -> None:
        self.now_ns = int(t_ns)
        self.now = pd.Timestamp(self.now_ns, tz="UTC")
        if self.md is not None:
            self.md.now_ns = self.now_ns

    def instrument(self, symbol: str) -> Instrument:
        inst = self._instruments.get(symbol)
        if inst is None:
            try:
                inst = self.portal.instrument(symbol)
            except Exception:
                inst = None
            if inst is None:
                inst = Instrument(symbol=symbol, asset_class=AssetClass.STOCK)
            self._instruments[symbol] = inst
        return inst

    def _master_option(self, inst: Instrument) -> tuple[Optional[Instrument], Optional[str]]:
        """Replace a strategy-built option instrument by the instrument-master
        record (deliverable, tick size, exercise/settlement style). Returns
        ``(master, None)``, or ``(None, reason)`` when the two disagree on
        multiplier / strike / right / expiry / underlying, or on an explicitly
        set (non-default) deliverable. Without a master record the strategy's
        instrument is used (and registered)."""
        sym = inst.symbol
        m = self._instruments.get(sym)
        if m is None or not m.is_option:
            try:
                m = self.portal.instrument(sym)
            except Exception:
                m = None
            if m is None or not m.is_option:
                self._instruments[sym] = inst
                return inst, None
            self._instruments[sym] = m
        if m is inst:
            return m, None
        bad = []
        if abs(float(inst.multiplier) - float(m.multiplier)) > 1e-9:
            bad.append(f"multiplier {inst.multiplier:g} vs {m.multiplier:g}")
        if inst.strike is not None and m.strike is not None and abs(float(inst.strike) - float(m.strike)) > 1e-9:
            bad.append(f"strike {inst.strike:g} vs {m.strike:g}")
        if inst.right is not None and m.right is not None and inst.right != m.right:
            bad.append(f"right {inst.right} vs {m.right}")
        if inst.expiry is not None and m.expiry is not None and \
                pd.Timestamp(inst.expiry).date() != pd.Timestamp(m.expiry).date():
            bad.append(f"expiry {pd.Timestamp(inst.expiry).date()} vs {pd.Timestamp(m.expiry).date()}")
        if inst.underlying and m.underlying and inst.underlying.upper() != m.underlying.upper():
            bad.append(f"underlying {inst.underlying} vs {m.underlying}")
        d_s, d_m = float(inst.deliverable_shares), float(m.deliverable_shares)
        if abs(d_s - d_m) > 1e-9:
            if abs(d_s - 100.0) > 1e-9:            # explicitly set by the strategy and wrong
                bad.append(f"deliverable {d_s:g} vs {d_m:g}")
            else:
                self.diag["warnings"].append(
                    f"{sym}: deliverable {d_m:g} shares taken from the instrument master (strategy built {d_s:g})")
        if bad:
            return None, f"option {sym} disagrees with the instrument master: " + "; ".join(bad)
        return m, None

    def _note_data_access(self, kind, symbol, fidelity) -> None:
        k = f"{kind}"
        self.diag["data_access"][k] = self.diag["data_access"].get(k, 0) + 1

    def flag_fidelity(self, tier: Fidelity, reason: str) -> None:
        self.fid.flag(self.now, tier, reason)

    def last_price(self, symbol: str) -> float:
        """Latest known price, PIT. Stocks: last completed bar close, adjusted
        for splits applied since that bar (a corporate-action-adjusted mark is
        used until a post-ex-date bar prints). Options: mark or quote mid."""
        inst = self.instrument(symbol)
        p = self.account.get(symbol)
        if inst.is_option:
            if p is not None and p.mark == p.mark:
                return p.mark
            q = self.portal.option_quote(inst, self.now)
            return q.mid if q is not None else float("nan")
        px, end_ns = self._last_bar_close(symbol)
        if p is not None and p.mark == p.mark and p.mark_ts is not None and (px != px or int(p.mark_ts.value) > end_ns):
            return p.mark
        if px == px:
            px /= self._split_factor_after(symbol, end_ns)
        return px

    def _last_bar_close(self, symbol: str) -> tuple[float, int]:
        if self.md is None:
            return float("nan"), -1
        b = self.md.bars(symbol)
        i = self.md.last_bar_index(symbol, self.now_ns)
        if i >= 0:
            return float(b.c[i]), int(b.end[i])
        st = self.md.daily(symbol)
        if st is not None:
            j = int(np.searchsorted(st.bars.end, self.now_ns, side="right")) - 1
            if j >= 0:
                return float(st.bars.c[j]), int(st.bars.end[j])
        return float("nan"), -1

    def _split_factor_after(self, symbol: str, end_ns: int) -> float:
        """Product of split ratios with ex-date after the bar's session and on
        or before the current session (ex-dates are known by then)."""
        sp = self.md.splits(symbol) if self.md is not None else {}
        if not sp or end_ns < 0:
            return 1.0
        bar_sess = int(self._session_of_ns(end_ns).value)
        cur = int(self.session.value)
        f = 1.0
        for d, r in sp.items():
            if bar_sess < d <= cur:
                f *= r
        return f

    def _uses_quotes(self, symbol: str) -> bool:
        """Does the portal have ANY quotes for the symbol (cached)? The fill
        source is decided per bar window by ``_quote_window`` ('auto')."""
        m = self._quotes_mode.get(symbol)
        if m is None:
            mode = self.cfg.use_quotes
            if mode == "never":
                m = False
            elif mode == "always":
                m = True
            else:
                try:
                    m = bool(self.portal.has(symbol, "quotes"))
                except Exception:
                    m = False
            self._quotes_mode[symbol] = m
        return m

    def _has_bars(self, symbol: str) -> bool:
        b = self.md.bars(symbol) if self.md is not None else None
        return b is not None and b.n > 0

    def _quote_window(self, sym: str, lo: int, hi: int):
        """Per-window fill source. Returns the quotes frame to fill from, or
        None for bar fills. 'auto': quotes are used only if they cover the
        window with at least ``quote_min_per_hour`` quotes per RTH hour (or if
        the symbol has no bars at all); otherwise bars. Never leaves an order
        unfillable for lack of quotes while bars exist."""
        if not self._uses_quotes(sym):
            return None
        try:
            df = self.portal.quotes(sym, ts_utc(lo + 1), ts_utc(hi + 1), as_of=self.now)
        except Exception:
            df = None
        n = 0 if df is None else len(df)
        if self.cfg.use_quotes == "always" or not self._has_bars(sym):
            return df if n else None
        o_ns, c_ns = (int(x.value) for x in self.session_open_close)
        rth = max(min(hi, c_ns) - max(lo, o_ns), 0)          # RTH time covered by the window
        hours = rth / 3.6e12
        need = max(1.0, self.cfg.quote_min_per_hour * hours)
        if n >= need:
            return df
        if n:
            self._sparse_quote_note = True
        return None

    def _roll_to_session(self, d: pd.Timestamp) -> pd.Timestamp:
        d = pd.Timestamp(d).normalize()
        return d if self.cal.is_session(d) else self.cal.next_session(d)

    def _session_of_ns(self, t_ns: int) -> pd.Timestamp:
        return pd.Timestamp(t_ns, tz="UTC").tz_convert(ET).tz_localize(None).normalize()

    # ================================================================= timeline
    def _steps(self, session: pd.Timestamp) -> np.ndarray:
        o, c = self.cal.open_close(session)
        if self.daily:
            return np.array([int(c.value)], dtype=np.int64)
        segs = []
        if "PRE" in self.timeline_sessions:
            segs.append((int(et(session, "04:00").value), int(o.value)))
        segs.append((int(o.value), int(c.value)))
        if "POST" in self.timeline_sessions:
            post_end = int(c.value) + 4 * 3600 * 10**9
            segs.append((int(c.value), post_end))
        step = int(self.step_td.value)
        out = []
        for a, b in segs:
            starts = np.arange(a, b, step, dtype=np.int64)
            ends = np.minimum(starts + step, b)
            out.append(ends)
        return np.concatenate(out) if out else np.zeros(0, np.int64)

    # ================================================================= requirements
    def _check_requirements(self, sessions=None) -> tuple[bool, str, dict]:
        """Critical requirements must be available (portal.has + fidelity).

        * symbols: ``Requirement.symbols`` or the strategy universe; a
          ``universe_name`` is resolved through ``portal.universe(name, as_of=
          first session open)`` and checked per symbol;
        * ``symbols=("*",)`` checks portal-wide kinds: ``portal.has("*", kind,
          freq)`` (e.g. kind 'events' with freq = event KIND, kind
          'fundamentals' with freq = FIELD);
        * bars requirements with extended sessions (PRE/POST) are probed: the
          portal must return bars of that session in the run's first sessions."""
        fidmap: dict[str, Optional[int]] = {}
        syms_default = list(self.strategy.universe)
        if self.strategy.universe_name and sessions is not None and len(sessions):
            members = None
            try:
                o0 = self.cal.open_close(sessions[0])[0]
                members = list(self.portal.universe(self.strategy.universe_name, o0))
            except Exception:
                members = None
            if members:
                # universe_name: per-member data is NOT required of every member; members
                # without verified data are excluded and coverage must reach the minimum
                kept = self._coverage_check(self.strategy.universe_name, o0, members, raise_=False)
                if self._coverage_fail:
                    return False, self._coverage_fail, fidmap
                syms_default = kept
        for r in self.strategy.requires:
            syms = list(r.symbols) if r.symbols else syms_default
            for s_ in syms:
                key = f"{r.kind}:{s_}:{r.freq or ''}"
                try:
                    has = bool(self.portal.has(s_, r.kind, r.freq))
                    f = self.portal.fidelity_of(s_, r.kind, r.freq) if has else Fidelity.NA
                except Exception:   # pragma: no cover - portal bug
                    has, f = False, Fidelity.NA
                f = Fidelity(int(f)) if f is not None else Fidelity.NA
                fidmap[key] = int(f)
                if r.critical and (not has or f == Fidelity.NA or f > r.min_fidelity):
                    why = f"required {r.kind} {r.freq or ''} for {s_} unavailable (fidelity {f.name}"
                    why += f", need {Fidelity(int(r.min_fidelity)).name} or better)" if has and f != Fidelity.NA else ")"
                    return False, why + (f": {r.note}" if r.note else ""), fidmap
                ext = [x for x in r.sessions if x in ("PRE", "POST")]
                if r.kind in ("bars", "quotes", "options") and ext and s_ != "*" and r.critical \
                        and sessions is not None and len(sessions):
                    missing = self._probe_sessions(s_, r.freq or self.base_freq, ext, sessions, kind=r.kind)
                    if missing:
                        fidmap[key + ":" + ",".join(missing)] = int(Fidelity.NA)
                        what = f"{r.freq or self.base_freq} bars" if r.kind == "bars" else r.kind
                        return False, (f"required {'/'.join(missing)} {what} for {s_} not "
                                       f"available in the backtest window"), fidmap
        return True, "", fidmap

    # ---------------------------------------------------------- universe coverage
    def _member_verified(self, sym: str) -> tuple[bool, str]:
        """A universe member is usable when the portal has its base-frequency bars and
        every critical requirement that applies to universe members (no explicit
        ``symbols``) is available at the required fidelity. Memoised per symbol."""
        v = self._verified.get(sym)
        if v is not None:
            return v
        checks = [("bars", self.base_freq, Fidelity.T3_MODELED)]
        checks += [(r.kind, r.freq, r.min_fidelity) for r in self.strategy.requires if r.critical and not r.symbols]
        v = (True, "")
        for kind, freq, mf in checks:
            try:
                has = bool(self.portal.has(sym, kind, freq))
                f = self.portal.fidelity_of(sym, kind, freq) if has else Fidelity.NA
                f = Fidelity(int(f)) if f is not None else Fidelity.NA
            except Exception:
                has, f = False, Fidelity.NA
            if not has or f == Fidelity.NA or f > mf:
                v = (False, f"{kind} {freq or ''} unavailable (fidelity {f.name})".replace("  ", " "))
                break
        self._verified[sym] = v
        return v

    def _coverage_check(self, name: str, as_of, members: list, raise_: bool = True) -> list:
        kept, dropped = [], []
        for s_ in members:
            ok, why = self._member_verified(s_)
            (kept if ok else dropped).append(s_)
        d = str(self._session_of_ns(int(pd.Timestamp(as_of).value)).date())
        key = (name, d)
        if key not in self._cov_log:
            self._cov_log[key] = (len(members), len(kept), dropped)
            cov = len(kept) / len(members) if members else 1.0
            if cov < 1.0:
                tag = (name, round(cov, 3))
                if tag not in self._cov_flagged:
                    self._cov_flagged.add(tag)
                    self.fid.flag(pd.Timestamp(as_of), Fidelity.T2_PROXY,
                                  f"universe {name} coverage {cov:.1%} (members without verified data excluded)")
            if cov < self.cfg.min_universe_coverage and self._coverage_fail is None:
                ex = ", ".join(f"{s_}: {self._verified[s_][1]}" for s_ in dropped[:3])
                self._coverage_fail = (f"universe {name} coverage {cov:.1%} on {d} ({len(dropped)}/{len(members)} "
                                       f"members without verified data, e.g. {ex}) is below "
                                       f"min_universe_coverage {self.cfg.min_universe_coverage:.0%}")
                if raise_:
                    raise NotTestableError(self._coverage_fail)
        return kept

    def _filter_universe(self, name: str, as_of, members: list) -> list:
        """PITPortal.universe hook: the strategy only sees members with verified data."""
        return self._coverage_check(name, as_of, members)

    def _coverage_meta(self) -> dict:
        if not self._cov_log:
            return {}
        rows = [{"date": d, "universe": n, "members": m, "kept": k, "coverage": (k / m if m else float("nan"))}
                for (n, d), (m, k, _) in sorted(self._cov_log.items(), key=lambda kv: kv[0][1])]
        dropped: dict[str, dict] = {}
        for (n, d), (_, _, dr) in sorted(self._cov_log.items(), key=lambda kv: kv[0][1]):
            for s_ in dr:
                e = dropped.setdefault(s_, {"first": d, "last": d, "n": 0, "reason": self._verified[s_][1]})
                e["last"] = d
                e["n"] += 1
        cov = [r["coverage"] for r in rows if r["coverage"] == r["coverage"]]
        return {"min_universe_coverage": self.cfg.min_universe_coverage,
                "coverage_min": min(cov) if cov else float("nan"),
                "series": pd.DataFrame(rows), "dropped": dropped}

    def _probe_sessions(self, sym: str, freq: str, ext: list, sessions, n_probe: int = 10,
                        kind: str = "bars") -> list:
        """Engine-internal availability probe for extended-hours bars / quotes /
        option snapshots over the first ``n_probe`` sessions of the run (no
        strategy-visible data)."""
        probe = list(sessions[:n_probe])
        a = self.cal.open_close(probe[0])[0] - pd.Timedelta(hours=12)
        b = self.cal.open_close(probe[-1])[1] + pd.Timedelta(hours=8)
        missing = []
        if kind == "quotes":
            try:
                df = self.portal.quotes(sym, a, b, as_of=b)
            except Exception:
                df = None
            from .timeutil import as_ns_array
            codes = {self._classify_ts(int(t))[0] for t in as_ns_array(df.index)} if df is not None and len(df) else set()
            return [c for c in ext if c not in codes]
        if kind == "options":
            for code in ext:
                found = False
                for d in probe:
                    o_, c_ = self.cal.open_close(d)
                    lo, hi = ((o_ - pd.Timedelta(hours=5, minutes=30), o_) if code == "PRE"
                              else (c_, c_ + pd.Timedelta(hours=4)))
                    try:
                        ch = self.portal.option_chain(sym, hi - pd.Timedelta(1, "ns"))
                    except Exception:
                        ch = None
                    if ch is not None and len(ch) and "ts" in ch.columns:
                        ts_ = pd.DatetimeIndex(ch["ts"])
                        ts_ = ts_.tz_localize("UTC") if ts_.tz is None else ts_.tz_convert("UTC")
                        if ((ts_ >= lo) & (ts_ < hi)).any():
                            found = True
                            break
                if not found:
                    missing.append(code)
            return missing
        for code in ext:
            try:
                df = self.portal.bars(sym, freq, a, b, as_of=b, adjusted=False, sessions=(code,))
            except Exception:
                df = None
            if df is None or len(df) == 0 or ("session" in df.columns and not (df["session"] == code).any()):
                missing.append(code)
        return missing

    # ================================================================= run
    def run(self, start, end, raise_errors: bool = False) -> BacktestResult:
        t0 = time.perf_counter()
        self._reset_state()
        sessions = self.cal.sessions(pd.Timestamp(start), pd.Timestamp(end))
        params = dict(self.strategy.params)
        if len(sessions) == 0:
            return self._result(RunStatus.FAILED, params, "no sessions in range", t0, {})
        ok, why, reqfid = self._check_requirements(sessions)
        if not ok:
            return self._result(RunStatus.NOT_TESTABLE, params, why, t0, reqfid)
        if not self.daily and self._session_rules.get("flat_by"):
            reg = [d for d in sessions[:10] if not self.cal.is_early_close(d)] or [sessions[0]]
            st_ = self._steps(reg[0])
            fb_ns = self._rule_time_ns(reg[0], self._session_rules["flat_by"])
            if not (st_ <= fb_ns).any():
                return self._result(RunStatus.NOT_TESTABLE, params,
                                    f"bar frequency {self.base_freq} has no step before flat_by "
                                    f"{self._session_rules['flat_by']} ET", t0, reqfid)
        if self.daily and self._session_rules.get("flat_by"):
            return self._result(RunStatus.NOT_TESTABLE, params,
                                f"{self.strategy.sizing or 'intraday'} session rules (flat by "
                                f"{self._session_rules['flat_by']} ET) need an intraday base frequency; "
                                f"daily bars cannot exit before the close", t0, reqfid)
        first_open_ = self.cal.open_close(sessions[0])[0]
        self.account = Account(self.cfg.initial_cash, self.cfg.account_type,
                               start_ts=first_open_ - pd.Timedelta(seconds=1))
        self._run_range = (sessions[0], sessions[-1])
        self.md = MarketData(self.portal, self.cal, sessions[0], sessions[-1], self.base_freq,
                             self.data_sessions, self.spread_model, self.cfg.lookback_sessions,
                             self.cfg.stats_window)
        self.ctx = EngineContext(self)
        first_open = self.cal.open_close(sessions[0])[0]
        steps0 = self._steps(sessions[0])
        start_ns = int(first_open.value) if self.daily or "PRE" not in self.timeline_sessions \
            else int(steps0[0] - self.step_td.value)
        self.prev_step_ns = start_ns - 1
        self._set_now(start_ns)
        self.session = sessions[0]
        self.session_open_close = self.cal.open_close(sessions[0])
        status = RunStatus.OK
        err = ""
        try:
            self.strategy.init(self.ctx)
            for i, s in enumerate(sessions):
                self.prev_session = sessions[i - 1] if i > 0 else None
                self._session_start(s)
                for t_ns in self._steps(s):
                    self._step(int(t_ns))
                    if self._coverage_fail:              # even if the strategy swallowed the exception
                        raise NotTestableError(self._coverage_fail)
                self._session_end(s, sessions[i + 1] if i + 1 < len(sessions) else self.cal.next_session(s))
                if self._coverage_fail:
                    raise NotTestableError(self._coverage_fail)
            self.strategy.on_run_end(self.ctx)
            self._flush_on_fill()
        except LookaheadError:
            raise
        except NotTestableError as e:              # a designed outcome, not an error (even with raise_errors)
            status = RunStatus.NOT_TESTABLE
            err = str(e) or "NotTestableError raised by strategy"
            self.diag["not_testable_raised_at"] = str(self.now)
        except Exception as e:
            if raise_errors or self.cfg.debug:
                raise
            status = RunStatus.FAILED
            err = f"{type(e).__name__}: {e}"
            self.diag["traceback"] = traceback.format_exc()
        res = self._result(status, params, err, t0, reqfid)
        return res

    # ================================================================= session start
    def _session_start(self, session: pd.Timestamp) -> None:
        self.session = session
        self.session_open_close = self.cal.open_close(session)
        o, c = self.session_open_close
        steps = self._steps(session)
        first_start = int(o.value) if self.daily else int(steps[0] - self.step_td.value)
        self._set_now(max(first_start, self.now_ns))
        self._flattened_session = False
        self._borrow_cache.clear()
        self._setup_session_rules(session, steps)
        ts = self.now
        # 1. dividends due
        for a in self.ca.pay_due(self.account, session, ts):
            self.diag["corporate_actions"].append({"ts": ts, "symbol": a.symbol, "type": f"PAY_{a.kind}",
                                                   "amount": a.amount})
        # 2. corporate actions with ex_date in (prev_session, session]
        syms = set(self.account.positions) | {s_ for s_ in self.om.active_symbols() if not s_.startswith("COMBO:")}
        stock_syms = set()
        for s_ in syms:
            inst = self.instrument(s_)
            stock_syms.add(inst.underlying.upper() if inst.is_option and inst.underlying else s_)
        for o_ in self.om.active():
            for leg in (o_.legs or ()):
                if leg.instrument.is_option and leg.instrument.underlying:
                    stock_syms.add(leg.instrument.underlying.upper())
        stock_syms = sorted(stock_syms)
        if stock_syms:
            prev = self.prev_session if self.prev_session is not None else self.cal.prev_session(session)
            try:
                acts = self.portal.corporate_actions(stock_syms, prev, session + pd.Timedelta(days=1), as_of=c)
            except Exception as e:
                acts = []
                self.diag["warnings"].append(f"corporate_actions query failed: {e}")
            due = []
            for a in acts:
                exd = as_session_date(a.ex_date)
                if prev < exd <= session:
                    due.append(a)
                    if exd != session:
                        self.diag["warnings"].append(
                            f"{a.type.value} {a.symbol} ex_date {exd.date()} is not a session: applied on {session.date()}")
            for a in sorted(due, key=lambda a: (a.symbol, a.type.value, as_session_date(a.ex_date))):
                self._apply_action(a, session, ts)
            # dividends with pay_date <= ex_date are due immediately
            for a in self.ca.pay_due(self.account, session, ts):
                self.diag["corporate_actions"].append({"ts": ts, "symbol": a.symbol, "type": f"PAY_{a.kind}",
                                                       "amount": a.amount})
        # 3. halts for symbols with orders
        self._load_halts(session)
        # 4. AM-settled options
        opts = self.account.option_positions()
        if opts:
            evs = self.lifecycle.at_session_start(session, self.prev_session, opts, _Prices(self))
            self._apply_option_events(evs)
        # 5. strategy hook
        self._flush_on_fill()
        self.strategy.on_session_start(self.ctx)
        self._flush_on_fill()

    def _rule_time_ns(self, session: pd.Timestamp, hhmm: Optional[str]) -> Optional[int]:
        """'HH:MM' rules are defined on a regular 16:00 ET session; on early
        closes they keep the same distance to the actual close."""
        if not hhmm:
            return None
        h, m = (int(x) for x in hhmm.split(":"))
        before_close = (16 * 60) - (h * 60 + m)
        _, c = self.cal.open_close(session)
        return int((c - pd.Timedelta(minutes=before_close)).value)

    def _setup_session_rules(self, session: pd.Timestamp, steps: np.ndarray) -> None:
        self._flat_by_ns = self._rule_time_ns(session, self._session_rules.get("flat_by"))
        self._no_entry_ns = self._rule_time_ns(session, self._session_rules.get("no_new_entries_after"))
        self._flat_step_ns = None
        self._flat_moc_mode = False
        if self._flat_by_ns is not None and not self.daily:
            ok = steps[steps <= self._flat_by_ns]
            if len(ok):
                self._flat_step_ns = int(ok[-1])
            else:
                self._flat_moc_mode = True

    def _apply_action(self, a: CorporateAction, session, ts) -> None:
        pos_before = self.account.position(a.symbol)
        self._orders_on_action(a, ts)
        res = self.ca.apply(a, self.account, ts, session, instrument_lookup=self.instrument,
                            price_lookup=lambda s_: self.last_price(s_))
        t = a.type.value
        if res.renamed_to:
            self._instruments.pop(res.renamed_to, None)
            self.trades.rekey(a.symbol, res.renamed_to)
        if t == "SPLIT" and res.split_ratio != 1.0:
            self.trades.on_split(a.symbol, res.split_ratio, res.cil_qty, res.cil_price, ts)
            self._adjust_options_for_split(a.symbol, res.split_ratio, ts)
        if t in ("CASH_DIVIDEND", "SPECIAL_DIVIDEND") and res.accrual is not None:
            self.trades.attribute(a.symbol, res.accrual.amount)
        if t == "SPINOFF" and res.spinoff_symbol:
            frac = float(a.amount) if 0.0 <= float(a.amount) < 1.0 else 0.0
            self.trades.on_spinoff(a.symbol, res.spinoff_symbol, res.spinoff_qty, res.spinoff_basis, frac, ts)
        self.diag["corporate_actions"].append({"ts": ts, "symbol": a.symbol, "type": t,
                                               "applied": res.applied, "note": res.note,
                                               "fidelity": Fidelity(res.fidelity).name})
        if res.fidelity >= Fidelity.T2_PROXY:
            self.fid.flag(ts, res.fidelity, f"corporate action {t}: {res.note}")
        if t == "DELIST" and res.closed_qty:
            inst = self.instrument(a.symbol)
            f = Fill(-1, inst, ts, res.closed_qty, res.close_price, CostBreakdown(), res.fidelity,
                     liquidity="CORP_ACTION", note=res.note)
            self.trades.on_fill(f, pos_before, None, "delist")
            self._record_fill(f, "delist", ("delisting cash-out",))

    def _order_event(self, action: str, o: Order, reason: str, **kw) -> None:
        ev = {"ts": self.now, "order_id": o.id, "symbol": self.om.symbol_of(o), "action": action,
              "reason": reason, "tag": o.tag, **kw}
        self.diag["order_events"].append(ev)
        self._pending_events.append(ev)

    def _cancel_with_event(self, o: Order, reason: str, status: OrderStatus = OrderStatus.CANCELLED) -> None:
        if self.om.cancel(o.id, self.now, reason, status):
            self._order_event("CANCELLED", o, reason)

    def _orders_on_action(self, a: CorporateAction, ts) -> None:
        """FINRA Rule 5330-style handling of resting orders on the ex-date:
        forward split -> sell stops / sell stop-limits / buy limits adjusted
        (price / ratio, quantity x ratio, bracket links kept), other orders
        cancelled; reverse split, symbol change, spin-off, delisting -> all
        orders on the symbol cancelled. Every change is delivered to the
        strategy (``on_order_event``) and logged in diagnostics."""
        t = a.type.value
        if t not in ("SPLIT", "SYMBOL_CHANGE", "SPINOFF", "DELIST"):
            return
        orders = self.om.active(a.symbol)
        if not orders:
            return
        r = float(a.ratio or 1.0)
        if t == "SPLIT" and r == 1.0:
            return
        for o in orders:
            buy = o.quantity > 0
            adjustable = t == "SPLIT" and r > 1.0 and (
                (o.type in (OrderType.STOP, OrderType.STOP_LIMIT) and not buy) or (o.type == OrderType.LIMIT and buy))
            if not adjustable:
                why = (f"{t} {a.symbol} (ratio {r:g}): order cancelled" if t == "SPLIT"
                       else f"{t} {a.symbol}: order cancelled")
                self._cancel_with_event(o, why)
                continue
            before = {"quantity": o.quantity, "limit_price": o.limit_price, "stop_price": o.stop_price}
            rem = o.remaining * r
            if not self.cfg.bar_fill.allow_fractional:
                rem = float(math.trunc(rem + (1e-9 if rem > 0 else -1e-9)))
            o.filled_qty *= r
            o.quantity = o.filled_qty + rem
            ref = (o.limit_price or o.stop_price or 1.0) / r
            tick = valid_tick(ref, o.instrument)
            if o.limit_price is not None:
                o.limit_price = round_limit_for_side(o.limit_price / r, tick, buy)
            if o.stop_price is not None:
                o.stop_price = round_stop_for_side(o.stop_price / r, tick, buy)
            ex = self.om.extras[o.id]
            if ex.initial_stop is not None:
                ex.initial_stop /= r
            if ex.planned_entry is not None:
                ex.planned_entry /= r
            spec = self.om.bracket_of(o.id)
            if spec is not None:
                for tpl in spec.templates:
                    if tpl.limit_price is not None:
                        tpl.limit_price /= r
                    if tpl.stop_price is not None:
                        tpl.stop_price /= r
            self._order_event("ADJUSTED", o, f"split {r:g} on {a.symbol}: price / {r:g}, quantity x {r:g}",
                              before=before, after={"quantity": o.quantity, "limit_price": o.limit_price,
                                                    "stop_price": o.stop_price})

    def _adjust_options_for_split(self, und: str, r: float, ts) -> None:
        """OCC contract adjustment of option positions and working option orders
        on a split of the underlying (OPTIONS agent's adjust_for_split)."""
        from ..options.lifecycle import adjust_for_split
        und = und.upper()
        for p in list(self.account.option_positions()):
            inst = p.instrument
            if (inst.underlying or "").upper() != und:
                continue
            new_inst, new_qty = adjust_for_split(inst, p.qty, r)
            if not new_qty:
                continue
            scale = p.qty / new_qty
            old_sym = inst.symbol
            old_qty = p.qty
            self.account.positions.pop(old_sym, None)
            p.instrument = new_inst
            p.qty = new_qty
            p.avg_price *= scale
            if p.mark == p.mark:
                p.mark *= scale
            p.mark_ts = ts
            other = self.account.positions.get(new_inst.symbol)
            if other is not None:
                tot = other.qty + p.qty
                other.avg_price = (other.qty * other.avg_price + p.qty * p.avg_price) / tot if tot else 0.0
                other.qty = tot
                if other.qty == 0:
                    del self.account.positions[new_inst.symbol]
            else:
                self.account.positions[new_inst.symbol] = p
            self._instruments[new_inst.symbol] = new_inst
            self.trades.on_split(old_sym, new_qty / old_qty)
            self.trades.rekey(old_sym, new_inst.symbol)
            self.diag["option_events"].append({"ts": ts, "kind": "OCC_SPLIT_ADJUSTMENT", "option": old_sym,
                                               "new_option": new_inst.symbol, "qty": old_qty, "new_qty": new_qty,
                                               "deliverable": new_inst.deliverable_shares, "note": f"split {r:g}"})
        for o in list(self.om.active()):
            legs = o.legs if o.type == OrderType.COMBO else ((Leg(o.instrument, 1),) if o.instrument is not None else ())
            if not any(l.instrument.is_option and (l.instrument.underlying or "").upper() == und for l in legs):
                continue
            old_key = self.om.symbol_of(o)
            before = {"instrument": old_key, "quantity": o.quantity, "limit_price": o.limit_price}
            if o.type == OrderType.COMBO:
                new_legs = []
                for l in o.legs:
                    if l.instrument.is_option and (l.instrument.underlying or "").upper() == und:
                        ni, nr = adjust_for_split(l.instrument, float(l.ratio), r)
                        new_legs.append(Leg(ni, int(round(nr))))
                        self._instruments[ni.symbol] = ni
                    else:
                        new_legs.append(l)
                o.legs = tuple(new_legs)
            else:
                ni, nq = adjust_for_split(o.instrument, o.quantity, r)
                scale = nq / o.quantity
                o.instrument = ni
                o.filled_qty *= scale
                o.quantity = nq
                if o.limit_price is not None:
                    o.limit_price /= scale
                self._instruments[ni.symbol] = ni
            self.om.rekey(o, old_key)
            self._order_event("ADJUSTED", o, f"OCC adjustment for {und} split {r:g}", before=before,
                              after={"instrument": self.om.symbol_of(o), "quantity": o.quantity,
                                     "limit_price": o.limit_price})

    def _load_halts(self, session) -> None:
        self._halts = {}
        self._halts_loaded: set[str] = set()

    def _halt_intervals(self, sym: str) -> list[tuple[int, int]]:
        """HALT events for `sym` around the current session (engine-internal:
        a halt blocks fills during its interval; it is never shown to the
        strategy before its known_at)."""
        if not self.cfg.halt_events:
            return []
        if sym in self._halts_loaded:
            return self._halts.get(sym, [])
        self._halts_loaded.add(sym)
        o, c = self.session_open_close
        try:
            evs = self.portal.events("HALT", [sym], o - pd.Timedelta(hours=12), c + pd.Timedelta(hours=8),
                                     as_of=c + pd.Timedelta(hours=8))
        except Exception:
            evs = []
        for e in evs or []:
            st = pd.Timestamp(e.effective_at if e.effective_at is not None else e.known_at)
            res = e.payload.get("resume_at") if e.payload else None
            en = pd.Timestamp(res) if res is not None else c + pd.Timedelta(hours=8)
            if st.tzinfo is None:
                st = st.tz_localize("UTC")
            if en.tzinfo is None:
                en = en.tz_localize("UTC")
            self._halts.setdefault(sym, []).append((int(st.value), int(en.value)))
        return self._halts.get(sym, [])

    # ================================================================= step
    def _step(self, t_ns: int) -> None:
        prev = self.prev_step_ns
        self._set_now(t_ns)
        if self.om.active_symbols():
            self._process_fills(prev, t_ns)
        self._flush_on_fill()          # on_fill / on_order_event: same step, ctx.now = t, before marking
        self._mark(t_ns)
        self._enforce_session_rules()
        self.strategy.on_bar(self.ctx)
        self._flush_on_fill()
        if self.trades.open:
            self._update_excursions(prev, t_ns)
        if self.cfg.record_intraday_equity:
            self.intraday_equity.append((self.now, self.account.nav))
        if self.cfg.debug:
            self._check_invariants()
        self.prev_step_ns = t_ns

    def _flush_on_fill(self) -> None:
        while self._pending_events:
            ev = self._pending_events.pop(0)
            self.strategy.on_order_event(self.ctx, ev)
        while self._pending_on_fill:
            f = self._pending_on_fill.pop(0)
            self.strategy.on_fill(self.ctx, f)

    # ================================================================= fills
    def _process_fills(self, lo: int, hi: int) -> None:
        for sym in self.om.active_symbols():
            orders = self.om.active(sym)
            if not orders:
                continue
            if sym.startswith("COMBO:") or orders[0].instrument is None or orders[0].instrument.is_option:
                self._fill_snapshot_orders(orders, lo, hi)
                continue
            inst = self.instrument(sym)
            self._clamp_reduce_only(sym)
            # chronological: opening auction, continuous trading, closing auction
            # (MOC children created by fills in this window, e.g. a bracket
            # time_exit_moc on a daily bar, still join this window's close)
            moo = [o for o in self.om.active(sym) if o.type == OrderType.MOO]
            if moo:
                self._fill_auction(sym, inst, moo, lo, hi)
            rest = [o for o in self.om.active(sym) if o.type not in (OrderType.MOO, OrderType.MOC)]
            if rest:
                self._sparse_quote_note = False
                qdf = self._quote_window(sym, lo, hi)
                if qdf is not None:
                    self._fill_quotes(sym, inst, lo, hi, qdf)
                else:
                    self._fill_bars(sym, inst, lo, hi)
            moc = [o for o in self.om.active(sym) if o.type == OrderType.MOC]
            if moc:
                self._clamp_reduce_only(sym)
                self._fill_auction(sym, inst, moc, lo, hi)
            if not rest:
                continue
            # IOC: cancel remainder after first evaluation
            for o in self.om.active(sym):
                ex = self.om.extras[o.id]
                if o.tif == TimeInForce.IOC and ex.evaluated:
                    self.om.cancel(o.id, self.now, "IOC remainder cancelled")

    def _clamp_reduce_only(self, sym: str) -> None:
        """Exit orders (bracket children, engine flatten/liquidation) never open
        or flip a position: shrink them to the position, cancel when flat."""
        key = (self._state_ver, self.account.position(sym))
        if self._clamp_seen.get(sym) == key:
            return
        self._clamp_seen[sym] = key
        pos = key[1]
        for o in self.om.active(sym):
            ex = self.om.extras[o.id]
            if not ex.reduce_only:
                continue
            rem = o.remaining
            if pos == 0 or (pos > 0) == (rem > 0):
                self._cancel_with_event(o, "reduce-only exit cancelled: no position left to reduce")
                continue
            if abs(rem) > abs(pos) + 1e-9:
                old = o.quantity
                o.quantity = o.filled_qty - pos
                self._order_event("ADJUSTED", o, "reduce-only exit shrunk to the remaining position",
                                  before={"quantity": old}, after={"quantity": o.quantity})

    def _short_sale_state(self, sym: str):
        """Fill-time short-sale classification: a sell is a short sale if the
        current position plus EARLIER-entered pending sells (not exits that
        can only reduce) plus this order goes below zero. Locate follows."""
        def state(o: Order) -> tuple[bool, bool]:
            pos = self.account.position(sym)
            earlier = sum(x.remaining for x in self.om.active(sym)
                          if x.id < o.id and x.remaining < 0 and not self.om.extras[x.id].reduce_only)
            short = pos + earlier + o.remaining < -1e-9
            if not short:
                return False, True
            if self.account.account_type == AccountType.CASH:
                return True, False
            if self.cfg.enforce_locates:
                ok, _, _ = self._borrow(sym)
                return True, ok
            return True, True
        return state

    def _symbol_market(self, sym: str, inst: Instrument) -> SymbolMarket:
        st = self.md.daily(sym)
        est, src, adv, sig = float("nan"), "default", float("nan"), float("nan")
        idx = -1
        if st is not None:
            idx = st.idx(int(self.session.value))
            if idx < 0:
                idx = int(np.searchsorted(st.session_ns, int(self.session.value), side="left"))
            if 0 <= idx < st.bars.n:
                est, src = st.spread.at(idx)
                adv = float(st.adv_dollar[idx])
                sig = float(st.sigma[idx])
        rules = self.cfg.short_sale_rules

        def ssr(bar: BarView, _st=st, _idx=idx) -> bool:
            era = rules.era(self.session)
            if era == "TICK_TEST":
                return True
            if era == "NONE" or _st is None or _idx < 0 or _idx >= _st.bars.n:
                return False
            if _idx >= 1 and _st.bars.l[_idx - 1] <= _st.prior_close[_idx - 1] * (1 - rules.trigger_decline):
                return True
            pc = _st.prior_close[_idx]
            if not (pc == pc):
                return False
            if self.daily:
                low = bar.l
            else:
                b = self.md.bars(sym)
                o_ns = int(self.session_open_close[0].value)
                i0 = int(np.searchsorted(b.start, o_ns, side="left"))
                i1 = int(np.searchsorted(b.end, min(bar.end, self.now_ns), side="right"))
                low = float(np.min(b.l[i0:i1])) if i1 > i0 else bar.l
                low = min(low, bar.l)
            return rules.triggered(pc, low)

        return SymbolMarket(inst, est, src, adv, sig, ssr, lambda: self.account.position(sym),
                            self.daily, self.cfg.daily_open_spread_mult, self._short_sale_state(sym))

    def _bar_views(self, b, i0: int, i1: int, sym: str) -> list[BarView]:
        o, c = self.session_open_close
        o_ns, c_ns = int(o.value), int(c.value)
        halts = self._halt_intervals(sym)
        out = []
        for i in range(i0, i1):
            st, en = int(b.start[i]), int(b.end[i])
            v = float(b.v[i])
            for hs, he in halts:
                if st < he and en > hs:
                    v = 0.0
            sess = {0: "RTH", 1: "PRE", 2: "POST", 3: "DAY"}[int(b.sess[i])]
            if sess == "DAY":
                out.append(BarView(st, en, b.o[i], b.h[i], b.l[i], b.c[i], v, sess, None, None))
            else:
                out.append(BarView(st, en, b.o[i], b.h[i], b.l[i], b.c[i], v, sess,
                                   (st - o_ns) / NS_MIN, (c_ns - en) / NS_MIN))
        return out

    def _orders_getter(self, sym: str):
        om = self.om

        def get():
            self._clamp_reduce_only(sym)
            return [(o, om.extras[o.id]) for o in om.active(sym)
                    if o.type not in (OrderType.MOO, OrderType.MOC)]
        return get

    def _needs_resolution(self, bar: BarView, orders: list[Order]) -> bool:
        n = 0
        for o in orders:
            buy = o.quantity > 0
            t = o.type
            if t == OrderType.MARKET:
                n += 1
            elif t == OrderType.LIMIT:
                if (buy and bar.l <= o.limit_price) or ((not buy) and bar.h >= o.limit_price):
                    n += 1
            elif t in (OrderType.STOP, OrderType.STOP_LIMIT):
                if (buy and bar.h >= o.stop_price) or ((not buy) and bar.l <= o.stop_price):
                    n += 1
            if self.om.bracket_of(o.id) is not None:
                n += 1
        return n >= 2

    def _fill_bars(self, sym: str, inst: Instrument, lo: int, hi: int) -> None:
        b, i0, i1 = self.md.window(sym, lo, hi)
        if i1 <= i0:
            return
        mkt = self._symbol_market(sym, inst)
        get = self._orders_getter(sym)
        views = self._bar_views(b, i0, i1, sym)
        fine = self.cfg.resolve_intrabar_freq
        can_resolve = bool(fine) and freq_to_timedelta(fine) < self.step_td
        for bv in views:
            orders = [o for o, _ in get()]
            if not orders:
                break
            resolved = None
            seq = [bv]
            if can_resolve and not bar_is_halted(bv.v, bv.sess, bv.o) and self._needs_resolution(bv, orders):
                try:
                    has_fine = self.portal.has(sym, "bars", fine)
                except Exception:
                    has_fine = False
                if has_fine:
                    df = self.portal.bars(sym, fine, ts_utc(bv.start), ts_utc(bv.end), as_of=self.now,
                                          adjusted=False, sessions=self.data_sessions)
                    if df is not None and len(df):
                        from .market import BarArrays
                        fb = BarArrays(sym, fine, df)
                        seq = self._bar_views(fb, 0, fb.n, sym)
                        resolved = canonical_freq(fine)
                        self.diag["sim"]["resolved_bars"] += 1
            stats = self.bar_model.simulate(mkt, seq, get, lambda d: self._on_decision(d, inst),
                                            lambda o, r: self._reject_working(o, r), resolved_with=resolved)
            self._add_stats(stats)

    def _fill_quotes(self, sym: str, inst: Instrument, lo: int, hi: int, df=None) -> None:
        if df is None:
            df = self.portal.quotes(sym, ts_utc(lo + 1), ts_utc(hi + 1), as_of=self.now)
        if df is None or len(df) == 0:
            return
        from .timeutil import as_ns_array
        tsa = as_ns_array(df.index)
        bid = df["bid"].to_numpy(float)
        ask = df["ask"].to_numpy(float)
        bs = df["bid_size"].to_numpy(float) if "bid_size" in df else np.full(len(df), np.nan)
        as_ = df["ask_size"].to_numpy(float) if "ask_size" in df else np.full(len(df), np.nan)
        halts = self._halt_intervals(sym)
        qs = []
        for i in range(len(df)):
            t = int(tsa[i])
            if not (lo < t <= hi) or any(hs <= t < he for hs, he in halts):
                continue
            code, sess_ns = self._classify_ts(t)
            if code == "CLOSED":
                continue
            qs.append(QuoteView(t, bid[i], ask[i], bs[i], as_[i], code, sess_ns))
        mkt = self._symbol_market(sym, inst)
        stats = self.quote_model.simulate(mkt, qs, self._orders_getter(sym),
                                          lambda d: self._on_decision(d, inst),
                                          lambda o, r: self._reject_working(o, r))
        self._add_stats(stats)

    def _classify_ts(self, t_ns: int) -> tuple[str, int]:
        """Calendar session classification of a timestamp: (PRE|RTH|POST|CLOSED, session ns)."""
        d = self._session_of_ns(t_ns)
        key = int(d.value)
        oc = self._oc_cache.get(key)
        if oc is None:
            oc = self.cal.open_close(d) if self.cal.is_session(d) else None
            oc = (int(oc[0].value), int(oc[1].value)) if oc is not None else (0, 0)
            self._oc_cache[key] = oc
        o, c = oc
        if o == 0:
            return "CLOSED", key
        if t_ns < o:
            return ("PRE" if t_ns >= o - int(5.5 * 3600e9) else "CLOSED"), key
        if t_ns < c:
            return "RTH", key
        return ("POST" if t_ns < c + 4 * 3600 * 10**9 else "CLOSED"), key

    def _fill_auction(self, sym: str, inst: Instrument, orders: list[Order], lo: int, hi: int) -> None:
        o_ts, c_ts = self.session_open_close
        sess_ns = int(self.session.value)
        b, i0, i1 = self.md.window(sym, lo, hi)
        mkt = None
        for od in orders:
            ex = self.om.extras[od.id]
            if ex.target_session_ns != sess_ns:
                continue
            is_open = od.type == OrderType.MOO
            target = int(o_ts.value) if is_open else int(c_ts.value)
            k = None
            for i in range(i0, i1):
                if b.sess[i] in (0, 3) and ((is_open and b.start[i] == target) or ((not is_open) and b.end[i] == target)):
                    k = i
                    break
            if k is None:
                continue
            if bar_is_halted(float(b.v[k]), "RTH", float(b.o[k])):
                continue
            if ex.reduce_only:
                self._clamp_reduce_only(sym)
                if od.status not in ACTIVE:
                    continue
            mkt = mkt or self._symbol_market(sym, inst)
            daily_bar = int(b.sess[k]) == 3
            px = float(b.o[k] if is_open else b.c[k])
            ts_ns = int(b.start[k]) if is_open else int(b.end[k])
            dec = self.auction_model.fill(od, px, float(b.v[k]), ts_ns, mkt, is_open, daily_bar)
            if dec is not None:
                self._on_decision(dec, inst)
            if od.status in ACTIVE:
                self.om.cancel(od.id, self.now, "auction remainder cancelled", OrderStatus.EXPIRED)

    def _fill_snapshot_orders(self, orders: list[Order], lo: int, hi: int) -> None:
        """Options / combos: evaluate every quote snapshot in (eligible_at, now]
        in time order (first snapshot after the order first), all option legs
        from the SAME snapshot. Stock legs are priced from the first bar that
        STARTS at/after the order became eligible (its open +/- half-spread +
        impact), never from a close known at submission."""
        for od in orders:
            ex = self.om.extras[od.id]
            if ex.eligible_ns >= hi or od.status not in ACTIVE:
                continue
            legs = od.legs if od.type == OrderType.COMBO else (Leg(od.instrument, 1),)
            start_ns = max(ex.eligible_ns, ex.last_snap_ns)
            opt_legs = [l for l in legs if l.instrument.is_option]
            frames = {}
            for leg in opt_legs:
                frames[leg.instrument.symbol] = self._option_snapshots(leg.instrument, start_ns)
            if opt_legs:
                ts_sets = [set(f.keys()) for f in frames.values()]
                common = sorted(set.intersection(*ts_sets)) if ts_sets else []
                if not common:
                    if all(ts_sets) and len(ts_sets) > 1:
                        self._reject_working(od, "legs not from the same snapshot")
                    continue
            else:
                common = [hi]
            ex.evaluated = True
            done = False
            for snap in common:
                lq: list[tuple[Leg, Optional[LegQuote]]] = []
                for leg in legs:
                    if leg.instrument.is_option:
                        bid, ask, fid = frames[leg.instrument.symbol][snap]
                        lq.append((leg, LegQuote(leg.instrument, snap, bid, ask, fid)))
                    else:
                        lq.append((leg, self._stock_leg_quote(leg, od, ex.eligible_ns, snap)))
                status, leg_fills, net, reason = self.combo_model.evaluate(od, lq, ex.eligible_ns)
                ex.last_snap_ns = snap
                if status == "REJECT":
                    self._reject_working(od, reason)
                    done = True
                    break
                if status != "FILL":
                    continue
                units = abs(leg_fills[0][1] / (legs[0].ratio or 1)) if od.type == OrderType.COMBO else abs(leg_fills[0][1])
                for k, (inst, q, px, mid, fid) in enumerate(leg_fills):
                    note_extra = lq[k][1].note if lq[k][1] is not None else ""
                    reasons = (reason, "option quote snapshot" if inst.is_option else "stock leg") + \
                        ((note_extra,) if note_extra else ())
                    dec = FillDecision(od, 0.0, snap, q, px, mid, "TAKER", abs(px - mid) * abs(q) * inst.multiplier,
                                       0.0, fid, reasons)
                    self._book(dec, inst, fee_key=od.id * 100 + k + 10**9 if od.type == OrderType.COMBO else od.id,
                               notify_manager=False)
                sgn = 1 if od.quantity > 0 else -1
                self._after_fill_order(od, sgn * units, pd.Timestamp(snap, tz="UTC"), 0.0, snap)
                done = True
                break
            if not done and od.tif == TimeInForce.IOC and common:
                self.om.cancel(od.id, self.now, "IOC not filled at first snapshot")

    def _option_snapshots(self, inst: Instrument, after_ns: int) -> dict[int, tuple[float, float, Fidelity]]:
        """{ts_ns: (bid, ask, fidelity)} for snapshots in (after_ns, now]."""
        out: dict[int, tuple[float, float, Fidelity]] = {}
        fn = getattr(self.portal, "option_quotes", None)
        if fn is not None:
            try:
                df = fn(inst, ts_utc(after_ns), self.now, as_of=self.now)
            except NotImplementedError:
                df = None
            if df is not None:
                if len(df):
                    from .timeutil import as_ns_array
                    tsa = as_ns_array(df.index)
                    fids = df["fidelity"].to_numpy() if "fidelity" in df.columns else [int(Fidelity.T0_QUOTE)] * len(df)
                    for t, b, a, f in zip(tsa, df["bid"].to_numpy(float), df["ask"].to_numpy(float), fids):
                        if after_ns < int(t) <= self.now_ns:
                            out[int(t)] = (float(b), float(a), Fidelity(int(f)))
                return out
        q = self.portal.option_quote(inst, self.now)     # fallback: latest snapshot only
        if q is not None:
            t = int(pd.Timestamp(q.ts).value)
            if t > after_ns:
                out[t] = (q.bid, q.ask, q.fidelity)
                self.fid.note(Fidelity.T2_PROXY, "portal lacks option_quotes: latest snapshot used")
        return out

    def _stock_leg_quote(self, leg: Leg, od: Order, eligible_ns: int, snap_ns: int) -> Optional[LegQuote]:
        sym = leg.instrument.symbol
        if self._uses_quotes(sym):
            # synchronized stock NBBO at the option snapshot time (T0), when quotes cover it
            age = self.cfg.combo.stock_quote_max_age
            lo_ns = max(eligible_ns, snap_ns - int(age.value))
            try:
                qd = self.portal.quotes(sym, ts_utc(lo_ns + 1), ts_utc(snap_ns + 1), as_of=self.now)
            except Exception:
                qd = None
            if qd is not None and len(qd):
                from .timeutil import as_ns_array
                tsa = as_ns_array(qd.index)
                for j in range(len(qd) - 1, -1, -1):
                    bid_, ask_ = float(qd["bid"].iloc[j]), float(qd["ask"].iloc[j])
                    if lo_ns < tsa[j] <= snap_ns and ask_ > bid_ > 0:
                        return LegQuote(leg.instrument, snap_ns, bid_, ask_, Fidelity.T0_QUOTE,
                                        "stock leg from synchronized NBBO quote")
        b = self.md.bars(sym)
        i = int(np.searchsorted(b.start, eligible_ns, side="left"))
        if i >= b.n or b.start[i] > snap_ns or b.start[i] > self.now_ns:
            return None                                   # first post-submission bar not started yet: WAIT
        px = float(b.o[i])                                # opening print is known at the bar start
        mkt = self._symbol_market(sym, leg.instrument)
        sess = {0: "RTH", 1: "PRE", 2: "POST", 3: "DAY"}[int(b.sess[i])]
        o_ns, c_ns = (int(x.value) for x in self.session_open_close)
        bv = BarView(int(b.start[i]), int(b.end[i]), px, px, px, px, float(b.v[i]), sess,
                     None if sess == "DAY" else (int(b.start[i]) - o_ns) / NS_MIN,
                     None if sess == "DAY" else (c_ns - int(b.end[i])) / NS_MIN)
        hs = self.bar_model.half_spread(mkt, px, bv)
        units = abs(od.remaining) if od.type == OrderType.COMBO else 1
        imp = self.cfg.impact.impact_fraction(abs(leg.ratio) * units * px, mkt.adv_dollar, mkt.sigma)
        tick = valid_tick(px, leg.instrument)
        from ..execution.constraints import round_to_tick
        bid = round_to_tick(px * (1 - hs - imp), tick, "down")
        ask = round_to_tick(px * (1 + hs + imp), tick, "up")
        return LegQuote(leg.instrument, snap_ns, bid, ask, Fidelity.T2_PROXY,
                        "stock leg from first post-submission bar open +/- est. half-spread + impact")

    def _add_stats(self, stats) -> None:
        d = self.diag["sim"]
        d["halted_bars"] += stats.halted_bars
        d["nan_volume_bars"] += stats.nan_volume_bars
        d["ambiguous_bars"] += stats.ambiguous_bars
        d["partial_fills"] += stats.partial_fills
        d["ssr_blocked"] += stats.ssr_blocked

    def _reject_working(self, order: Order, reason: str) -> None:
        if order.status in ACTIVE:
            self.om.cancel(order.id, self.now, reason, OrderStatus.REJECTED)
            self.om.rejected.append({"ts": self.now, "order_id": order.id, "symbol": self.om.symbol_of(order),
                                     "quantity": order.quantity, "type": order.type.value, "tag": order.tag,
                                     "reason": reason})

    # ================================================================= booking
    def _on_decision(self, d: FillDecision, inst: Instrument) -> None:
        if d.liquidity in ("TAKER", "MAKER") and "fill source: quotes" not in d.reasons and \
                self._uses_quotes(inst.symbol):
            d.reasons = tuple(d.reasons) + (("fill source: bars (quotes sparse in window)",) if self._sparse_quote_note
                                            else ("fill source: bars (no quotes in window)",))
        self._book(d, inst, fee_key=d.order.id, notify_manager=True)

    def _book(self, d: FillDecision, inst: Instrument, fee_key: int, notify_manager: bool) -> None:
        if d.ts_ns > self.now_ns:
            raise LookaheadError(f"fill at {ts_utc(d.ts_ns)} after now {self.now}")
        o = d.order
        ts = pd.Timestamp(d.ts_ns, tz="UTC")
        ex_c = self.fees.explicit_costs(inst, d.qty, d.price, ts, d.liquidity, order_id=fee_key)
        costs = CostBreakdown(ex_c.commission, ex_c.exchange_fees, ex_c.regulatory_fees, d.spread_cost, d.impact_cost)
        fill = Fill(o.id, inst, ts, d.qty, d.price, costs, d.fidelity, d.base, d.liquidity, "; ".join(d.reasons))
        pos_before = self.account.position(inst.symbol)
        self.account.apply_fill(fill)
        self._state_ver += 1
        ex = self.om.extras.get(o.id)
        istop = None
        planned = None
        if ex is not None:
            istop = ex.initial_stop
            planned = ex.planned_entry
            if istop is None and ex.parent_id is not None:
                pex = self.om.extras.get(ex.parent_id)
                istop = pex.initial_stop if pex else None
        self.trades.on_fill(fill, pos_before, istop, o.tag, planned)
        self._record_fill(fill, o.tag, d.reasons)
        self._pending_on_fill.append(fill)
        if notify_manager:
            self._after_fill_order(o, d.qty, ts, d.pos, d.ts_ns, d.price)

    def _record_fill(self, fill: Fill, tag: str, reasons: tuple) -> None:
        c = fill.costs
        inst = fill.instrument
        adv = float("nan")
        und_px = float("nan")
        if inst.is_option:
            und_px = self._underlying_price(inst.underlying or "")
        elif self.md is not None:
            st = self.md.daily(inst.symbol)
            if st is not None:
                i = st.idx(int(self.session.value))
                if 0 <= i < st.bars.n:
                    px = st.prior_close[i]
                    if px == px and px > 0:
                        adv = float(st.adv_dollar[i]) / px
        self.fill_rows.append({
            "ts": fill.ts, "order_id": fill.order_id, "symbol": inst.symbol,
            "asset_class": inst.asset_class.value, "multiplier": inst.multiplier,
            "quantity": fill.quantity, "price": fill.price,
            "commission": c.commission, "exchange_fees": c.exchange_fees, "regulatory_fees": c.regulatory_fees,
            "spread_cost": c.spread_cost, "impact_cost": c.impact_cost,
            "fidelity": int(fill.fidelity), "fidelity_reason": "; ".join(r for r in reasons if r),
            "liquidity": fill.liquidity, "note": fill.note, "reference_mid": fill.reference_mid,
            "adv": adv, "underlying_price": und_px,
            "notional": abs(fill.quantity * fill.price * inst.multiplier), "tag": tag,
            "fidelity_name": Fidelity(fill.fidelity).name})
        self.fid.on_fill(fill, tuple(reasons))

    def _after_fill_order(self, o: Order, qty: float, ts: pd.Timestamp, pos: float, ts_ns: int,
                          fill_px: Optional[float] = None) -> None:
        touched, templates, child_updates = self.om.on_fill(o, qty, ts)
        spec = self.om.bracket_of(o.id)
        if spec is None:
            return
        for k, dq in child_updates:
            cid = spec.child_ids.get(k)
            child = self.om.orders.get(cid) if cid is not None else None
            if child is not None and child.status in ACTIVE:
                child.quantity += dq
                continue
            if getattr(templates[k], "skip", False):
                continue
            self._create_child(o, spec, k, dq, ts, ts_ns, pos, fill_px)

    def _create_child(self, o: Order, spec, k: int, dq: float, ts: Optional[pd.Timestamp] = None,
                      ts_ns: Optional[int] = None, pos: Optional[float] = None,
                      fill_px: Optional[float] = None) -> Order:
        ts = self.now if ts is None else ts
        ts_ns = self.now_ns if ts_ns is None else ts_ns
        t = spec.templates[k]
        if t.rel_fn is not None:
            ref = fill_px
            if ref is None or ref != ref:
                p_ = self.account.get(o.instrument.symbol)
                ref = p_.avg_price if p_ is not None else float("nan")
            side = 1 if o.quantity > 0 else -1
            raw = t.rel_fn(float(ref), side)
            lvl = float(raw) if raw is not None else float("nan")
            if not (lvl == lvl and lvl > 0):
                t.rel_fn = None
                t.skip = True
                ev = {"ts": self.now, "order_id": o.id, "symbol": o.instrument.symbol, "action": "CHILD_SKIPPED",
                      "reason": f"fill-relative {t.tag} level is None/NaN: child not created (OCO group kept)",
                      "tag": t.tag}
                self.diag["order_events"].append(ev)
                self._pending_events.append(ev)
                return None
            if t.type == OrderType.STOP:
                t.stop_price = lvl
                pex_ = self.om.extras[o.id]
                if pex_.initial_stop is None or pex_.initial_stop != pex_.initial_stop or pex_.anchor_stop:
                    pex_.initial_stop = lvl
                    pex_.anchor_stop = True
                    tr = self.trades.open.get(o.instrument.symbol)
                    if tr is not None:
                        tr.initial_stop = lvl
            else:
                t.limit_price = lvl
            t.rel_fn = None                      # anchored once, to the first entry fill
        co = Order(o.instrument, dq, t.type, t.limit_price, t.stop_price, t.tif, tag=t.tag,
                   oco_group=spec.oco_group)
        co.strategy_id = o.strategy_id
        co.submitted_at = ts
        co.eligible_at = ts
        tick = valid_tick(t.limit_price or t.stop_price or 1.0, o.instrument)
        if co.limit_price is not None:
            co.limit_price = round_limit_for_side(co.limit_price, tick, dq > 0)
        if co.stop_price is not None:
            co.stop_price = round_stop_for_side(co.stop_price, tick, dq > 0)
        pex = self.om.extras[o.id]
        cex = OrderExtras(outside_rth=t.outside_rth or pex.outside_rth, parent_id=o.id,
                          activate_at_ns=int(t.activate_at.value) if t.activate_at is not None else None,
                          eligible_ns=ts_ns, submitted_ns=ts_ns, in_bar_from=pos, reduce_only=True)
        cex.valid_session_ns = self._validity_session(ts_ns, cex.outside_rth) if t.tif == TimeInForce.DAY else None
        if t.type == OrderType.MOC:
            tgt = t.target_session if t.target_session is not None else self._session_of_ns(ts_ns)
            tgt = tgt if self.cal.is_session(tgt) else self.cal.next_session(tgt)
            earliest = self._auction_session(ts_ns, False)
            cex.target_session_ns = max(int(pd.Timestamp(tgt).value), earliest)
        self.om.add(co, cex, ts)
        self._state_ver += 1
        spec.child_ids[k] = co.id
        return co

    # ================================================================= marking
    def _mark(self, t_ns: int) -> None:
        for sym, p in list(self.account.positions.items()):
            if p.instrument.is_option:
                continue
            b = self.md.bars(sym)
            if b.n == 0 and self._uses_quotes(sym):   # quote-only symbol
                q = self.ctx.last_quote(sym)
                if q is not None and q.ask > q.bid > 0:
                    p.mark = q.mid
                    p.mark_ts = pd.Timestamp(q.ts)
                continue
            i = b.last_index_at(t_ns)
            if i >= 0 and (p.mark_ts is None or b.end[i] >= int(p.mark_ts.value) or p.mark != p.mark):
                p.mark = float(b.c[i])
                p.mark_ts = ts_utc(int(b.end[i]))

    def _mark_options(self) -> None:
        """Same-session option marks only. Per position, the latest observation
        time-stamped in THIS session wins: a fill in the contract, or a snapshot
        from ``portal.option_quotes`` (real quote T0-T2; the data portal falls
        back to a same-day chain-surface mark T2 and then a PIT model value T3).
        ``Position.mark_ts`` / ``mark_fidelity`` record where the mark came from.
        With no same-session observation the previous mark is kept for NAV only
        (its old ``mark_ts`` tells the lifecycle it is stale, so it is never used
        as an early-assignment price) and the run is flagged T2."""
        o_ts, _ = self.session_open_close
        lo_ns = int(o_ts.value) - int(6 * 3600e9)            # session day incl. pre-market
        for p in self.account.option_positions():
            cur_ns = int(pd.Timestamp(p.mark_ts).value) if p.mark_ts is not None else None
            cur_ok = p.mark == p.mark and cur_ns is not None and lo_ns < cur_ns <= self.now_ns
            snaps = self._option_snapshots(p.instrument, lo_ns)
            best = None
            for t in sorted(snaps, reverse=True):
                b, a, f = snaps[t]
                if b == b and a == a and a > 0 and b >= 0 and a >= b:
                    best = (t, 0.5 * (b + a), f)
                    break
            if best is not None and (not cur_ok or best[0] >= cur_ns):
                p.mark = best[1]
                p.mark_ts = ts_utc(best[0])
                p.mark_fidelity = best[2]
                if best[2] >= Fidelity.T2_PROXY:
                    self.fid.note(best[2], "option marks from quote snapshots / surface / model")
            elif not cur_ok:
                self.fid.flag(self.now, Fidelity.T2_PROXY,
                              f"no same-session option mark {p.symbol}: last mark kept for NAV only")

    def _update_excursions(self, lo: int, hi: int) -> None:
        for sym in list(self.trades.open):
            inst = self.instrument(sym)
            if inst.is_option:
                p = self.account.get(sym)
                if p is not None:
                    self.trades.on_bar(sym, p.mark, p.mark, count_bar=True)
                continue
            b, i0, i1 = self.md.window(sym, lo, hi)
            if i1 > i0:
                self.trades.on_bar(sym, float(np.max(b.h[i0:i1])), float(np.min(b.l[i0:i1])))
            else:
                self.trades.on_bar(sym, float("nan"), float("nan"))

    # ================================================================= session rules
    def _pending_net(self, sym: str, include_children: bool = False) -> float:
        """Position plus remaining quantity of pending non-child working orders
        (engine flatten/liquidation orders included)."""
        net = self.account.position(sym)
        for o in self.om.active(sym):
            ex = self.om.extras[o.id]
            if ex.parent_id is not None and not include_children:
                continue
            net += o.remaining
        return net

    def _enforce_session_rules(self) -> None:
        if self.daily:
            return
        if self._no_entry_ns is not None and self.now_ns >= self._no_entry_ns and self.om.active_symbols():
            ne = self._session_rules.get("no_new_entries_after")
            for o in list(self.om.active()):
                ex = self.om.extras[o.id]
                if o.instrument is None or ex.parent_id is not None or ex.reduce_only or ex.is_flatten:
                    continue
                pos = self.account.position(o.instrument.symbol)
                if abs(pos + o.remaining) > abs(pos) + 1e-12:
                    self._cancel_with_event(o, f"resting entry cancelled: no new entries after {ne} ET "
                                               f"(close-adjusted)")
        if self._flat_moc_mode:
            self._ensure_flat(OrderType.MOC)
            return
        if self._flat_step_ns is None or self.now_ns < self._flat_step_ns:
            return
        if not self._flattened_session:
            self._flattened_session = True
            fb = self._session_rules.get("flat_by")
            for o in list(self.om.active()):
                self._cancel_with_event(o, f"flat_by {fb} ET (close-adjusted): working orders cancelled")
        self._ensure_flat(OrderType.MARKET)

    def _ensure_flat(self, otype: OrderType) -> None:
        """Every step until the close: position + pending flatten orders must
        net to zero (a strategy exit racing the flatten can never flip it)."""
        syms = set(s_ for s_, p in self.account.positions.items() if not p.instrument.is_option)
        syms |= {s_ for s_ in self.om.active_symbols() if not s_.startswith("COMBO:")}
        for sym in sorted(syms):
            inst = self.instrument(sym)
            if inst.is_option:
                continue
            pos = self.account.position(sym)
            flat_orders = [o for o in self.om.active(sym) if self.om.extras[o.id].is_flatten]
            net = pos + sum(o.remaining for o in flat_orders)
            if abs(net) <= 1e-9:
                continue
            for o in flat_orders:
                self.om.cancel(o.id, self.now, "flatten re-issued")
            if pos != 0:
                o = Order(inst, -pos, otype, tif=TimeInForce.DAY, tag="flat_by")
                self.submit(o, internal=True)

    # ================================================================= submission
    def _validity_session(self, eligible_ns: int, outside_rth: bool) -> int:
        d = self._session_of_ns(eligible_ns)
        if self.cal.is_session(d):
            o, c = self.cal.open_close(d)
            end = int(c.value) + (4 * 3600 * 10**9 if outside_rth else 0)
            if eligible_ns < end:
                return int(d.value)
        return int(self.cal.next_session(d).value)

    def _auction_session(self, eligible_ns: int, is_open: bool) -> int:
        d = self._session_of_ns(eligible_ns)
        cfg = self.cfg.auction
        for _ in range(10):
            if self.cal.is_session(d):
                o, c = self.cal.open_close(d)
                cut = (o - cfg.moo_cutoff) if is_open else (c - cfg.moc_cutoff)
                if eligible_ns <= int(cut.value):
                    return int(d.value)
            d = self.cal.next_session(d)
        return int(d.value)

    def _exec_rank(self, otype: OrderType, eligible_ns: int, target_session_ns: Optional[int]) -> int:
        """Earliest time an order can execute: auction time for MOO/MOC, else eligible_at."""
        if otype in (OrderType.MOO, OrderType.MOC):
            sess_ns = target_session_ns if target_session_ns is not None else \
                self._auction_session(eligible_ns, otype == OrderType.MOO)
            o_, c_ = self.cal.open_close(pd.Timestamp(sess_ns))
            return int((o_ if otype == OrderType.MOO else c_).value)
        return eligible_ns

    def _projected_position(self, sym: str, order: Order, eligible_ns: int) -> float:
        """Position this order will meet when it executes: current position plus
        pending same-symbol orders that execute no later than it. Pending sells
        always count (conservative); pending buys count only when they are sure
        to execute first (MARKET / MOO / earlier-or-same auction MOC)."""
        mine = self._exec_rank(order.type, eligible_ns, None)
        proj = self.account.position(sym)
        for o in self.om.active(sym):
            ex = self.om.extras[o.id]
            if ex.parent_id is not None or ex.reduce_only:
                continue
            rem = o.remaining
            if rem < 0:
                proj += rem
                continue
            if o.type not in (OrderType.MARKET, OrderType.MOO, OrderType.MOC):
                continue
            if self._exec_rank(o.type, ex.eligible_ns, ex.target_session_ns) <= mine:
                proj += rem
        return proj

    @staticmethod
    def _conservative_price(otype: OrderType, q: float, limit: Optional[float], stop: Optional[float],
                            ref: float) -> float:
        """Pre-trade pricing: the worse of order price and reference price; a
        non-marketable buy limit keeps its limit."""
        if otype == OrderType.LIMIT and q > 0 and limit is not None and ref == ref and limit < ref:
            return float(limit)
        cands = [x for x in (limit, stop, ref) if x is not None and x == x and x > 0]
        return float(max(cands)) if cands else float("nan")

    def submit(self, order: Order, outside_rth: bool = False, initial_stop: Optional[float] = None,
               activate_at: Optional[pd.Timestamp] = None, children: Optional[list[ChildTemplate]] = None,
               internal: bool = False, planned_entry: Optional[float] = None) -> Order:
        now = self.now
        order.strategy_id = self.strategy.id
        order.submitted_at = now
        order.id = -1
        sym = None
        inst = order.instrument
        if order.type == OrderType.COMBO or order.legs:
            if not order.legs:
                return self.om.reject(order, "COMBO order without legs", now)
            legs = []
            for leg in order.legs:
                li = leg.instrument
                if isinstance(li, str):
                    li = self.instrument(li)
                if li.is_option:
                    li, why = self._master_option(li)
                    if why:
                        return self.om.reject(order, why, now)
                legs.append(leg if li is leg.instrument else replace(leg, instrument=li))
            order.legs = tuple(legs)
            order.type = OrderType.COMBO
            q = order.quantity
            if q != int(q) or q == 0:
                return self.om.reject(order, "combo quantity must be a non-zero integer", now)
            quotes = False
        else:
            if inst is None:
                return self.om.reject(order, "order without instrument", now)
            if isinstance(inst, str):
                inst = self.instrument(inst)
                order.instrument = inst
            if inst.is_option:
                inst, why = self._master_option(inst)
                if why:
                    return self.om.reject(order, why, now)
                order.instrument = inst
            sym = inst.symbol
            quotes = (not inst.is_option) and self._uses_quotes(sym) and not self._has_bars(sym)
        lat = self.cfg.latency_quotes if (quotes or (inst is not None and inst.is_option) or order.type == OrderType.COMBO) \
            else self.cfg.latency_bars
        order.eligible_at = now + lat
        q = float(order.quantity)
        if not math.isfinite(q) or q == 0:
            return self.om.reject(order, "quantity must be finite and non-zero", now)
        if inst is not None and order.type != OrderType.COMBO:
            if inst.is_option:
                if q != int(q):
                    return self.om.reject(order, "options trade in integer contracts", now)
                if order.type not in (OrderType.MARKET, OrderType.LIMIT):
                    return self.om.reject(order, f"{order.type.value} not supported for options (snapshot fills)", now)
                if inst.multiplier <= 0 or inst.deliverable_shares <= 0:
                    return self.om.reject(order, "invalid option multiplier/deliverable", now)
            elif not self.cfg.bar_fill.allow_fractional and q != int(q):
                q2 = float(math.trunc(q))
                if q2 == 0:
                    return self.om.reject(order, "quantity rounds to zero whole shares", now)
                self.diag["warnings"].append(f"order qty {q} on {sym} truncated to {q2} (no fractional shares)")
                order.quantity = q = q2
        # price fields
        t = order.type
        if t in (OrderType.LIMIT, OrderType.STOP_LIMIT) and order.limit_price is None:
            return self.om.reject(order, f"{t.value} requires limit_price", now)
        if t in (OrderType.STOP, OrderType.STOP_LIMIT) and order.stop_price is None:
            return self.om.reject(order, f"{t.value} requires stop_price", now)
        if inst is not None and t != OrderType.COMBO:
            ref = order.limit_price or order.stop_price or 1.0
            tick = valid_tick(ref, inst)
            if order.limit_price is not None:
                lp = round_limit_for_side(order.limit_price, tick, q > 0)
                if lp <= 0:
                    return self.om.reject(order, "limit price <= 0 after tick rounding", now)
                order.limit_price = lp
            if order.stop_price is not None:
                order.stop_price = round_stop_for_side(order.stop_price, tick, q > 0)
        # session rules (intraday)
        cur = self.account.position(sym) if sym else 0.0
        net = self._pending_net(sym) if sym else 0.0
        increases = sym is not None and abs(net + q) > abs(net) + 1e-12
        if not internal and not self.daily and sym is not None:
            if self._flattened_session or (self._flat_moc_mode and self._session_rules.get("flat_by")):
                if not abs(net + q) < abs(net) - 1e-12:
                    return self.om.reject(order, f"after flat_by {self._session_rules.get('flat_by')} ET: only orders "
                                                 f"reducing exposure net of pending orders are accepted", now)
            ne = self._session_rules.get("no_new_entries_after")
            if increases and ne and self._no_entry_ns is not None and (
                    self.now_ns > self._no_entry_ns or
                    (self.now_ns == self._no_entry_ns and order.type != OrderType.MARKET)):
                return self.om.reject(order, f"no new entries after {ne} ET (close-adjusted)", now)
        # short sales: locate + account type
        if inst is not None and not inst.is_option and t != OrderType.COMBO and q < 0:
            projected = self._projected_position(sym, order, int((now + lat).value))
            if projected + q < -1e-9:
                order.is_short_sale = True                 # only the part below zero is short; re-decided at fill
                if self.account.account_type == AccountType.CASH:
                    return self.om.reject(order, "cash account: short selling not permitted", now)
                if self.cfg.enforce_locates:
                    ok, fee, fid = self._borrow(sym)
                    if not ok:
                        return self.om.reject(order, "no locate / not shortable", now)
                    if fid >= Fidelity.T2_PROXY:
                        self.fid.note(Fidelity.T2_PROXY, f"borrow default {fee:.2%} p.a. (no borrow data)")
        # buying power (conservative price; no credit for pending exposure-reducing orders)
        if self.cfg.check_margin_on_submit and not internal and t != OrderType.COMBO and inst is not None \
                and increases:
            px = self._conservative_price(order.type, q, order.limit_price, order.stop_price, self.last_price(sym))
            pending = []
            for o2 in self.om.active():
                if o2.instrument is None or self.om.extras[o2.id].reduce_only:
                    continue
                p2 = self._conservative_price(o2.type, o2.remaining, o2.limit_price, o2.stop_price,
                                              self.last_price(o2.instrument.symbol))
                if p2 == p2:
                    pending.append((o2.instrument, o2.remaining, p2))
            try:
                est = self.fees.estimate(inst, q, px, now).explicit if px == px else 0.0
            except Exception:
                est = 0.0
            ok, why = self.margin.check_order(self.account, inst, q, px, self._underlying_price, pending,
                                              est_cost=est, as_of=now)
            if not ok:
                return self.om.reject(order, why, now)
        # extras
        ex = OrderExtras(outside_rth=outside_rth, initial_stop=initial_stop,
                         activate_at_ns=int(pd.Timestamp(activate_at).value) if activate_at is not None else None,
                         submitted_ns=int(now.value), eligible_ns=int(order.eligible_at.value), is_flatten=internal,
                         reduce_only=internal, planned_entry=planned_entry)
        ex.quote_eligible_ns = int((now + self.cfg.latency_quotes).value)
        if t in (OrderType.MOO, OrderType.MOC):
            ex.target_session_ns = self._auction_session(ex.eligible_ns, t == OrderType.MOO)
        if order.tif == TimeInForce.DAY:
            ex.valid_session_ns = self._validity_session(ex.eligible_ns, outside_rth)
        if outside_rth and not any(s in self.data_sessions for s in ("PRE", "POST")):
            self.diag["warnings"].append(f"order {sym} allows outside RTH but no PRE/POST data is loaded")
        self.om.add(order, ex, now)
        self._state_ver += 1
        if children:
            self.om.register_bracket(order, children, f"bracket-{order.id}")
        return order

    def strategy_cancel(self, order_id: int) -> bool:
        ex = self.om.extras.get(order_id)
        if ex is not None and ex.is_flatten:
            self.diag["warnings"].append(f"{self.now}: strategy tried to cancel engine-internal order {order_id} "
                                         f"(flat_by / margin liquidation): ignored")
            return False
        return self.om.cancel(order_id, self.now, "cancelled by strategy")

    def strategy_cancel_all(self, symbol: Optional[str] = None) -> int:
        n = 0
        for oid in list(self.om.active_ids(symbol)):
            if self.om.extras[oid].is_flatten:
                continue
            n += self.om.cancel(oid, self.now, "cancel_all by strategy")
        return n

    def amend_bracket(self, parent_or_group, stop=None, target=None, time_exit=None) -> list[Order]:
        if isinstance(parent_or_group, Order):
            pid = parent_or_group.id
        elif isinstance(parent_or_group, str):
            pid = int(parent_or_group.rsplit("-", 1)[-1])
        else:
            pid = int(parent_or_group)
        spec = self.om.bracket_of(pid)
        if spec is None:
            raise ValueError(f"no bracket for {parent_or_group!r}")
        parent = self.om.orders[pid]
        changes = {"stop": (OrderType.STOP, stop), "target": (OrderType.LIMIT, target),
                   "time_exit": (OrderType.MARKET, time_exit)}
        out = []
        for tag, (otype, val) in changes.items():
            if val is None:
                continue
            k = next((i for i, t in enumerate(spec.templates) if t.tag == tag), None)
            if k is None:
                spec.templates.append(ChildTemplate(otype, tag=tag, tif=TimeInForce.GTC))
                k = len(spec.templates) - 1
            tpl = spec.templates[k]
            if tag == "stop":
                tpl.stop_price = float(val)
            elif tag == "target":
                tpl.limit_price = float(val)
            else:
                tpl.activate_at = pd.Timestamp(val)
            if tag == "stop":
                ex = self.om.extras[pid]
                ex.initial_stop = ex.initial_stop if ex.initial_stop is not None else float(val)
            cid = spec.child_ids.get(k)
            child = self.om.orders.get(cid) if cid is not None else None
            if child is not None and child.status in ACTIVE:
                tick = valid_tick(float(val) if tag != "time_exit" else 1.0, child.instrument)
                if tag == "stop":
                    child.stop_price = round_stop_for_side(float(val), tick, child.quantity > 0)
                elif tag == "target":
                    child.limit_price = round_limit_for_side(float(val), tick, child.quantity > 0)
                else:
                    self.om.extras[cid].activate_at_ns = int(pd.Timestamp(val).value)
                cex = self.om.extras[cid]
                cex.eligible_ns = max(cex.eligible_ns, self.now_ns)      # replace takes effect from now
                self._state_ver += 1
                out.append(child)
                continue
            # leg not live yet: create it now if the entry already (partly) filled
            if abs(parent.filled_qty) > 1e-12:
                siblings = [self.om.orders[c] for c in spec.child_ids.values()
                            if c in self.om.orders and self.om.orders[c].status in ACTIVE]
                q = siblings[0].remaining if siblings else -parent.filled_qty
                c_ = self._create_child(parent, spec, k, q)
                if c_ is not None:
                    out.append(c_)
        return out

    def _borrow(self, sym: str) -> tuple[bool, float, Fidelity]:
        key = (sym, int(self.session.value))
        r = self._borrow_cache.get(key)
        if r is None:
            try:
                ok, fee, fid = self.portal.borrow(sym, self.now)
            except Exception:
                ok, fee, fid = False, float("nan"), Fidelity.NA
            bc = self.cfg.borrow
            if fid == Fidelity.NA or not (fee == fee):
                if bc.require_borrow_data:
                    ok, fee, fid = False, float("nan"), Fidelity.NA
                else:
                    ok, fee, fid = bc.default_shortable, bc.default_fee, Fidelity.T2_PROXY
            r = (bool(ok), float(fee), Fidelity(fid))
            self._borrow_cache[key] = r
        return r

    def _underlying_price(self, sym: str) -> float:
        if not sym:
            return float("nan")
        try:
            return self.last_price(sym)
        except LookaheadError:
            return float("nan")

    # ================================================================= options
    def _apply_option_events(self, evs: list[OptionEvent]) -> None:
        for ev in evs:
            inst = ev.option
            pos_before = self.account.position(inst.symbol)
            if pos_before == 0:
                continue
            if ev.kind == "EXPIRE_WORTHLESS" and not (ev.settlement_value == ev.settlement_value):
                # plug-in had no settlement price at all: never let a liability vanish at 0
                p_ = self.account.get(inst.symbol)
                mk = p_.mark if p_ is not None and p_.mark == p_.mark else (p_.avg_price if p_ else 0.0)
                ev = replace(ev, kind="SETTLE_AT_MARK", option_price=float(mk), fidelity=Fidelity.T2_PROXY,
                             note="no underlying price at expiry: closed at last option mark")
            liq = {"EXERCISE": "EXERCISE", "ASSIGNMENT": "ASSIGNMENT"}.get(ev.kind, "SETTLEMENT")
            reason = {"EXERCISE": Reason.EXERCISE, "ASSIGNMENT": Reason.ASSIGNMENT}.get(ev.kind, Reason.CASH_SETTLEMENT)
            f = Fill(-1, inst, self.now, ev.option_qty, ev.option_price, CostBreakdown(), ev.fidelity,
                     liquidity=liq, note=f"{ev.kind}: {ev.note}")
            self.account.apply_fill(f, cash_reason=reason)
            self._state_ver += 1
            self.trades.on_fill(f, pos_before, None, ev.kind.lower())
            self._record_fill(f, ev.kind.lower(), (ev.note,) if ev.note else ())
            self._pending_on_fill.append(f)          # Strategy.on_fill, same step
            if ev.underlying and ev.underlying_qty:
                uinst = self.instrument(ev.underlying)
                c = self.fees.explicit_costs(uinst, ev.underlying_qty, ev.underlying_price, self.now, liq)
                uf = Fill(-1, uinst, self.now, ev.underlying_qty, ev.underlying_price, c, ev.fidelity,
                          liquidity=liq, note=f"{ev.kind} delivery of {inst.symbol}")
                ub = self.account.position(uinst.symbol)
                self.account.apply_fill(uf, cash_reason=reason)
                p = self.account.get(uinst.symbol)
                if p is not None and ev.settlement_value == ev.settlement_value:
                    p.mark = ev.settlement_value
                    p.mark_ts = self.now
                self.trades.on_fill(uf, ub, None, ev.kind.lower())
                self._record_fill(uf, ev.kind.lower(), ("option delivery",))
                self._pending_on_fill.append(uf)
            self.diag["option_events"].append({"ts": self.now, "kind": ev.kind, "option": inst.symbol,
                                               "qty": ev.option_qty, "settlement_value": ev.settlement_value,
                                               "underlying_qty": ev.underlying_qty, "note": ev.note})
            if ev.fidelity >= Fidelity.T2_PROXY:
                self.fid.flag(self.now, ev.fidelity, f"option {ev.kind.lower()}: {ev.note}")
            self.om.cancel_all(self.now, inst.symbol, "option expired/settled")

    # ================================================================= session end
    def _session_end(self, session: pd.Timestamp, next_session: pd.Timestamp) -> None:
        self.strategy.on_session_end(self.ctx)
        self._flush_on_fill()
        ts = self.now
        # option expiry (PM / physical)
        opts = self.account.option_positions()
        if opts:
            self._mark_options()          # same-session marks BEFORE expiry / early-assignment tests
            evs = self.lifecycle.at_session_end(session, self.account.option_positions(), _Prices(self))
            self._apply_option_events(evs)
        self._flush_on_fill()
        # financing for calendar days until next session
        days = int((pd.Timestamp(next_session) - pd.Timestamp(session)).days)
        try:
            rf = float(self.portal.risk_free_rate(ts, self.cfg.rf_tenor))
        except Exception:
            rf = float("nan")
        if rf != rf:
            rf = 0.0
            if self.account.cash != 0:
                self.fid.note(Fidelity.T2_PROXY, "risk-free rate unavailable: 0 used")
        fin = accrue_financing(self.account, ts, days, rf, lambda s: self._borrow(s)[1], self.cfg.financing)
        f = self.diag["financing"]
        f["credit"] += fin.credit; f["debit"] += fin.debit; f["borrow"] += fin.borrow; f["rebate"] += fin.rebate
        for s_, amt in fin.by_symbol.items():
            self.trades.attribute(s_, amt)
        _, c_ = self.session_open_close
        self.rf_daily[c_] = rf * days / self.cfg.financing.day_count
        self._rf_days_total += days
        # DAY order expiry
        self.om.expire_session(int(session.value), ts)
        # margin
        if self.account.account_type == AccountType.REG_T_MARGIN and self.cfg.margin_calls:
            snap = self.margin.snapshot(self.account, self._underlying_price, as_of=ts)
            if snap.is_call:
                plan = self.margin.liquidation_plan(self.account, snap)
                self.diag["margin_calls"].append({"ts": ts, "equity": snap.equity, "maintenance": snap.maintenance,
                                                  "orders": [(p.symbol, dq) for p, dq in plan]})
                # cancel only exposure-increasing orders; exits are kept (and shrink as
                # the liquidation fills, reduce-only)
                for o in list(self.om.active()):
                    ex = self.om.extras[o.id]
                    if o.instrument is None:
                        self._cancel_with_event(o, "margin call: combo order cancelled")
                        continue
                    if ex.reduce_only:
                        continue
                    pos = self.account.position(o.instrument.symbol)
                    if abs(pos + o.remaining) > abs(pos) + 1e-12:
                        self._cancel_with_event(o, "margin call: exposure-increasing order cancelled")
                for p, dq in plan:
                    o = Order(p.instrument, dq, OrderType.MARKET, tif=TimeInForce.DAY, tag="margin_liquidation")
                    self.submit(o, internal=True)
        elif self.account.account_type == AccountType.CASH and self.account.cash < -1e-6:
            self.diag["cash_account_negative_cash_days"] += 1
        self._check_stale_prices(session)
        if self._session_rules.get("flat_by") and not self.daily:
            left = [s_ for s_, p in self.account.positions.items() if not p.instrument.is_option]
            if left:
                self.diag["warnings"].append(f"{session.date()}: positions {left} carried overnight despite "
                                             f"flat_by (liquidity/halt)")
        nav = self.account.nav
        _, c = self.session_open_close
        self.equity[c] = nav
        self.pos_rows.extend(self.account.snapshot(c))
        self.prior_nav = nav
        if self.cfg.debug:
            self._check_invariants()

    def _check_stale_prices(self, session: pd.Timestamp) -> None:
        n = self.cfg.stale_price_sessions
        if not n:
            return
        cur_i = self.cal.session_index(session)
        for sym, p in self.account.positions.items():
            if p.instrument.is_option or sym in self._stale_flagged:
                continue
            _, end_ns = self._last_bar_close(sym)
            if end_ns < 0:
                k = n
            else:
                d = self._session_of_ns(end_ns - 1)
                try:
                    k = cur_i - self.cal.session_index(d)
                except KeyError:
                    k = n
            if k >= n:
                self._stale_flagged.add(sym)
                msg = f"stale price: no bar for {sym} in {k} sessions (position {p.qty:g} carried at {p.mark:.4g})"
                self.diag["warnings"].append(msg)
                self.fid.flag(self.now, Fidelity.T2_PROXY, msg)

    # ================================================================= invariants
    def _check_invariants(self) -> None:
        self.diag["invariant_checks"] += 1
        errs = self.account.check_invariants()
        for oid in self.om.active_ids():
            o = self.om.orders[oid]
            if abs(o.filled_qty) > abs(o.quantity) + 1e-9:
                errs.append(f"order {oid} overfilled")
        if self.fill_rows and self.fill_rows[-1]["ts"] > self.now:
            errs.append("fill timestamp after now")
        if errs:
            raise AssertionError(f"accounting invariant violated at {self.now}: {errs}")

    # ================================================================= results
    def _result(self, status: RunStatus, params: dict, reason: str, t0: float, reqfid: dict) -> BacktestResult:
        eq = pd.Series(self.equity, dtype=float)
        eq.index = pd.DatetimeIndex(eq.index, name="ts")
        if len(eq):
            prev = eq.shift(1)
            prev.iloc[0] = self.cfg.initial_cash
            rets = eq / prev - 1.0
        else:
            rets = pd.Series(dtype=float)
        marks = {s: p.mark for s, p in self.account.positions.items()}
        diag = self.diag
        diag["rejected_orders"] = list(self.om.rejected)
        diag["orders"] = self.om.to_frame()
        diag["logs"] = self.logs
        diag["lookahead_guard_violations"] = self.md.guard_violations if self.md else 0
        diag["runtime_sec"] = time.perf_counter() - t0
        diag["fees_total"] = self.account.fees_total
        diag["ledger_by_reason"] = self.account.ledger.by_reason()
        if self.cfg.record_intraday_equity:
            diag["intraday_equity"] = pd.Series({t: v for t, v in self.intraday_equity}, dtype=float)
        rfd = pd.Series(self.rf_daily, dtype=float)
        rng = getattr(self, "_run_range", (None, None))
        grid = getattr(self.strategy, "grid", {}) or {}
        n_trials = 1
        for v in grid.values():
            n_trials *= max(len(v), 1)
        meta = {
            "initial_capital": self.cfg.initial_cash,
            "rf_daily": rfd,
            "risk_free_rate": (float(rfd.sum() * self.cfg.financing.day_count / self._rf_days_total)
                               if len(rfd) and self._rf_days_total else float("nan")),
            "start": rng[0], "end": rng[1], "bar_freq": self.strategy.bar_freq,
            "universe": list(self.strategy.universe) or self.strategy.universe_name,
            "config": _config_summary(self.cfg),
            "data_sources": sorted({str(getattr(self.portal, "name", type(self.portal).__name__))}),
            "survivorship_bias": ("static universe list: survivorship bias possible"
                                  if self.strategy.universe and not self.strategy.universe_name else
                                  "PIT universe via portal.universe" if self.strategy.universe_name else "n/a"),
            "n_trials": n_trials,
            "base_freq": self.base_freq, "declared_bar_freq": self.strategy.bar_freq,
            "sizing": self.strategy.sizing, "asset": self.strategy.asset,
            "deviations": list(self.strategy.deviations), "fees": getattr(self.fees, "name", str(self.fees)),
            "account_type": self.cfg.account_type.value, "initial_cash": self.cfg.initial_cash,
            "data_sessions": self.data_sessions, "timeline_sessions": self.timeline_sessions,
            "session_rules": self._session_rules,
            "config_overrides": dict(self.config_overrides),
        }
        cm = self._coverage_meta()
        if cm:
            meta["universe_coverage"] = cm
            if cm["coverage_min"] == cm["coverage_min"] and cm["coverage_min"] < 1.0:
                meta["survivorship_bias"] += (f"; members without verified data excluded (coverage min "
                                              f"{cm['coverage_min']:.1%}, {len(cm['dropped'])} members dropped)")
        return BacktestResult(
            strategy_id=self.strategy.id, params=params, status=status, equity=eq, returns=rets,
            fills=fills_frame(self.fill_rows), trades=self.trades.frame(marks),
            positions=pd.DataFrame(self.pos_rows, columns=["ts", "symbol", "quantity", "mark", "market_value",
                                                           "avg_price", "unrealized", "multiplier", "asset_class"]),
            ledger=self.account.ledger.to_frame(), fidelity=self.fid.scorecard(reqfid), diagnostics=diag,
            not_testable_reason=reason if status != RunStatus.OK else "", meta=meta)


def _config_summary(cfg: BacktestConfig) -> dict:
    return {
        "initial_cash": cfg.initial_cash, "account_type": cfg.account_type.value,
        "fees": cfg.fees if isinstance(cfg.fees, str) else getattr(cfg.fees, "name", type(cfg.fees).__name__),
        "latency_bars": str(cfg.latency_bars), "latency_quotes": str(cfg.latency_quotes),
        "participation_cap": cfg.bar_fill.participation_cap, "ssr_mode": cfg.bar_fill.ssr_mode,
        "impact_k": cfg.impact.k, "spread_method": cfg.spread.method,
        "credit_spread": cfg.financing.credit_spread, "debit_spread": cfg.financing.debit_spread,
        "borrow_default_fee": cfg.borrow.default_fee, "resolve_intrabar_freq": cfg.resolve_intrabar_freq,
        "use_quotes": cfg.use_quotes,
    }


class EngineExecutionState:
    """core.protocols.ExecutionState view of a running backtest (for
    stand-alone ``FillModel.try_fill`` calls and external cost models)."""

    def __init__(self, bt: "Backtester"):
        self.bt = bt

    def position(self, symbol: str) -> float:
        return self.bt.account.position(symbol)

    def _stats(self, symbol: str):
        st = self.bt.md.daily(symbol) if self.bt.md is not None else None
        if st is None:
            return None, -1
        return st, st.idx(int(self.bt.session.value))

    def is_ssr_active(self, symbol: str, ts: pd.Timestamp) -> bool:
        mkt = self.bt._symbol_market(symbol, self.bt.instrument(symbol))
        t = min(int(pd.Timestamp(ts).value), self.bt.now_ns)
        return bool(mkt.ssr_active(BarView(t, t, math.nan, math.nan, math.inf, math.nan, math.nan)))

    def adv(self, symbol: str, as_of: pd.Timestamp) -> float:
        st, i = self._stats(symbol)
        if st is None or not (0 <= i < st.bars.n):
            return float("nan")
        px = st.prior_close[i]
        return float(st.adv_dollar[i] / px) if px == px and px > 0 else float("nan")

    def daily_vol(self, symbol: str, as_of: pd.Timestamp) -> float:
        st, i = self._stats(symbol)
        return float(st.sigma[i]) if st is not None and 0 <= i < st.bars.n else float("nan")


class _Prices:
    """UnderlyingPrices adapter for the option lifecycle hook."""

    def __init__(self, bt: Backtester):
        self.bt = bt

    def close(self, symbol: str, session) -> float:
        try:
            v = self.bt.md.session_close_price(symbol, session)
        except LookaheadError:
            v = float("nan")
        if v != v:
            v = self.last_close(symbol)
            if v == v:
                self.bt.fid.flag(self.bt.now, Fidelity.T2_PROXY,
                                 f"settlement close of {symbol} missing: last known close used")
        return v

    def open(self, symbol: str, session) -> float:
        try:
            return self.bt.md.session_open_price(symbol, session)
        except LookaheadError:
            return float("nan")

    def last_close(self, symbol: str) -> float:
        try:
            return self.bt.last_price(symbol)
        except LookaheadError:
            return float("nan")


def run_backtest(strategy: Strategy, portal, start, end, config: Optional[BacktestConfig] = None,
                 calendar=None, raise_errors: bool = False) -> BacktestResult:
    return Backtester(strategy, portal, config, calendar).run(start, end, raise_errors=raise_errors)
