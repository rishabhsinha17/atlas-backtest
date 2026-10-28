"""The backtest event loop (docs/ARCHITECTURE.md §4, docs/ENGINE.md).

Master timeline = bar END times at the strategy's base frequency. At each
step t (end of bar [t-Δ, t)):

1. session start (first step of a session): dividends due, corporate
   actions with ex_date == session, AM-settled option settlements,
   ``strategy.on_session_start`` (ctx.now = session start);
2. fills: every working order is offered the data in (prev_t, t] whose bars
   START at/after the order's ``eligible_at`` (never a price from before the
   order existed);
3. mark-to-market at t (last completed bar close);
4. queued ``on_fill`` callbacks, then ``strategy.on_bar`` with ctx.now = t;
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
    LookaheadError, Order, OrderStatus, OrderType, RunStatus, TimeInForce,
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

    # ================================================================= state
    def _reset_state(self) -> None:
        cfg = self.cfg
        self.account = Account(cfg.initial_cash, cfg.account_type)
        self.om = OrderManager()
        self.fees = copy.deepcopy(cfg.make_fees())
        self.spread_model = SpreadModel(cfg.spread)
        self.bar_model = BarFillModel(cfg.bar_fill, self.spread_model, cfg.impact)
        self.quote_model = QuoteFillModel(cfg.impact, cfg.bar_fill.ssr_mode, cfg.bar_fill.allow_fractional)
        self.auction_model = AuctionFillModel(cfg.auction, cfg.impact, cfg.bar_fill.allow_fractional)
        self.combo_model = ComboFillModel(cfg.combo)
        self.margin = RegTMarginModel(option_model=cfg.option_margin)
        self.ca = CorporateActionProcessor(cfg.corporate_actions)
        self.lifecycle = cfg.lifecycle()
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
        }
        self._instruments: dict[str, Instrument] = {}
        self._pending_on_fill: list[Fill] = []
        self._quotes_mode: dict[str, bool] = {}
        self._halts: dict[str, list[tuple[int, int]]] = {}
        self._halts_loaded: set[str] = set()
        self._borrow_cache: dict[tuple[str, int], tuple[bool, float, Fidelity]] = {}
        self._flattened_session = False
        self._session_rules = self.strategy.session_rules() if not self.daily else {}
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

    def _note_data_access(self, kind, symbol, fidelity) -> None:
        k = f"{kind}"
        self.diag["data_access"][k] = self.diag["data_access"].get(k, 0) + 1

    def flag_fidelity(self, tier: Fidelity, reason: str) -> None:
        self.fid.flag(self.now, tier, reason)

    def last_price(self, symbol: str) -> float:
        p = self.account.get(symbol)
        if p is not None and p.mark == p.mark and (p.mark_ts is None or p.mark_ts <= self.now):
            if not p.instrument.is_option:
                px = self.md.last_close(symbol, self.now_ns) if self.md else float("nan")
                return px if px == px else p.mark
            return p.mark
        inst = self.instrument(symbol)
        if inst.is_option:
            q = self.portal.option_quote(inst, self.now)
            return q.mid if q is not None else float("nan")
        px = self.md.last_close(symbol, self.now_ns)
        if px != px:
            st = self.md.daily(symbol)
            if st is not None:
                i = int(np.searchsorted(st.bars.end, self.now_ns, side="right")) - 1
                if i >= 0:
                    px = float(st.bars.c[i])
        return px

    def _uses_quotes(self, symbol: str) -> bool:
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
    def _check_requirements(self) -> tuple[bool, str, dict]:
        fidmap: dict[str, Optional[int]] = {}
        syms_default = list(self.strategy.universe)
        for r in self.strategy.requires:
            syms = list(r.symbols) if r.symbols else syms_default
            for s in syms:
                key = f"{r.kind}:{s}:{r.freq or ''}"
                try:
                    has = self.portal.has(s, r.kind, r.freq)
                    f = self.portal.fidelity_of(s, r.kind, r.freq) if has else Fidelity.NA
                except Exception as e:   # pragma: no cover - portal bug
                    has, f = False, Fidelity.NA
                fidmap[key] = int(f)
                if r.critical and (not has or f == Fidelity.NA or f > r.min_fidelity):
                    return False, f"required {r.kind} {r.freq or ''} for {s} unavailable (fidelity {Fidelity(f).name})", fidmap
        return True, "", fidmap

    # ================================================================= run
    def run(self, start, end, raise_errors: bool = False) -> BacktestResult:
        t0 = time.perf_counter()
        self._reset_state()
        sessions = self.cal.sessions(pd.Timestamp(start), pd.Timestamp(end))
        params = dict(self.strategy.params)
        if len(sessions) == 0:
            return self._result(RunStatus.FAILED, params, "no sessions in range", t0, {})
        ok, why, reqfid = self._check_requirements()
        if not ok:
            return self._result(RunStatus.NOT_TESTABLE, params, why, t0, reqfid)
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
                self._session_end(s, sessions[i + 1] if i + 1 < len(sessions) else self.cal.next_session(s))
        except LookaheadError:
            raise
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
        fb = self._session_rules.get("flat_by")
        ne = self._session_rules.get("no_new_entries_after")
        self._flat_by_ns = int(et(session, fb).value) if fb else None
        self._no_entry_ns = int(et(session, ne).value) if ne else None
        ts = self.now
        # 1. dividends due
        for a in self.ca.pay_due(self.account, session, ts):
            self.diag["corporate_actions"].append({"ts": ts, "symbol": a.symbol, "type": f"PAY_{a.kind}",
                                                   "amount": a.amount})
        # 2. corporate actions effective today
        syms = sorted(set(self.account.positions) | {s for s in self.om.active_symbols() if not s.startswith("COMBO:")})
        stock_syms = [s for s in syms if not self.instrument(s).is_option]
        if stock_syms:
            try:
                acts = self.portal.corporate_actions(stock_syms, session - pd.Timedelta(days=1),
                                                     session + pd.Timedelta(days=1), as_of=c)
            except Exception as e:
                acts = []
                self.diag["warnings"].append(f"corporate_actions query failed: {e}")
            todays = sorted([a for a in acts if as_session_date(a.ex_date) == session],
                            key=lambda a: (a.symbol, a.type.value))
            for a in todays:
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
        self.strategy.on_session_start(self.ctx)
        self._flush_on_fill()

    def _apply_action(self, a: CorporateAction, session, ts) -> None:
        pos_before = self.account.position(a.symbol)
        res = self.ca.apply(a, self.account, ts, session, instrument_lookup=self.instrument,
                            price_lookup=lambda s: self.last_price(s))
        for s in res.cancel_orders_for:
            n = self.om.cancel_all(ts, s, f"cancelled on corporate action {a.type.value}")
            if n:
                self.diag["warnings"].append(f"{n} open order(s) on {s} cancelled for {a.type.value}")
        if res.renamed_to:
            self._instruments.pop(res.renamed_to, None)
        if a.type.value in ("CASH_DIVIDEND", "SPECIAL_DIVIDEND"):
            for acc in self.account.accruals:
                if acc.symbol == a.symbol and acc.ex_date == session:
                    self.trades.attribute(a.symbol, acc.amount)
        self.diag["corporate_actions"].append({"ts": ts, "symbol": a.symbol, "type": a.type.value,
                                               "applied": res.applied, "note": res.note,
                                               "fidelity": Fidelity(res.fidelity).name})
        if res.fidelity >= Fidelity.T2_PROXY:
            self.fid.flag(ts, res.fidelity, f"corporate action {a.type.value}: {res.note}")
        if a.type.value == "DELIST" and res.closed_qty:
            inst = self.instrument(a.symbol)
            f = Fill(-1, inst, ts, res.closed_qty, res.close_price, CostBreakdown(), res.fidelity,
                     liquidity="CORP_ACTION", note=res.note)
            self.trades.on_fill(f, pos_before, None, "delist")
            self._record_fill(f, "delist", ("delisting cash-out",))

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
        self._mark(t_ns)
        self._flush_on_fill()
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
                self._fill_snapshot_orders(orders, hi)
                continue
            inst = self.instrument(sym)
            auction = [o for o in orders if o.type in (OrderType.MOO, OrderType.MOC)]
            if auction:
                self._fill_auction(sym, inst, auction, lo, hi)
            rest = [o for o in self.om.active(sym) if o.type not in (OrderType.MOO, OrderType.MOC)]
            if not rest:
                continue
            for o in rest:
                if o.quantity < 0 and not inst.is_option:
                    o.is_short_sale = self.account.position(sym) + o.remaining < -1e-9
            if self._uses_quotes(sym):
                self._fill_quotes(sym, inst, lo, hi)
            else:
                self._fill_bars(sym, inst, lo, hi)
            # IOC: cancel remainder after first evaluation
            for o in self.om.active(sym):
                ex = self.om.extras[o.id]
                if o.tif == TimeInForce.IOC and ex.evaluated:
                    self.om.cancel(o.id, self.now, "IOC remainder cancelled")

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
                            self.daily, self.cfg.daily_open_spread_mult)

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

    def _fill_quotes(self, sym: str, inst: Instrument, lo: int, hi: int) -> None:
        df = self.portal.quotes(sym, ts_utc(lo + 1), ts_utc(hi + 1), as_of=self.now)
        if df is None or len(df) == 0:
            return
        from .timeutil import as_ns_array
        tsa = as_ns_array(df.index)
        bid = df["bid"].to_numpy(float)
        ask = df["ask"].to_numpy(float)
        bs = df["bid_size"].to_numpy(float) if "bid_size" in df else np.full(len(df), np.nan)
        as_ = df["ask_size"].to_numpy(float) if "ask_size" in df else np.full(len(df), np.nan)
        qs = [QuoteView(int(tsa[i]), bid[i], ask[i], bs[i], as_[i]) for i in range(len(df))
              if lo < tsa[i] <= hi and not any(hs <= tsa[i] < he for hs, he in self._halt_intervals(sym))]
        mkt = self._symbol_market(sym, inst)
        stats = self.quote_model.simulate(mkt, qs, self._orders_getter(sym),
                                          lambda d: self._on_decision(d, inst),
                                          lambda o, r: self._reject_working(o, r))
        self._add_stats(stats)

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
            mkt = mkt or self._symbol_market(sym, inst)
            daily_bar = int(b.sess[k]) == 3
            px = float(b.o[k] if is_open else b.c[k])
            ts_ns = int(b.start[k]) if is_open else int(b.end[k])
            dec = self.auction_model.fill(od, px, float(b.v[k]), ts_ns, mkt, is_open, daily_bar)
            if dec is not None:
                self._on_decision(dec, inst)
            if od.status in ACTIVE:
                self.om.cancel(od.id, self.now, "auction remainder cancelled", OrderStatus.EXPIRED)

    def _fill_snapshot_orders(self, orders: list[Order], hi: int) -> None:
        for od in orders:
            ex = self.om.extras[od.id]
            if ex.eligible_ns >= hi:
                continue
            legs = od.legs if od.type == OrderType.COMBO else (Leg(od.instrument, 1),)
            lq: list[tuple[Leg, Optional[LegQuote]]] = []
            opt_ts = None
            for leg in legs:
                if leg.instrument.is_option:
                    q = self.portal.option_quote(leg.instrument, self.now)
                    if q is None:
                        lq.append((leg, None))
                        continue
                    t = int(pd.Timestamp(q.ts).value)
                    opt_ts = t if opt_ts is None else opt_ts
                    lq.append((leg, LegQuote(leg.instrument, t, q.bid, q.ask, q.fidelity)))
                else:
                    lq.append((leg, None))   # filled below once the option snapshot time is known
            # stock legs priced from the last completed bar at/before the option snapshot
            for k, (leg, q) in enumerate(lq):
                if q is None and not leg.instrument.is_option:
                    t = opt_ts if opt_ts is not None else self.now_ns
                    try:
                        b = self.md.bars(leg.instrument.symbol)
                        i = b.last_index_at(min(t, self.now_ns))
                    except LookaheadError:
                        i = -1
                    if i >= 0:
                        px = float(b.c[i])
                        mkt = self._symbol_market(leg.instrument.symbol, leg.instrument)
                        hs = 0.5 * self.spread_model.full_spread(mkt.spread_estimate, px)
                        lq[k] = (leg, LegQuote(leg.instrument, t, px * (1 - hs), px * (1 + hs), Fidelity.T2_PROXY,
                                               "stock leg from bar close +/- estimated half-spread"))
            ex.evaluated = True
            status, leg_fills, net, reason = self.combo_model.evaluate(od, lq, ex.eligible_ns)
            if status == "REJECT":
                self._reject_working(od, reason)
                continue
            if status != "FILL":
                if od.tif == TimeInForce.IOC and opt_ts is not None and opt_ts > ex.eligible_ns:
                    self.om.cancel(od.id, self.now, f"IOC not filled: {reason}")
                continue
            snap_ts = max(q.ts for _, q in lq)
            units = abs(leg_fills[0][1] / (legs[0].ratio or 1)) if od.type == OrderType.COMBO else abs(leg_fills[0][1])
            for k, (inst, q, px, mid, fid) in enumerate(leg_fills):
                note_extra = lq[k][1].note if lq[k][1] is not None else ""
                reasons = (reason, "option quote snapshot") + ((note_extra,) if note_extra else ())
                dec = FillDecision(od, 0.0, snap_ts, q, px, mid, "TAKER", abs(px - mid) * abs(q) * inst.multiplier,
                                   0.0, fid, reasons)
                self._book(dec, inst, fee_key=od.id * 100 + k + 10**9 if od.type == OrderType.COMBO else od.id,
                           notify_manager=False)
            sgn = 1 if od.quantity > 0 else -1
            self._after_fill_order(od, sgn * units, pd.Timestamp(snap_ts, tz="UTC"), 0.0, snap_ts)

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
        ex = self.om.extras.get(o.id)
        istop = None
        if ex is not None:
            istop = ex.initial_stop
            if istop is None and ex.parent_id is not None:
                pex = self.om.extras.get(ex.parent_id)
                istop = pex.initial_stop if pex else None
        self.trades.on_fill(fill, pos_before, istop, o.tag)
        self._record_fill(fill, o.tag, d.reasons)
        self._pending_on_fill.append(fill)
        if notify_manager:
            self._after_fill_order(o, d.qty, ts, d.pos, d.ts_ns)

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

    def _after_fill_order(self, o: Order, qty: float, ts: pd.Timestamp, pos: float, ts_ns: int) -> None:
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
            t = templates[k]
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
                              eligible_ns=ts_ns, submitted_ns=ts_ns, in_bar_from=pos)
            cex.valid_session_ns = self._validity_session(ts_ns, cex.outside_rth) if t.tif == TimeInForce.DAY else None
            self.om.add(co, cex, ts)
            spec.child_ids[k] = co.id

    # ================================================================= marking
    def _mark(self, t_ns: int) -> None:
        for sym, p in list(self.account.positions.items()):
            if p.instrument.is_option:
                continue
            b = self.md.bars(sym)
            if b.n == 0 and self._uses_quotes(sym):
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
        for p in self.account.option_positions():
            q = self.portal.option_quote(p.instrument, self.now)
            if q is not None and q.bid == q.bid and q.ask == q.ask and q.ask > 0:
                p.mark = 0.5 * (q.bid + q.ask)
                p.mark_ts = pd.Timestamp(q.ts)
                if q.fidelity >= Fidelity.T2_PROXY:
                    self.fid.note(q.fidelity, "option marks from quote snapshots")
            else:
                age = self.now - p.mark_ts if p.mark_ts is not None else pd.Timedelta(days=999)
                if age > self.cfg.max_option_mark_staleness:
                    self.fid.flag(self.now, Fidelity.T2_PROXY, f"stale option mark {p.symbol}")

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
    def _enforce_session_rules(self) -> None:
        if self.daily or self._flattened_session or self._flat_by_ns is None:
            return
        fb = self._session_rules.get("flat_by")
        if self.now_ns >= self._flat_by_ns:
            self._flattened_session = True
            self.om.cancel_all(self.now, None, f"flat_by {fb}")
            for sym, p in sorted(self.account.positions.items()):
                if p.instrument.is_option:
                    continue
                o = Order(p.instrument, -p.qty, OrderType.MARKET, tif=TimeInForce.DAY, tag="flat_by")
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

    def submit(self, order: Order, outside_rth: bool = False, initial_stop: Optional[float] = None,
               activate_at: Optional[pd.Timestamp] = None, children: Optional[list[ChildTemplate]] = None,
               internal: bool = False) -> Order:
        now = self.now
        order.strategy_id = self.strategy.id
        order.submitted_at = now
        order.id = -1
        sym = None
        inst = order.instrument
        if order.type == OrderType.COMBO or order.legs:
            if not order.legs:
                return self.om.reject(order, "COMBO order without legs", now)
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
            sym = inst.symbol
            quotes = (not inst.is_option) and self._uses_quotes(sym)
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
        increases = sym is not None and abs(cur + q) > abs(cur) + 1e-12
        if not internal and not self.daily and increases:
            if self._flattened_session:
                return self.om.reject(order, f"after flat_by {self._session_rules.get('flat_by')}: no new exposure", now)
            ne = self._session_rules.get("no_new_entries_after")
            if ne and self._no_entry_ns is not None and self.now_ns > self._no_entry_ns:
                return self.om.reject(order, f"no new entries after {ne} ET", now)
        # short sales: locate + account type
        if inst is not None and not inst.is_option and t != OrderType.COMBO and q < 0:
            pend_sells = sum(o.remaining for o in self.om.active(sym) if o.remaining < 0
                             and self.om.extras[o.id].parent_id is None and not o.oco_group)
            if cur + pend_sells + q < -1e-9:
                order.is_short_sale = True
                if self.account.account_type == AccountType.CASH:
                    return self.om.reject(order, "cash account: short selling not permitted", now)
                if self.cfg.enforce_locates:
                    ok, fee, fid = self._borrow(sym)
                    if not ok:
                        return self.om.reject(order, "no locate / not shortable", now)
                    if fid >= Fidelity.T2_PROXY:
                        self.fid.note(Fidelity.T2_PROXY, f"borrow default {fee:.2%} p.a. (no borrow data)")
        # buying power
        if self.cfg.check_margin_on_submit and not internal and t != OrderType.COMBO and inst is not None and increases:
            px = order.limit_price or order.stop_price or self.last_price(sym)
            pending = []
            for o2 in self.om.active():
                if o2.instrument is None or self.om.extras[o2.id].parent_id is not None or o2.oco_group:
                    continue
                p2 = o2.limit_price or o2.stop_price or self.last_price(o2.instrument.symbol)
                if p2 == p2:
                    pending.append((o2.instrument, o2.remaining, p2))
            ok, why = self.margin.check_order(self.account, inst, q, px, self._underlying_price, pending)
            if not ok:
                return self.om.reject(order, why, now)
        # extras
        ex = OrderExtras(outside_rth=outside_rth, initial_stop=initial_stop,
                         activate_at_ns=int(pd.Timestamp(activate_at).value) if activate_at is not None else None,
                         submitted_ns=int(now.value), eligible_ns=int(order.eligible_at.value), is_flatten=internal)
        if t in (OrderType.MOO, OrderType.MOC):
            ex.target_session_ns = self._auction_session(ex.eligible_ns, t == OrderType.MOO)
        if order.tif == TimeInForce.DAY:
            ex.valid_session_ns = self._validity_session(ex.eligible_ns, outside_rth)
        if outside_rth and not any(s in self.data_sessions for s in ("PRE", "POST")):
            self.diag["warnings"].append(f"order {sym} allows outside RTH but no PRE/POST data is loaded")
        self.om.add(order, ex, now)
        if children:
            self.om.register_bracket(order, children, f"bracket-{order.id}")
        return order

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
            liq = {"EXERCISE": "EXERCISE", "ASSIGNMENT": "ASSIGNMENT"}.get(ev.kind, "SETTLEMENT")
            reason = {"EXERCISE": Reason.EXERCISE, "ASSIGNMENT": Reason.ASSIGNMENT}.get(ev.kind, Reason.CASH_SETTLEMENT)
            f = Fill(-1, inst, self.now, ev.option_qty, ev.option_price, CostBreakdown(), ev.fidelity,
                     liquidity=liq, note=f"{ev.kind}: {ev.note}")
            self.account.apply_fill(f, cash_reason=reason)
            self.trades.on_fill(f, pos_before, None, ev.kind.lower())
            self._record_fill(f, ev.kind.lower(), (ev.note,) if ev.note else ())
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
            evs = self.lifecycle.at_session_end(session, opts, _Prices(self))
            self._apply_option_events(evs)
            self._mark_options()
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
        # DAY order expiry
        self.om.expire_session(int(session.value), ts)
        # margin
        if self.account.account_type == AccountType.REG_T_MARGIN and self.cfg.margin_calls:
            snap = self.margin.snapshot(self.account, self._underlying_price)
            if snap.is_call:
                plan = self.margin.liquidation_plan(self.account, snap)
                self.diag["margin_calls"].append({"ts": ts, "equity": snap.equity, "maintenance": snap.maintenance,
                                                  "orders": [(p.symbol, dq) for p, dq in plan]})
                self.om.cancel_all(ts, None, "margin call")
                for p, dq in plan:
                    o = Order(p.instrument, dq, OrderType.MARKET if not p.instrument.is_option else OrderType.MARKET,
                              tif=TimeInForce.DAY, tag="margin_liquidation")
                    self.submit(o, internal=True)
        elif self.account.account_type == AccountType.CASH and self.account.cash < -1e-6:
            self.diag["cash_account_negative_cash_days"] += 1
        nav = self.account.nav
        _, c = self.session_open_close
        self.equity[c] = nav
        self.pos_rows.extend(self.account.snapshot(c))
        self.prior_nav = nav
        if self.cfg.debug:
            self._check_invariants()

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
            "risk_free_rate": float(rfd.mean() * 360.0) if len(rfd) else float("nan"),
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
        }
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
            return self.bt.md.session_close_price(symbol, session)
        except LookaheadError:
            return float("nan")

    def open(self, symbol: str, session) -> float:
        try:
            return self.bt.md.session_open_price(symbol, session)
        except LookaheadError:
            return float("nan")


def run_backtest(strategy: Strategy, portal, start, end, config: Optional[BacktestConfig] = None,
                 calendar=None, raise_errors: bool = False) -> BacktestResult:
    return Backtester(strategy, portal, config, calendar).run(start, end, raise_errors=raise_errors)
