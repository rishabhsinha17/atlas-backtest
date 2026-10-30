"""Every symbol named in a strategy's static universe or explicit requirement must be in the core fetch set,
so `python -m atlasbt.data.fetch --set core` makes it testable (found: S108/S111 ETFs were missing)."""
from atlasbt.data import universes as U
from atlasbt.strategies import registry as R

# Index symbols / vol indices come from the Cboe/FRED tasks, not the ETF list.
NON_ETF_SOURCES = {"VIX", "^VIX", "SPX", "^GSPC", "VIX9D", "VIX3M", "VIX6M", "VIX1D", "SKEW"}


def test_static_strategy_symbols_are_in_core_fetch_set():
    core = set(U.CORE_ETFS) | set(U.MEGACAPS) | NON_ETF_SOURCES
    missing = set()
    for sid in R.all_ids():
        c = R.get(sid)
        names = set(getattr(c, "universe", ()) or ())
        for q in getattr(c, "requires", ()) or ():
            names |= {s for s in (q.symbols or ()) if s != "*"}
        missing |= {s for s in names if "~" not in s} - core
    assert not missing, f"add to atlasbt.data.universes CORE_ETFS: {sorted(missing)}"
