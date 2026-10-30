"""Static universe definitions (fixed ETF lists from the strategy sheet) and
sector ETF metadata. Point-in-time universes (SP500) are served by the Portal
from the store (see providers/sp500.py)."""
from __future__ import annotations

import pandas as pd

# Select Sector SPDRs with inception (first trading) dates. XLRE launched
# 2015-10-08 (Real Estate became the 11th GICS sector 2016-09-19); XLC launched
# 2018-06-19 (Communication Services GICS reshuffle 2018-09-24).
SECTOR_SPDRS = {
    "XLB": ("Materials", "1998-12-22"),
    "XLE": ("Energy", "1998-12-22"),
    "XLF": ("Financials", "1998-12-22"),
    "XLI": ("Industrials", "1998-12-22"),
    "XLK": ("Information Technology", "1998-12-22"),
    "XLP": ("Consumer Staples", "1998-12-22"),
    "XLU": ("Utilities", "1998-12-22"),
    "XLV": ("Health Care", "1998-12-22"),
    "XLY": ("Consumer Discretionary", "1998-12-22"),
    "XLRE": ("Real Estate", "2015-10-08"),
    "XLC": ("Communication Services", "2018-06-19"),
}
SECTOR_TO_ETF = {v[0]: k for k, v in SECTOR_SPDRS.items()}

BROAD_ETFS = ["SPY", "QQQ", "IWM", "DIA", "RSP"]
BOND_CASH_ETFS = ["TLT", "IEF", "SHY", "BIL", "SHV", "SGOV", "AGG", "LQD", "HYG"]
OTHER_ETFS = ["GLD", "EFA", "EEM", "VNQ"]
LEVERAGED_ETFS = ["SSO", "UPRO", "QLD", "TQQQ", "SH", "SDS", "PSQ"]
# unlevered reference for each leveraged/inverse fund (S118/S119)
LEVERAGED_REFERENCE = {"SSO": ("SPY", 2), "UPRO": ("SPY", 3), "QLD": ("QQQ", 2), "TQQQ": ("QQQ", 3),
                       "SH": ("SPY", -1), "SDS": ("SPY", -2), "PSQ": ("QQQ", -1)}

# ETFs named by strategy universes beyond the core lists (S108 SPDR-vs-Vanguard sector pairs, S111 same-index
# pairs). Kept here so `fetch --set core` covers every static strategy symbol.
PAIR_ETFS = ["VGT", "VFH", "VHT", "VPU", "VDC", "VCR", "VAW", "VDE", "VIS",   # Vanguard sector (S108)
             "IVV", "MDY", "IJH", "IJR", "VIOO", "VTWO", "QQQM"]              # same-index pairs (S111)
CORE_ETFS = BROAD_ETFS + list(SECTOR_SPDRS) + BOND_CASH_ETFS + OTHER_ETFS + LEVERAGED_ETFS
CORE_ETFS = CORE_ETFS + [s for s in PAIR_ETFS if s not in CORE_ETFS]

MEGACAPS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "GOOG", "META", "TSLA", "BRK-B", "AVGO", "JPM", "LLY",
            "V", "UNH", "XOM", "MA", "COST", "HD", "PG", "JNJ", "WMT", "NFLX", "AMD", "BAC", "ABBV", "CRM",
            "KO", "ORCL", "PEP", "CVX", "MRK", "ADBE", "INTC", "CSCO"]
INTRADAY_CORE = ["SPY", "QQQ", "IWM", "DIA", "TLT", "GLD"] + list(SECTOR_SPDRS) + MEGACAPS

# Fixed (frozen, pre-declared) ETF lists used by the sheet's allocation strategies.
STATIC_UNIVERSES: dict[str, list[str]] = {
    "SECTOR_SPDR": list(SECTOR_SPDRS),
    "SECTOR_SPDR_9": [k for k, v in SECTOR_SPDRS.items() if v[1] == "1998-12-22"],
    "BROAD_INDEX_ETF": ["SPY", "QQQ", "IWM"],            # S116 gap fade
    "GEM_DUAL_MOMENTUM": ["SPY", "EFA", "AGG", "BIL"],   # S089-style
    "ACCEL_DUAL_MOM": ["SPY", "IEF", "GLD", "SHY"],      # S101: equity, int. T-note, gold, short Treasury
    "CROSS_ASSET": ["SPY", "EFA", "BIL", "TLT", "GLD", "VNQ"],   # S122
    "TREASURY_DURATION": ["SHY", "IEF", "TLT"],          # S124
    "TBILL_PROXY": ["BIL", "SHV", "SGOV"],
    "LEVERAGED": LEVERAGED_ETFS,
    "EQUAL_WEIGHT_PAIR": ["RSP", "SPY"],                 # S123
    "CORE_ETF": CORE_ETFS,
    "MEGACAP": MEGACAPS,
}

# SIC code -> GICS-like sector (coarse, documented approximation; T2).
# Ranges per SEC SIC manual divisions, refined to match the 11 SPDR sectors.
_SIC_RULES = [
    ((100, 999), "Consumer Staples"),          # agriculture
    ((1000, 1099), "Materials"),               # metal mining
    ((1200, 1399), "Energy"),                  # coal, oil & gas extraction
    ((1400, 1499), "Materials"),
    ((1500, 1799), "Industrials"),             # construction
    ((2000, 2199), "Consumer Staples"),        # food, tobacco
    ((2200, 2399), "Consumer Discretionary"),  # textiles, apparel
    ((2400, 2499), "Materials"),               # lumber
    ((2500, 2599), "Consumer Discretionary"),  # furniture
    ((2600, 2699), "Materials"),               # paper
    ((2700, 2799), "Communication Services"),  # publishing
    ((2800, 2829), "Materials"),               # chemicals
    ((2830, 2836), "Health Care"),             # pharma
    ((2837, 2839), "Materials"),
    ((2840, 2844), "Consumer Staples"),        # soaps, cosmetics
    ((2845, 2899), "Materials"),
    ((2900, 2999), "Energy"),                  # petroleum refining
    ((3000, 3299), "Materials"),               # rubber, plastics, glass
    ((3300, 3399), "Materials"),               # primary metals
    ((3400, 3499), "Industrials"),
    ((3500, 3569), "Industrials"),
    ((3570, 3579), "Information Technology"),  # computers
    ((3580, 3659), "Industrials"),
    ((3660, 3699), "Information Technology"),  # communications equip, semis
    ((3700, 3719), "Consumer Discretionary"),  # motor vehicles
    ((3720, 3799), "Industrials"),             # aerospace, rail
    ((3800, 3849), "Information Technology"),  # instruments
    ((3850, 3851), "Health Care"),
    ((3841, 3845), "Health Care"),
    ((3852, 3899), "Industrials"),
    ((3900, 3999), "Consumer Discretionary"),
    ((4000, 4799), "Industrials"),             # transportation
    ((4800, 4899), "Communication Services"),  # telecom, broadcasting
    ((4900, 4949), "Utilities"),
    ((4950, 4959), "Industrials"),             # waste mgmt
    ((4960, 4999), "Utilities"),
    ((5000, 5199), "Industrials"),             # wholesale
    ((5200, 5399), "Consumer Discretionary"),
    ((5400, 5499), "Consumer Staples"),        # food stores
    ((5500, 5799), "Consumer Discretionary"),
    ((5800, 5899), "Consumer Discretionary"),  # restaurants
    ((5900, 5912), "Consumer Staples"),        # drug stores
    ((5913, 5999), "Consumer Discretionary"),
    ((6000, 6499), "Financials"),
    ((6500, 6553), "Real Estate"),
    ((6798, 6798), "Real Estate"),             # REITs
    ((6700, 6797), "Financials"),
    ((6799, 6799), "Financials"),
    ((7000, 7299), "Consumer Discretionary"),  # hotels, personal services
    ((7300, 7369), "Industrials"),             # business services
    ((7370, 7379), "Information Technology"),  # software
    ((7380, 7399), "Industrials"),
    ((7800, 7999), "Communication Services"),  # motion pictures, amusement
    ((7500, 7799), "Consumer Discretionary"),
    ((8000, 8099), "Health Care"),
    ((8100, 8999), "Industrials"),
]


def sic_to_sector(sic) -> str | None:
    try:
        s = int(sic)
    except (TypeError, ValueError):
        return None
    # specific (narrow) rules first
    for (a, b), sec in sorted(_SIC_RULES, key=lambda r: r[0][1] - r[0][0]):
        if a <= s <= b:
            return sec
    return None


def sector_etf_for(sector: str | None, as_of) -> str | None:
    """Sector SPDR for a sector *if it existed at as_of*; else None."""
    if not sector or sector not in SECTOR_TO_ETF:
        return None
    etf = SECTOR_TO_ETF[sector]
    t = pd.Timestamp(as_of)
    if t.tzinfo is not None:
        t = t.tz_convert("America/New_York").tz_localize(None)
    if t < pd.Timestamp(SECTOR_SPDRS[etf][1]):
        return None
    return etf
