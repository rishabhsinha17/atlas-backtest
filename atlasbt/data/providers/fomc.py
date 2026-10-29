"""FOMC statement release timestamps (federalreserve.gov).

Statement links are scraped from /monetarypolicy/fomccalendars.htm (recent years) and
/monetarypolicy/fomchistorical<YEAR>.htm (1994-...). Each statement page is fetched and
its release line parsed ("For release at 2:00 p.m. EST" / "For immediate release").

known_at:
  * parsed clock time on the page (2008+ pages carry it)          -> T1_BAR
  * else a hand-verified time for well-known unscheduled actions  -> T2_PROXY
  * else the era rule: 14:15 ET (1994-2012 scheduled meetings; the 14:00 ET release time
    starts with the 2013 meetings -- pages from 2013 on state 2:00 p.m.)          -> T2_PROXY
Events: kind 'FOMC', payload {type:'STATEMENT', scheduled:bool, url, time_source}.
"""
from __future__ import annotations

import logging
import re
from typing import Optional

import pandas as pd
from bs4 import BeautifulSoup

from atlasbt.core.clock import et
from atlasbt.core.types import Fidelity

from ..http import FetchError, client
from ..store import ParquetStore, default_store

log = logging.getLogger("atlasbt.data.fomc")
SOURCE = "fomc"
BASE = "https://www.federalreserve.gov"
CAL_URL = BASE + "/monetarypolicy/fomccalendars.htm"
HIST_URL = BASE + "/monetarypolicy/fomchistorical{year}.htm"

# Unscheduled/intermeeting statements with widely reported release times (ET). T2.
# 1994 statements (first year of announced policy changes): times from the Gurkaynak-Sack-Swanson
# (2005) intraday event table, not printed on the Fed pages -> T2 with source note.
GSS_1994 = {"1994-02-04": "11:05", "1994-03-22": "14:20", "1994-04-18": "10:06", "1994-05-17": "14:26",
            "1994-08-16": "13:18", "1994-11-15": "14:20"}
MANUAL_TIMES = {
    "1998-10-15": "15:15", "2001-01-03": "13:13", "2001-04-18": "10:54", "2001-09-17": "08:20",
    "2007-08-10": "09:15", "2007-08-17": "08:15", "2008-01-22": "08:20", "2008-10-08": "07:00",
    "2020-03-03": "10:00", "2020-03-15": "17:00",
    # 2011-2012 press-conference meetings: statement at 12:30 ET, press conference 14:15 ET
    "2011-04-27": "12:30", "2011-06-22": "12:30", "2011-11-02": "12:30", "2012-01-25": "12:30",
    "2012-04-25": "12:30", "2012-06-20": "12:30", "2012-09-13": "12:30", "2012-12-12": "12:30",
}
UNSCHEDULED = {"1994-04-18", "1998-10-15", "2001-01-03", "2001-04-18", "2001-09-17", "2007-08-10", "2007-08-17",
               "2008-01-22", "2008-10-08", "2020-03-03", "2020-03-15"}
_TIME_RE = re.compile(r"for\s+release\s+at\s+(\d{1,2}):(\d{2})\s*([ap])\.?\s*m", re.I)


def statement_links(html: str, calendars_page: bool = False) -> list[tuple[str, str]]:
    """-> list of (YYYY-MM-DD, absolute url) for FOMC statements on a calendar page."""
    soup = BeautifulSoup(html, "lxml")
    out = {}
    for a in soup.find_all("a", href=True):
        href, text = a["href"], a.get_text(" ", strip=True)
        m = re.search(r"(?:monetary/?|/fomc/)(\d{8})[a-z]?(?:\.htm|DEFAULT\.htm)", href, re.I) or \
            re.search(r"/(?:monetary|general)/\d{4}/(\d{8})/?", href)
        if not m:
            continue
        if calendars_page:
            if not (re.search(r"pressreleases/monetary\d{8}a\.htm$", href) and text in ("HTML", "Statement")):
                continue
        elif text != "Statement":
            continue
        d = pd.Timestamp(m.group(1)).strftime("%Y-%m-%d")
        href = re.sub(r"/newsevents/press/monetary/(\d{8}[a-z]?\.htm)", r"/newsevents/pressreleases/monetary\1", href)
        out.setdefault(d, href if href.startswith("http") else BASE + href)
    return sorted(out.items())


def parse_release_time(html: str) -> Optional[str]:
    text = re.sub(r"\s+", " ", BeautifulSoup(html, "lxml").get_text(" "))
    m = _TIME_RE.search(text)
    if not m:
        return None
    h, mi, ap = int(m.group(1)), int(m.group(2)), m.group(3).lower()
    if ap == "p" and h != 12:
        h += 12
    if ap == "a" and h == 12:
        h = 0
    return f"{h:02d}:{mi:02d}"


def fetch_fomc(store: Optional[ParquetStore] = None, first_year: int = 1994) -> dict:
    store = store or default_store()
    links: dict[str, str] = {}
    cal = client().get(CAL_URL, ttl=86400).decode("utf-8", "ignore")
    for d, u in statement_links(cal, calendars_page=True):
        links.setdefault(d, u)
    this_year = pd.Timestamp.now().year
    for y in range(first_year, this_year + 1):
        try:
            body = client().get(HIST_URL.format(year=y), allow_404=True,
                                ttl=None if y < this_year - 6 else 7 * 86400)
        except FetchError as e:
            log.warning("fomc historical %s: %s", y, e)
            continue
        if body:
            for d, u in statement_links(body.decode("utf-8", "ignore")):
                links.setdefault(d, u)
    rows = []
    for d, u in sorted(links.items()):
        t, src = None, None
        try:
            page = client().get(u, allow_404=True)
            t = parse_release_time(page.decode("utf-8", "ignore")) if page else None
            src = "page" if t else None
        except FetchError as e:
            log.warning("fomc statement %s: %s", u, e)
        if t is None and d in GSS_1994:
            t, src = GSS_1994[d], "gss_2005_table"
        if t is None and d in MANUAL_TIMES:
            t, src = MANUAL_TIMES[d], "manual_unscheduled"
        if t is None:
            # 14:00 ET from the March 2013 meeting on; Jan-2013 and earlier: 14:15 ET (the later,
            # i.e. conservative, of the plausible times when the page does not state it)
            t, src = ("14:00", "era_rule_2013+") if d >= "2013-03-01" else ("14:15", "era_rule_1994-2013")
        known = et(d, t)
        rows.append({"kind": "FOMC", "symbol": None, "known_at": known, "effective_at": known,
                     "type": "STATEMENT", "date": pd.Timestamp(d), "release_time_et": t, "time_source": src,
                     "scheduled": d not in UNSCHEDULED, "url": u,
                     "fidelity": int(Fidelity.T1_BAR if src == "page" else Fidelity.T2_PROXY),
                     "source": SOURCE})
    df = pd.DataFrame(rows)
    store.write("events", SOURCE, "FOMC", df, key=["date"], replace=True, fidelity=int(Fidelity.T1_BAR))
    return {"statements": len(df), "first": str(df["date"].min().date()), "last": str(df["date"].max().date()),
            "time_sources": df["time_source"].value_counts().to_dict()}
