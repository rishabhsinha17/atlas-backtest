"""Static PNG charts for per-strategy reports (matplotlib, Agg backend).

Palette: reference data-viz palette (single series = categorical slot 1
blue #2a78d6; diverging monthly heatmap = blue <-> red with a neutral gray
midpoint #f0efec; text in ink tokens, recessive grid). One y-axis per chart.
Every function returns the written path, or None when matplotlib is missing.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
BLUE = "#2a78d6"
RED = "#e34948"
NEUTRAL = "#f0efec"


def _plt():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except Exception:  # pragma: no cover
        return None


def _style(ax, title: str):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=11, fontweight="bold")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _fig(plt, w=8.0, h=3.2):
    fig, ax = plt.subplots(figsize=(w, h), dpi=110)
    fig.patch.set_facecolor(SURFACE)
    return fig, ax


def _save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    import matplotlib.pyplot as plt

    plt.close(fig)
    return str(path)


def equity_chart(equity: pd.Series, path, title="Equity (NAV, log scale)"):
    plt = _plt()
    if plt is None or equity.empty:
        return None
    fig, ax = _fig(plt)
    ax.plot(equity.index, equity.to_numpy(float), color=BLUE, linewidth=2)
    if (equity > 0).all():
        ax.set_yscale("log")
    _style(ax, title)
    return _save(fig, path)


def drawdown_chart(equity: pd.Series, path, title="Drawdown"):
    plt = _plt()
    if plt is None or equity.empty:
        return None
    dd = (equity / equity.cummax() - 1.0) * 100
    fig, ax = _fig(plt)
    ax.fill_between(dd.index, dd.to_numpy(), 0, color=BLUE, alpha=0.35, linewidth=0)
    ax.plot(dd.index, dd.to_numpy(), color=BLUE, linewidth=1.2)
    ax.set_ylabel("% below peak", color=INK2, fontsize=8)
    _style(ax, title)
    return _save(fig, path)


def rolling_sharpe_chart(returns: pd.Series, path, window: int = 126, title=None):
    plt = _plt()
    if plt is None or len(returns) <= window:
        return None
    r = returns.astype(float)
    rs = r.rolling(window).mean() / r.rolling(window).std() * np.sqrt(252)
    fig, ax = _fig(plt)
    ax.axhline(0, color=INK2, linewidth=0.8)
    ax.plot(rs.index, rs.to_numpy(), color=BLUE, linewidth=1.6)
    _style(ax, title or f"Rolling {window}-day Sharpe (annualised)")
    return _save(fig, path)


def monthly_table(returns: pd.Series) -> pd.DataFrame:
    r = returns.astype(float)
    idx = pd.DatetimeIndex(r.index)
    if idx.tz is not None:
        idx = idx.tz_convert("America/New_York").tz_localize(None)
    s = pd.Series(r.to_numpy(), index=idx)
    m = (1 + s).groupby([idx.year, idx.month]).prod() - 1
    tab = m.unstack()
    tab.index.name, tab.columns.name = "year", "month"
    return tab


def monthly_heatmap(returns: pd.Series, path, title="Monthly returns (%)"):
    plt = _plt()
    if plt is None or returns.empty:
        return None
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    tab = monthly_table(returns) * 100
    tab = tab.reindex(columns=range(1, 13))
    cmap = LinearSegmentedColormap.from_list("div", [RED, NEUTRAL, BLUE])
    v = np.nanmax(np.abs(tab.to_numpy())) if np.isfinite(tab.to_numpy()).any() else 1.0
    v = v if v > 0 else 1.0
    h = max(2.0, 0.32 * len(tab) + 1.2)
    fig, ax = _fig(plt, 8.0, h)
    im = ax.imshow(tab.to_numpy(), cmap=cmap, norm=TwoSlopeNorm(0, -v, v), aspect="auto")
    ax.set_xticks(range(12), ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
    ax.set_yticks(range(len(tab)), [str(y) for y in tab.index])
    if len(tab) <= 30:
        for i in range(tab.shape[0]):
            for j in range(12):
                val = tab.iat[i, j]
                if np.isfinite(val):
                    ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=6.5, color=INK)
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=11, fontweight="bold")
    ax.tick_params(colors=INK2, labelsize=7, length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.ax.tick_params(colors=INK2, labelsize=7)
    cb.outline.set_visible(False)
    return _save(fig, path)


def r_distribution_chart(r_multiples, path, title="Trade outcome distribution (R multiples)"):
    plt = _plt()
    r = pd.Series(r_multiples, dtype=float).dropna() if r_multiples is not None else pd.Series(dtype=float)
    if plt is None or r.empty:
        return None
    fig, ax = _fig(plt)
    bins = min(40, max(8, int(np.sqrt(len(r)) * 2)))
    ax.hist(r.to_numpy(), bins=bins, color=BLUE, edgecolor=SURFACE, linewidth=1.0)
    ax.axvline(0, color=INK2, linewidth=0.8)
    ax.axvline(r.mean(), color=INK, linewidth=1.2, linestyle="--")
    ax.text(r.mean(), ax.get_ylim()[1] * 0.95, f" mean {r.mean():.2f}R ", color=INK, fontsize=8, va="top",
            bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1.5})
    ax.set_xlabel("R multiple", color=INK2, fontsize=8)
    ax.set_ylabel("trades", color=INK2, fontsize=8)
    _style(ax, title)
    return _save(fig, path)
