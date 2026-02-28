"""Regime chart generator for Telegram notifications.

Produces a PNG with two panels:
  Top:    BTC price + EMA200, green/red background (bull/bear regime)
  Bottom: Position size (0-100%) as filled area
"""
import io
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd


def generate_regime_chart(close: pd.Series, components: dict,
                          lookback_days: int = 365) -> bytes:
    """Generate regime chart and return PNG bytes.

    Args:
        close: daily close prices
        components: dict from compute_g8_components() with keys:
            signal, prob, ema200, ema50, regime
        lookback_days: how many days to show (default 365)
    """
    cutoff = close.index[-1] - pd.Timedelta(days=lookback_days)
    mask = close.index >= cutoff
    c = close[mask]
    sig = components["signal"][mask]
    prob = components["prob"][mask]
    ema200 = components["ema200"][mask]
    regime = components["regime"][mask]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 7), height_ratios=[3, 1],
        sharex=True, gridspec_kw={"hspace": 0.08}
    )

    # -- Top panel: price + regime shading --
    ax1.plot(c.index, c.values, color="#1a1a2e", linewidth=1.5, label="BTC/GBP")
    ax1.plot(ema200.index, ema200.values, color="#e94560", linewidth=1,
             alpha=0.7, linestyle="--", label="EMA200")

    bull = regime == "BULL"
    _shade_regions(ax1, c.index, bull, "#2ecc71", "#e74c3c", alpha=0.12)

    ax1.set_ylabel("BTC / GBP", fontsize=11)
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_title("G8 Regime Filter", fontsize=13, fontweight="bold")

    current_regime = regime.iloc[-1]
    current_prob = prob.iloc[-1]
    current_pos = sig.iloc[-1] * 100
    ax1.annotate(
        f"{current_regime}  prob={current_prob:.2f}  pos={current_pos:.0f}%",
        xy=(0.99, 0.97), xycoords="axes fraction", fontsize=10,
        ha="right", va="top",
        bbox=dict(
            boxstyle="round,pad=0.3",
            fc="#2ecc71" if current_regime == "BULL" else "#e74c3c",
            alpha=0.8, ec="none"
        ),
        color="white", fontweight="bold"
    )

    # -- Bottom panel: position sizing --
    ax2.fill_between(sig.index, sig.values * 100, 0,
                     color="#3498db", alpha=0.5, step="post")
    ax2.plot(sig.index, sig.values * 100, color="#2c3e50",
             linewidth=1, drawstyle="steps-post")
    ax2.set_ylabel("Position %", fontsize=11)
    ax2.set_ylim(-2, 105)
    ax2.set_yticks([0, 25, 50, 75, 100])
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    fig.autofmt_xdate(rotation=30)

    fig.text(0.99, 0.01, f"Generated {datetime.utcnow():%Y-%m-%d %H:%M} UTC",
             ha="right", fontsize=7, color="gray")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def _shade_regions(ax, index, bull_mask, bull_color, bear_color, alpha=0.12):
    """Shade background green for bull, red for bear."""
    ymin, ymax = ax.get_ylim()
    prev = None
    start = index[0]
    for i, (dt, is_bull) in enumerate(zip(index, bull_mask)):
        if prev is not None and is_bull != prev:
            color = bull_color if prev else bear_color
            ax.axvspan(start, dt, color=color, alpha=alpha)
            start = dt
        prev = is_bull
    if prev is not None:
        color = bull_color if prev else bear_color
        ax.axvspan(start, index[-1], color=color, alpha=alpha)
