#!/usr/bin/env python3
"""G8 Regime Filter - Live Runner

Daily cron script that:
1. Fetches BTC/GBP OHLC from Kraken
2. Computes G8 signal and regime status
3. Calculates required trade (if any)
4. Executes on Kraken (if --live flag)
5. Sends Telegram notification with regime chart

Usage:
    python3 live_runner.py              # dry-run (default)
    python3 live_runner.py --live       # execute real trades

Cron example (daily at 00:05 UTC):
    5 0 * * * cd /path/to/g8-standalone && python3 live_runner.py >> cron.log 2>&1
"""
import argparse
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from g8_strategy import compute_g8_signal, compute_g8_components
from kraken_client import KrakenClient
from telegram_bot import TelegramBot
from chart import generate_regime_chart

MIN_TRADE_PCT = 0.02
MIN_XBT_ORDER = 0.0001


def log(msg: str):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{ts}] {msg}", flush=True)


def fetch_close_from_kraken(client: KrakenClient) -> pd.Series:
    """Fetch ~400 days of daily BTC/GBP close prices from Kraken."""
    since = int((datetime.now(timezone.utc).timestamp() - 400 * 86400))
    candles = client.get_ohlc(since=since)
    rows = []
    for c in candles:
        ts = pd.Timestamp(int(c[0]), unit="s")
        rows.append({"date": ts, "close": float(c[4])})
    df = pd.DataFrame(rows).set_index("date").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df["close"]


def compute_current_btc_pct(xbt: float, gbp: float, price: float) -> float:
    """Current portfolio allocation to BTC (0.0-1.0)."""
    btc_value = xbt * price
    total = btc_value + gbp
    if total < 1.0:
        return 0.0
    return btc_value / total


def calculate_trade(current_pct: float, target_pct: float,
                    total_value_gbp: float, price: float) -> dict:
    """Determine trade needed to reach target allocation."""
    delta = target_pct - current_pct
    if abs(delta) < MIN_TRADE_PCT:
        return {"action": "HOLD", "side": None, "volume": 0.0, "delta": delta}

    trade_gbp = abs(delta) * total_value_gbp
    volume_xbt = trade_gbp / price

    if volume_xbt < MIN_XBT_ORDER:
        return {"action": "HOLD", "side": None, "volume": 0.0, "delta": delta}

    side = "buy" if delta > 0 else "sell"
    return {
        "action": side.upper(),
        "side": side,
        "volume": round(volume_xbt, 8),
        "delta": delta,
    }


def format_telegram_message(date_str: str, price: float, regime: str,
                            prob: float, current_pct: float,
                            target_pct: float, trade: dict,
                            portfolio_gbp: float, is_live: bool) -> str:
    mode = "LIVE" if is_live else "DRY RUN"
    trade_line = "HOLD" if trade["action"] == "HOLD" else (
        f"{trade['action']} {trade['volume']:.6f} XBT"
    )
    return (
        f"<b>G8 Regime Filter - Daily Update</b>\n"
        f"<code>{'=' * 36}</code>\n"
        f"<b>Date:</b>      {date_str}\n"
        f"<b>BTC/GBP:</b>   {price:,.0f}\n"
        f"<b>Regime:</b>    {regime} (prob={prob:.2f})\n"
        f"<b>Position:</b>  {current_pct*100:.0f}% -> {target_pct*100:.0f}%\n"
        f"<b>Trade:</b>     {trade_line}\n"
        f"<b>Portfolio:</b> {portfolio_gbp:,.0f} GBP\n"
        f"<code>{'=' * 36}</code>\n"
        f"<i>Mode: {mode}</i>"
    )


def main():
    parser = argparse.ArgumentParser(description="G8 Regime Filter Live Runner")
    parser.add_argument("--live", action="store_true",
                        help="Execute real trades (default: dry-run)")
    args = parser.parse_args()
    is_live = args.live

    log(f"Starting G8 live runner ({'LIVE' if is_live else 'DRY RUN'})")

    kraken = KrakenClient()
    log("Fetching OHLC data from Kraken...")
    close = fetch_close_from_kraken(kraken)
    log(f"Got {len(close)} daily candles: {close.index[0].date()} to {close.index[-1].date()}")

    if len(close) < 250:
        log(f"ERROR: Need >= 250 days for EMA200, got {len(close)}. Aborting.")
        sys.exit(1)

    log("Computing G8 signal...")
    signal = compute_g8_signal(close)
    components = compute_g8_components(close)
    target_pct = float(signal.iloc[-1])
    regime = components["regime"].iloc[-1]
    prob = float(components["prob"].iloc[-1])
    log(f"Regime: {regime}, prob={prob:.3f}, target_position={target_pct*100:.1f}%")

    log("Fetching Kraken balances...")
    xbt, gbp = kraken.get_xbt_gbp_balance()
    ticker = kraken.get_ticker()
    price = ticker["last"]
    current_pct = compute_current_btc_pct(xbt, gbp, price)
    total_value = xbt * price + gbp
    log(f"Balances: {xbt:.8f} XBT + {gbp:.2f} GBP = {total_value:.2f} GBP total")
    log(f"Current allocation: {current_pct*100:.1f}% BTC")

    trade = calculate_trade(current_pct, target_pct, total_value, price)
    log(f"Trade decision: {trade['action']} "
        f"(delta={trade['delta']*100:+.1f}%, volume={trade['volume']:.8f} XBT)")

    if trade["action"] != "HOLD" and is_live:
        log(f"EXECUTING {trade['side']} {trade['volume']:.8f} XBT on Kraken...")
        try:
            result = kraken.place_market_order(trade["side"], trade["volume"])
            txids = result.get("txid", [])
            log(f"Order placed: {txids}")
        except Exception as e:
            log(f"ORDER FAILED: {e}")
            trade["action"] = f"FAILED: {e}"
    elif trade["action"] != "HOLD":
        log(f"DRY RUN: would {trade['side']} {trade['volume']:.8f} XBT")

    log("Generating regime chart...")
    chart_png = generate_regime_chart(close, components, lookback_days=365)
    log(f"Chart generated ({len(chart_png)} bytes)")

    log("Sending Telegram notification...")
    try:
        tg = TelegramBot()
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        msg = format_telegram_message(
            date_str, price, regime, prob,
            current_pct, target_pct, trade, total_value, is_live
        )
        tg.send_photo(chart_png, caption=msg)
        log("Telegram notification sent")
    except Exception as e:
        log(f"Telegram error (non-fatal): {e}")

    log("Done.")


if __name__ == "__main__":
    main()
