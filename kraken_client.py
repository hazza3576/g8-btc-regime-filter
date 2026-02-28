"""Kraken REST API client for G8 regime filter.

Authentication: HMAC-SHA512 per https://docs.kraken.com/api/docs/guides/spot-rest-auth
Requires KRAKEN_API_KEY and KRAKEN_API_SECRET environment variables.
"""
import os
import time
import base64
import hashlib
import hmac
import urllib.parse
from typing import Optional

import requests

API_URL = "https://api.kraken.com"
PAIR = "XBTGBP"


class KrakenClient:
    def __init__(self, api_key: Optional[str] = None,
                 api_secret: Optional[str] = None):
        self.api_key = api_key or os.environ.get("KRAKEN_API_KEY", "")
        self.api_secret = api_secret or os.environ.get("KRAKEN_API_SECRET", "")
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "G8-RegimeFilter/1.0"})

    def _sign(self, urlpath: str, data: dict) -> str:
        post_data = urllib.parse.urlencode(data)
        encoded = (str(data["nonce"]) + post_data).encode()
        message = urlpath.encode() + hashlib.sha256(encoded).digest()
        secret = base64.b64decode(self.api_secret)
        signature = hmac.new(secret, message, hashlib.sha512)
        return base64.b64encode(signature.digest()).decode()

    def _public(self, endpoint: str, params: Optional[dict] = None) -> dict:
        url = f"{API_URL}/0/public/{endpoint}"
        resp = self.session.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if data.get("error"):
            raise RuntimeError(f"Kraken API error: {data['error']}")
        return data["result"]

    def _private(self, endpoint: str, params: Optional[dict] = None) -> dict:
        if not self.api_key or not self.api_secret:
            raise RuntimeError("KRAKEN_API_KEY and KRAKEN_API_SECRET required")
        urlpath = f"/0/private/{endpoint}"
        url = f"{API_URL}{urlpath}"
        data = params or {}
        data["nonce"] = str(int(time.time() * 1000))
        headers = {
            "API-Key": self.api_key,
            "API-Sign": self._sign(urlpath, data),
        }
        resp = self.session.post(url, data=data, headers=headers, timeout=30)
        resp.raise_for_status()
        result = resp.json()
        if result.get("error"):
            raise RuntimeError(f"Kraken API error: {result['error']}")
        return result["result"]

    def get_ticker(self, pair: str = PAIR) -> dict:
        """Returns ticker with ask, bid, last price."""
        result = self._public("Ticker", {"pair": pair})
        info = list(result.values())[0]
        return {
            "ask": float(info["a"][0]),
            "bid": float(info["b"][0]),
            "last": float(info["c"][0]),
        }

    def get_ohlc(self, pair: str = PAIR, interval: int = 1440,
                 since: Optional[int] = None) -> list:
        """Returns daily OHLC candles. interval=1440 is 1 day.
        Each candle: [time, open, high, low, close, vwap, volume, count]
        """
        params = {"pair": pair, "interval": interval}
        if since:
            params["since"] = since
        result = self._public("OHLC", params)
        candles = list(result.values())[0]
        if isinstance(candles, dict):
            candles = list(result.values())[0]
        return candles

    def get_balance(self) -> dict:
        """Returns dict with available balances (e.g. ZGBP, XXBT)."""
        return self._private("Balance")

    def get_xbt_gbp_balance(self) -> tuple:
        """Returns (xbt_balance, gbp_balance) as floats."""
        bal = self.get_balance()
        xbt = float(bal.get("XXBT", bal.get("XBT", 0)))
        gbp = float(bal.get("ZGBP", bal.get("GBP", 0)))
        return xbt, gbp

    def place_market_order(self, side: str, volume: float,
                           pair: str = PAIR) -> dict:
        """Place a market buy or sell order.
        side: 'buy' or 'sell'
        volume: amount of XBT to trade
        """
        if side not in ("buy", "sell"):
            raise ValueError(f"side must be 'buy' or 'sell', got '{side}'")
        if volume <= 0:
            raise ValueError(f"volume must be positive, got {volume}")
        params = {
            "pair": pair,
            "type": side,
            "ordertype": "market",
            "volume": f"{volume:.8f}",
        }
        return self._private("AddOrder", params)
