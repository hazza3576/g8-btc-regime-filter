"""Telegram Bot API wrapper for G8 regime filter notifications.

Requires TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID environment variables.
"""
import os
from typing import Optional

import requests

BASE_URL = "https://api.telegram.org"


class TelegramBot:
    def __init__(self, token: Optional[str] = None,
                 chat_id: Optional[str] = None):
        self.token = token or os.environ.get("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID", "")
        if not self.token or not self.chat_id:
            raise RuntimeError(
                "TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID required"
            )

    @property
    def _url(self) -> str:
        return f"{BASE_URL}/bot{self.token}"

    def send_message(self, text: str,
                     parse_mode: str = "HTML") -> dict:
        resp = requests.post(
            f"{self._url}/sendMessage",
            data={
                "chat_id": self.chat_id,
                "text": text,
                "parse_mode": parse_mode,
            },
            timeout=30,
        )
        resp.raise_for_status()
        result = resp.json()
        if not result.get("ok"):
            raise RuntimeError(f"Telegram error: {result}")
        return result

    def send_photo(self, image_bytes: bytes, caption: str = "",
                   parse_mode: str = "HTML") -> dict:
        resp = requests.post(
            f"{self._url}/sendPhoto",
            data={
                "chat_id": self.chat_id,
                "caption": caption,
                "parse_mode": parse_mode,
            },
            files={"photo": ("regime_chart.png", image_bytes, "image/png")},
            timeout=60,
        )
        resp.raise_for_status()
        result = resp.json()
        if not result.get("ok"):
            raise RuntimeError(f"Telegram error: {result}")
        return result
