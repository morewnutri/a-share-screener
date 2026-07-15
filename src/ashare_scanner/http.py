from __future__ import annotations

import json
import random
import threading
import time
from typing import Any

import requests


USER_AGENTS = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_5) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36",
)


class HttpClient:
    def __init__(self, timeout: float, max_retries: int) -> None:
        self.timeout = timeout
        self.max_retries = max_retries
        self._local = threading.local()

    def _session(self) -> requests.Session:
        if not hasattr(self._local, "session"):
            session = requests.Session()
            adapter = requests.adapters.HTTPAdapter(pool_connections=32, pool_maxsize=32)
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            self._local.session = session
        return self._local.session

    @staticmethod
    def headers(referer: str = "https://quote.eastmoney.com/") -> dict[str, str]:
        return {
            "User-Agent": random.choice(USER_AGENTS),
            "Accept": "application/json, text/javascript, */*; q=0.01",
            "Referer": referer,
            "Connection": "keep-alive",
        }

    def get_text(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        referer: str = "https://quote.eastmoney.com/",
        encoding: str | None = None,
    ) -> str:
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = self._session().get(
                    url,
                    params=params,
                    headers=self.headers(referer),
                    timeout=self.timeout,
                )
                response.raise_for_status()
                if encoding:
                    response.encoding = encoding
                return response.text
            except Exception as exc:  # requests exposes several transport exceptions
                last_error = exc
                if attempt + 1 < self.max_retries:
                    time.sleep(0.4 * (attempt + 1) + random.random() * 0.25)
        assert last_error is not None
        raise last_error

    def get_json(self, url: str, params: dict[str, Any], referer: str) -> dict[str, Any]:
        text = self.get_text(url, params=params, referer=referer).strip()
        if text.startswith("{"):
            return json.loads(text)
        start, end = text.find("("), text.rfind(")")
        if start < 0 or end <= start:
            raise ValueError("Response is neither JSON nor JSONP.")
        return json.loads(text[start + 1 : end])
