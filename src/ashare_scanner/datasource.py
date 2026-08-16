from __future__ import annotations

import threading
import time
from datetime import date
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd

from .http import HttpClient


EASTMONEY_UT = "bd1d9ddb04089700cf9c27f6f7426281"
PUSH2_HOSTS = (
    "https://push2.eastmoney.com",
    "https://7.push2.eastmoney.com",
    "https://19.push2.eastmoney.com",
    "https://80.push2.eastmoney.com",
)
PUSH2HIS_HOSTS = (
    "https://push2his.eastmoney.com",
    "https://7.push2his.eastmoney.com",
    "https://19.push2his.eastmoney.com",
    "https://63.push2his.eastmoney.com",
    "https://80.push2his.eastmoney.com",
)
TENCENT_KLINE_HOSTS = (
    "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get",
    "https://proxy.finance.qq.com/ifzqgtimg/appstock/app/newfqkline/get",
)
EASTMONEY_FS = "m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23,m:0+t:81+s:2048"
MAINBOARD_PREFIXES = ("600", "601", "603", "605", "000", "001", "002", "003")
HISTORY_PROVIDER_FAILURE_LIMIT = 12
HISTORY_PROVIDER_COOLDOWN_SECONDS = 30.0
BAOSTOCK_STOCK_FIELDS = (
    "date,code,open,high,low,close,preclose,volume,amount,turn,"
    "tradestatus,pctChg,isST"
)
BAOSTOCK_INDEX_FIELDS = "date,code,open,high,low,close,preclose,volume,amount,pctChg"


def safe_float(value: Any) -> float:
    try:
        if value is None or value == "-":
            return np.nan
        return float(str(value).replace(",", "").replace("%", "").strip())
    except (TypeError, ValueError):
        return np.nan


def code_to_secid(code: str) -> str:
    code = str(code).zfill(6)
    market = "1" if code.startswith(("600", "601", "603", "605")) else "0"
    return f"{market}.{code}"


def code_to_tencent_symbol(code: str) -> str:
    code = str(code).zfill(6)
    market = "sh" if code.startswith(("600", "601", "603", "605")) else "sz"
    return f"{market}{code}"


def is_mainboard_code(code: str) -> bool:
    return str(code).zfill(6).startswith(MAINBOARD_PREFIXES)


class EastmoneyDataSource:
    """Public market data with Tencent history and Eastmoney fallbacks."""

    def __init__(self, http: HttpClient, fqt: int = 1) -> None:
        if fqt != 1:
            raise ValueError("Only forward-adjusted stock bars (fqt=1) are supported.")
        self.http = http
        self.fqt = fqt
        self._history_health_lock = threading.Lock()
        self._history_failure_streak = {"tencent": 0, "eastmoney": 0}
        self._history_cooldown_until = {"tencent": 0.0, "eastmoney": 0.0}

    def _history_provider_available(self, provider: str) -> bool:
        with self._history_health_lock:
            return time.monotonic() >= self._history_cooldown_until[provider]

    def _record_history_result(self, provider: str, succeeded: bool) -> None:
        with self._history_health_lock:
            if succeeded:
                self._history_failure_streak[provider] = 0
                self._history_cooldown_until[provider] = 0.0
                return
            self._history_failure_streak[provider] += 1
            if self._history_failure_streak[provider] >= HISTORY_PROVIDER_FAILURE_LIMIT:
                self._history_failure_streak[provider] = 0
                self._history_cooldown_until[provider] = (
                    time.monotonic() + HISTORY_PROVIDER_COOLDOWN_SECONDS
                )

    def fetch_universe(self, min_size: int = 2000) -> tuple[pd.DataFrame, str]:
        errors: list[str] = []
        snapshot_responded = False

        # A single stable snapshot avoids page drift and is also less likely to
        # trigger rate limits. Some edge nodes cap pz, so paged failover remains.
        for host in PUSH2_HOSTS:
            try:
                diff, total = self._request_universe_page(host, page=1, page_size=10_000)
                snapshot_responded = True
                if total and len(diff) >= total * 0.98:
                    frame = self._universe_frame(diff)
                    return self._filter_universe(frame, min_size), f"eastmoney:{host}:snapshot"
                errors.append(f"snapshot {host}: partial response {len(diff)}/{total}")
            except Exception as exc:
                errors.append(f"snapshot {host}: {type(exc).__name__}: {exc}")

        if snapshot_responded:
            try:
                frame, hosts_used = self._fetch_universe_paged()
                return self._filter_universe(frame, min_size), "eastmoney:paged:" + ",".join(hosts_used)
            except Exception as exc:
                errors.append(f"paged: {type(exc).__name__}: {exc}")

        try:
            frame = self._sina_scan_mainboard()
            return self._filter_universe(frame, min_size), "sina:batch_quote_scan"
        except Exception as exc:
            errors.append(f"sina scan: {type(exc).__name__}: {exc}")
            raise RuntimeError("All universe policies failed: " + " | ".join(errors)) from exc

    def _fetch_universe_paged(self) -> tuple[pd.DataFrame, list[str]]:
        page, page_size, total = 1, 500, None
        all_items: list[dict[str, Any]] = []
        seen_pages: set[tuple[str, ...]] = set()
        hosts_used: list[str] = []
        while total is None or len(all_items) < total:
            page_errors: list[str] = []
            diff: list[dict[str, Any]] = []
            for offset in range(len(PUSH2_HOSTS)):
                host = PUSH2_HOSTS[(page - 1 + offset) % len(PUSH2_HOSTS)]
                try:
                    diff, page_total = self._request_universe_page(host, page, page_size)
                    total = page_total
                    hosts_used.append(host)
                    break
                except Exception as exc:
                    page_errors.append(f"{host}: {type(exc).__name__}: {exc}")
            if not diff:
                raise RuntimeError(f"page {page} failed: " + " | ".join(page_errors))
            page_codes = tuple(str(item.get("f12", "")).zfill(6) for item in diff)
            if page_codes in seen_pages:
                raise RuntimeError("universe pagination repeated a page")
            seen_pages.add(page_codes)
            all_items.extend(diff)
            page += 1
            time.sleep(0.12)
        frame = self._universe_frame(all_items)
        if not len(frame):
            raise RuntimeError("empty universe response")
        unique = frame["code"].nunique()
        if total and unique < total * 0.98:
            raise RuntimeError(f"unstable pagination coverage: unique={unique}, total={total}")
        return frame, sorted(set(hosts_used))

    def _request_universe_page(
        self,
        host: str,
        page: int,
        page_size: int,
    ) -> tuple[list[dict[str, Any]], int]:
        params = {
            "pn": page,
            "pz": page_size,
            "po": 1,
            "np": 1,
            "ut": EASTMONEY_UT,
            "fltt": 2,
            "invt": 2,
            "fid": "f12",
            "fs": EASTMONEY_FS,
            "fields": "f2,f3,f5,f6,f8,f12,f13,f14,f20,f21,f62,f184",
            "_": str(int(time.time() * 1000)),
        }
        payload = self.http.get_json(
            f"{host}/api/qt/clist/get",
            params,
            "https://quote.eastmoney.com/center/gridlist.html",
        ).get("data") or {}
        return payload.get("diff") or [], int(payload.get("total") or 0)

    @staticmethod
    def _universe_frame(items: list[dict[str, Any]]) -> pd.DataFrame:
        rows = [
            {
                "code": str(item.get("f12", "")).zfill(6),
                "name": str(item.get("f14", "")).strip(),
                "market": item.get("f13"),
                "latest": safe_float(item.get("f2")),
                "pct_chg": safe_float(item.get("f3")),
                "volume": safe_float(item.get("f5")),
                "amount": safe_float(item.get("f6")),
                "turnover": safe_float(item.get("f8")),
                "market_cap": safe_float(item.get("f20")),
                "float_market_cap": safe_float(item.get("f21")),
                "main_net_inflow_amount": safe_float(item.get("f62")),
                "main_net_inflow_ratio_pct": safe_float(item.get("f184")),
            }
            for item in items
        ]
        return pd.DataFrame(rows)

    @staticmethod
    def _filter_universe(frame: pd.DataFrame, min_size: int) -> pd.DataFrame:
        frame = frame[frame["code"].map(is_mainboard_code)].copy()
        frame["is_st"] = frame["name"].str.upper().str.contains("ST", na=False)
        frame["is_delisting"] = frame["name"].str.contains("\u9000", regex=False, na=False)
        frame = frame[~frame["is_st"] & ~frame["is_delisting"]]
        frame = frame.drop_duplicates("code").sort_values("code").reset_index(drop=True)
        if len(frame) < min_size:
            raise RuntimeError(f"main-board universe too small: {len(frame)} < {min_size}")
        return frame

    def _sina_scan_mainboard(self) -> pd.DataFrame:
        codes = [
            f"{prefix}{suffix:03d}"
            for prefix in ("000", "001", "002", "003", "600", "601", "603", "605")
            for suffix in range(1000)
        ]
        symbols = [
            ("sh" if code.startswith(("600", "601", "603", "605")) else "sz") + code
            for code in codes
        ]
        rows: list[dict[str, Any]] = []
        failures: list[str] = []
        batch_size = 300
        for start in range(0, len(symbols), batch_size):
            batch = symbols[start : start + batch_size]
            try:
                text = self.http.get_text(
                    "https://hq.sinajs.cn/list=" + ",".join(batch),
                    referer="https://finance.sina.com.cn/",
                    encoding="gbk",
                )
                rows.extend(self._parse_sina_quotes(text))
            except Exception as exc:
                failures.append(f"batch {start // batch_size + 1}: {type(exc).__name__}: {exc}")
            time.sleep(0.05)
        if not rows:
            raise RuntimeError("all Sina quote batches were empty: " + " | ".join(failures[:5]))
        return pd.DataFrame(rows).drop_duplicates("code").reset_index(drop=True)

    @staticmethod
    def _parse_sina_quotes(text: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for line in text.splitlines():
            if "hq_str_" not in line or "=" not in line:
                continue
            left, right = line.split("=", 1)
            symbol = left.rsplit("hq_str_", 1)[-1].strip()
            values = right.strip().strip(";").strip('"').split(",")
            name = values[0].strip() if values else ""
            if len(symbol) != 8 or not name or name == "NULL":
                continue
            rows.append(
                {
                    "code": symbol[2:],
                    "name": name,
                    "market": symbol[:2],
                    "latest": safe_float(values[3]) if len(values) > 3 else np.nan,
                    "pct_chg": np.nan,
                    "volume": safe_float(values[8]) if len(values) > 8 else np.nan,
                    "amount": safe_float(values[9]) if len(values) > 9 else np.nan,
                    "turnover": np.nan,
                    "market_cap": np.nan,
                    "float_market_cap": np.nan,
                    "main_net_inflow_amount": np.nan,
                    "main_net_inflow_ratio_pct": np.nan,
                }
            )
        return rows

    def fetch_stock_history(
        self,
        code: str,
        start: str,
        end: date,
        *,
        latest_price: float = np.nan,
        float_market_cap: float = np.nan,
        turnover_hint: float = np.nan,
    ) -> tuple[pd.DataFrame, str]:
        errors: list[str] = []
        if self._history_provider_available("tencent"):
            try:
                result = self._fetch_tencent_history(
                    code,
                    start,
                    end,
                    latest_price=latest_price,
                    float_market_cap=float_market_cap,
                    turnover_hint=turnover_hint,
                )
                self._record_history_result("tencent", True)
                return result
            except Exception as exc:
                self._record_history_result("tencent", False)
                errors.append(f"tencent: {type(exc).__name__}: {exc}")
        else:
            errors.append("tencent: cooling down after consecutive failures")
        if self._history_provider_available("eastmoney"):
            try:
                result = self._fetch_history(code_to_secid(code), code, start, end, self.fqt)
                self._record_history_result("eastmoney", True)
                return result
            except Exception as exc:
                self._record_history_result("eastmoney", False)
                errors.append(f"eastmoney: {type(exc).__name__}: {exc}")
        else:
            errors.append("eastmoney: cooling down after consecutive failures")
        raise RuntimeError(f"all history sources failed for {code}: " + " | ".join(errors))

    def _fetch_tencent_history(
        self,
        code: str,
        start: str,
        end: date,
        *,
        latest_price: float,
        float_market_cap: float,
        turnover_hint: float,
    ) -> tuple[pd.DataFrame, str]:
        symbol = code_to_tencent_symbol(code)
        start_date = date.fromisoformat(start)
        estimated_sessions = int((end - start_date).days * 5 / 7) + 60
        count = int(np.clip(estimated_sessions, 400, 1500))
        errors: list[str] = []
        for host in TENCENT_KLINE_HOSTS:
            try:
                payload = self.http.get_json(
                    host,
                    {"param": f"{symbol},day,{start},{end.isoformat()},{count},qfq"},
                    f"https://gu.qq.com/{symbol}",
                )
                node = (payload.get("data") or {}).get(symbol) or {}
                klines = node.get("qfqday") or node.get("day") or []
                quote_node = node.get("qt") or {}
                quote = quote_node.get(symbol) if isinstance(quote_node, dict) else quote_node
                quote = quote or []
                frame, turnover_source = self._parse_tencent_klines(
                    klines,
                    code,
                    quote=quote,
                    latest_price=latest_price,
                    float_market_cap=float_market_cap,
                    turnover_hint=turnover_hint,
                )
                if frame.empty:
                    raise RuntimeError("empty Tencent history")
                return frame, f"tencent:{host}:{turnover_source}"
            except Exception as exc:
                errors.append(f"{host}: {type(exc).__name__}: {exc}")
        raise RuntimeError(" | ".join(errors))

    @staticmethod
    def _parse_tencent_klines(
        klines: list[list[Any]],
        code: str,
        *,
        quote: list[Any] | None = None,
        latest_price: float = np.nan,
        float_market_cap: float = np.nan,
        turnover_hint: float = np.nan,
    ) -> tuple[pd.DataFrame, str]:
        rows = [list(row[:6]) for row in klines if isinstance(row, list) and len(row) >= 6]
        columns = ("date", "open", "close", "high", "low", "volume")
        if not rows:
            return pd.DataFrame(columns=(*columns, "amount", "turnover", "code")), "missing"
        frame = pd.DataFrame(rows, columns=columns)
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        for column in columns[1:]:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame = frame.dropna(subset=list(columns)).drop_duplicates("date", keep="last")
        frame = frame.sort_values("date").reset_index(drop=True)

        float_volume_units = np.nan
        turnover_source = ""
        quote = quote or []
        quote_volume = safe_float(quote[36]) if len(quote) > 38 else np.nan
        quote_turnover = safe_float(quote[38]) if len(quote) > 38 else np.nan
        if np.isfinite(quote_volume) and quote_volume > 0 and np.isfinite(quote_turnover) and quote_turnover > 0:
            float_volume_units = quote_volume * 100 / quote_turnover
            turnover_source = "turnover_from_quote"
        elif (
            np.isfinite(float_market_cap)
            and float_market_cap > 0
            and np.isfinite(latest_price)
            and latest_price > 0
        ):
            # Tencent stock volume is in lots; one lot is 100 shares.
            float_volume_units = float_market_cap / latest_price / 100
            turnover_source = "turnover_from_float_market_cap"
        elif np.isfinite(turnover_hint) and turnover_hint > 0 and frame.iloc[-1]["volume"] > 0:
            float_volume_units = frame.iloc[-1]["volume"] * 100 / turnover_hint
            turnover_source = "turnover_from_snapshot_hint"
        if not np.isfinite(float_volume_units) or float_volume_units <= 0:
            raise RuntimeError("Tencent history has no usable turnover basis")

        frame["turnover"] = frame["volume"] / float_volume_units * 100
        typical_price = (frame["open"] + frame["high"] + frame["low"] + frame["close"]) / 4
        frame["amount"] = frame["volume"] * 100 * typical_price
        previous_close = frame["close"].shift(1)
        frame["pct_chg"] = frame["close"].pct_change(fill_method=None) * 100
        frame["chg"] = frame["close"].diff()
        frame["amplitude"] = (frame["high"] - frame["low"]) / previous_close * 100
        frame["code"] = str(code).zfill(6)
        return frame, turnover_source

    def fetch_stock_fund_flow(
        self,
        code: str,
        limit: int = 100,
    ) -> tuple[pd.DataFrame, str]:
        errors: list[str] = []
        for host in PUSH2HIS_HOSTS:
            try:
                params = {
                    "lmt": limit,
                    "klt": 101,
                    "secid": code_to_secid(code),
                    "fields1": "f1,f2,f3,f7",
                    "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63,f64,f65",
                    "ut": "b2884a393a59ad64002292a3e90d46a5",
                    "_": str(int(time.time() * 1000)),
                }
                data = self.http.get_json(
                    f"{host}/api/qt/stock/fflow/daykline/get",
                    params,
                    "https://data.eastmoney.com/zjlx/detail.html",
                ).get("data")
                if not data:
                    raise RuntimeError("empty fund-flow payload")
                frame = self._parse_fund_flow(data.get("klines") or [], code)
                if frame.empty:
                    raise RuntimeError("empty fund-flow history")
                return frame, f"eastmoney:{host}:fund_flow"
            except Exception as exc:
                errors.append(f"{host}: {type(exc).__name__}: {exc}")
        raise RuntimeError(f"fund flow failed for {code}: " + " | ".join(errors))

    def fetch_benchmark_history(
        self,
        secid: str,
        start: str,
        end: date,
    ) -> tuple[pd.DataFrame, str]:
        return self._fetch_history(secid, secid.replace(".", "_"), start, end, 0)

    def _fetch_history(
        self,
        secid: str,
        code: str,
        start: str,
        end: date,
        fqt: int,
    ) -> tuple[pd.DataFrame, str]:
        errors: list[str] = []
        for host in PUSH2HIS_HOSTS:
            try:
                params = {
                    "secid": secid,
                    "ut": EASTMONEY_UT,
                    "fields1": "f1,f2,f3,f4,f5,f6",
                    "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
                    "klt": 101,
                    "fqt": fqt,
                    "beg": start.replace("-", ""),
                    "end": end.strftime("%Y%m%d"),
                    "lmt": 5000,
                    "_": str(int(time.time() * 1000)),
                }
                data = self.http.get_json(
                    f"{host}/api/qt/stock/kline/get",
                    params,
                    "https://quote.eastmoney.com/",
                ).get("data")
                if not data:
                    raise RuntimeError("empty history payload")
                frame = self._parse_klines(data.get("klines") or [], code)
                return frame, f"eastmoney:{host}"
            except Exception as exc:
                errors.append(f"{host}: {type(exc).__name__}: {exc}")
        raise RuntimeError(f"history failed for {code}: " + " | ".join(errors))

    @staticmethod
    def _parse_klines(klines: list[str], code: str) -> pd.DataFrame:
        columns = (
            "date",
            "open",
            "close",
            "high",
            "low",
            "volume",
            "amount",
            "amplitude",
            "pct_chg",
            "chg",
            "turnover",
        )
        rows = []
        for line in klines:
            parts = str(line).split(",")
            if len(parts) >= len(columns):
                rows.append(parts[: len(columns)])
        if not rows:
            return pd.DataFrame(columns=(*columns, "code"))
        frame = pd.DataFrame(rows, columns=columns)
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        for column in columns[1:]:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame["code"] = str(code).zfill(6)
        frame = frame.dropna(subset=["date", "open", "high", "low", "close", "volume"])
        frame = frame.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)
        return frame

    @staticmethod
    def _parse_fund_flow(klines: list[str], code: str) -> pd.DataFrame:
        columns = (
            "date",
            "main_net_inflow_amount",
            "small_net_inflow_amount",
            "medium_net_inflow_amount",
            "large_net_inflow_amount",
            "super_large_net_inflow_amount",
            "main_net_inflow_ratio_pct",
            "small_net_inflow_ratio_pct",
            "medium_net_inflow_ratio_pct",
            "large_net_inflow_ratio_pct",
            "super_large_net_inflow_ratio_pct",
            "close",
            "pct_chg",
            "unused_1",
            "unused_2",
        )
        rows = []
        for line in klines:
            parts = str(line).split(",")
            if len(parts) >= len(columns):
                rows.append(parts[: len(columns)])
        if not rows:
            return pd.DataFrame(columns=(*columns, "code"))
        frame = pd.DataFrame(rows, columns=columns)
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        for column in columns[1:]:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame["code"] = str(code).zfill(6)
        return frame.dropna(subset=["date", "main_net_inflow_amount"]).sort_values("date").reset_index(drop=True)


def code_to_baostock_symbol(code: str) -> str:
    code = str(code).zfill(6)
    market = "sh" if code.startswith(("600", "601", "603", "605")) else "sz"
    return f"{market}.{code}"


def secid_to_baostock_symbol(secid: str) -> str:
    market, code = str(secid).split(".", 1)
    return f"{'sh' if market == '1' else 'sz'}.{code.zfill(6)}"


class BaoStockDataSource:
    """Thread-safe adapter around BaoStock's process-global socket client."""

    def __init__(self, module: ModuleType | Any | None = None, max_retries: int = 2) -> None:
        self._lock = threading.RLock()
        self._logged_in = False
        self._failure_streak = 0
        self._cooldown_until = 0.0
        self.max_retries = max(1, int(max_retries))
        self.import_error = ""
        if module is not None:
            self.bs = module
            return
        try:
            import baostock as bs

            self.bs = bs
        except ImportError as exc:
            self.bs = None
            self.import_error = f"{type(exc).__name__}: {exc}"

    @property
    def available(self) -> bool:
        return self.bs is not None

    def _login(self) -> None:
        if not self.available:
            raise RuntimeError(f"BaoStock is not installed: {self.import_error}")
        if self._logged_in:
            return
        response = self.bs.login()
        if str(response.error_code) != "0":
            raise RuntimeError(
                f"BaoStock login failed [{response.error_code}]: {response.error_msg}"
            )
        self._logged_in = True

    def _reset(self) -> None:
        if self.available and self._logged_in:
            try:
                self.bs.logout()
            except Exception:
                pass
        self._logged_in = False

    @staticmethod
    def _result_frame(result: Any, operation: str) -> pd.DataFrame:
        if str(result.error_code) != "0":
            raise RuntimeError(
                f"BaoStock {operation} failed [{result.error_code}]: {result.error_msg}"
            )
        if hasattr(result, "get_data"):
            frame = result.get_data()
            return frame.copy() if isinstance(frame, pd.DataFrame) else pd.DataFrame(frame)
        rows: list[list[Any]] = []
        while result.next():
            rows.append(result.get_row_data())
        return pd.DataFrame(rows, columns=result.fields)

    def _query(self, operation: str, callback: Any) -> pd.DataFrame:
        errors: list[str] = []
        with self._lock:
            if time.monotonic() < self._cooldown_until:
                raise RuntimeError("BaoStock is cooling down after repeated connection failures")
            for attempt in range(self.max_retries):
                try:
                    self._login()
                    frame = self._result_frame(callback(), operation)
                    self._failure_streak = 0
                    self._cooldown_until = 0.0
                    return frame
                except Exception as exc:
                    errors.append(f"attempt {attempt + 1}: {type(exc).__name__}: {exc}")
                    self._reset()
                    if attempt + 1 < self.max_retries:
                        time.sleep(0.5 * (attempt + 1))
            self._failure_streak += 1
            if self._failure_streak >= 3:
                self._failure_streak = 0
                self._cooldown_until = time.monotonic() + 60.0
        raise RuntimeError(" | ".join(errors))

    def fetch_universe(self, expected: date, min_size: int = 2000) -> tuple[pd.DataFrame, str]:
        frame = self._query(
            "query_all_stock",
            lambda: self.bs.query_all_stock(expected.isoformat()),
        )
        if frame.empty or "code" not in frame:
            raise RuntimeError("BaoStock returned an empty stock universe")
        names = frame.get("code_name", pd.Series("", index=frame.index)).astype(str)
        symbols = frame["code"].astype(str)
        trade_status = pd.to_numeric(
            frame.get("tradeStatus", pd.Series(1, index=frame.index)), errors="coerce"
        ).fillna(0)
        universe = pd.DataFrame(
            {
                "code": symbols.str.rsplit(".", n=1).str[-1].str.zfill(6),
                "name": names,
                "market": symbols.str.split(".", n=1).str[0],
                "trade_status": trade_status,
            }
        )
        exchange_is_stock = (
            symbols.str.startswith("sh.")
            & universe["code"].str.startswith(MAINBOARD_PREFIXES[:4])
        ) | (
            symbols.str.startswith("sz.")
            & universe["code"].str.startswith(MAINBOARD_PREFIXES[4:])
        )
        universe = universe[exchange_is_stock].copy()
        for column in (
            "latest",
            "pct_chg",
            "volume",
            "amount",
            "turnover",
            "market_cap",
            "float_market_cap",
            "main_net_inflow_amount",
            "main_net_inflow_ratio_pct",
        ):
            universe[column] = np.nan
        universe = EastmoneyDataSource._filter_universe(universe, min_size)
        return universe, "baostock:query_all_stock"

    def fetch_stock_history(
        self,
        code: str,
        start: str,
        end: date,
        **_: Any,
    ) -> tuple[pd.DataFrame, str]:
        symbol = code_to_baostock_symbol(code)
        frame = self._query(
            f"history {symbol}",
            lambda: self.bs.query_history_k_data_plus(
                symbol,
                BAOSTOCK_STOCK_FIELDS,
                start_date=start,
                end_date=end.isoformat(),
                frequency="d",
                adjustflag="2",
            ),
        )
        return self._normalize_history(frame, code, stock=True), "baostock:qfq"

    def fetch_benchmark_history(
        self,
        secid: str,
        start: str,
        end: date,
    ) -> tuple[pd.DataFrame, str]:
        symbol = secid_to_baostock_symbol(secid)
        frame = self._query(
            f"index history {symbol}",
            lambda: self.bs.query_history_k_data_plus(
                symbol,
                BAOSTOCK_INDEX_FIELDS,
                start_date=start,
                end_date=end.isoformat(),
                frequency="d",
                adjustflag="3",
            ),
        )
        code = secid.replace(".", "_")
        return self._normalize_history(frame, code, stock=False), "baostock:raw:index"

    @staticmethod
    def _normalize_history(frame: pd.DataFrame, code: str, stock: bool) -> pd.DataFrame:
        columns = (
            "date",
            "open",
            "close",
            "high",
            "low",
            "volume",
            "amount",
            "amplitude",
            "pct_chg",
            "chg",
            "turnover",
            "code",
        )
        if frame.empty:
            return pd.DataFrame(columns=columns)
        result = frame.copy()
        result["date"] = pd.to_datetime(result.get("date"), errors="coerce")
        numeric = ("open", "high", "low", "close", "preclose", "volume", "amount", "turn", "pctChg")
        for column in numeric:
            if column not in result:
                result[column] = np.nan
            result[column] = pd.to_numeric(result[column], errors="coerce")
        if stock and "tradestatus" in result:
            result = result[result["tradestatus"].astype(str) == "1"].copy()
        result["pct_chg"] = result["pctChg"]
        result["chg"] = result["close"] - result["preclose"]
        result["amplitude"] = (
            (result["high"] - result["low"]) / result["preclose"].replace(0, np.nan) * 100
        )
        result["turnover"] = result["turn"] if stock else np.nan
        result["code"] = str(code).zfill(6) if stock else str(code)
        result = result.dropna(subset=["date", "open", "high", "low", "close", "volume"])
        return (
            result[list(columns)]
            .drop_duplicates("date", keep="last")
            .sort_values("date")
            .reset_index(drop=True)
        )


class HybridDataSource:
    """BaoStock-first daily bars with public-web fallbacks and fund flow."""

    def __init__(
        self,
        web: EastmoneyDataSource,
        baostock: BaoStockDataSource | None = None,
    ) -> None:
        self.web = web
        self.baostock = baostock or BaoStockDataSource()

    def fetch_universe(self, expected: date, min_size: int = 2000) -> tuple[pd.DataFrame, str]:
        errors: list[str] = []
        try:
            return self.web.fetch_universe(min_size)
        except Exception as exc:
            errors.append(f"web: {type(exc).__name__}: {exc}")
        try:
            return self.baostock.fetch_universe(expected, min_size)
        except Exception as exc:
            errors.append(f"baostock: {type(exc).__name__}: {exc}")
        raise RuntimeError("all universe sources failed: " + " | ".join(errors))

    def fetch_stock_history(self, code: str, start: str, end: date, **kwargs: Any) -> tuple[pd.DataFrame, str]:
        errors: list[str] = []
        if self.baostock.available:
            try:
                return self.baostock.fetch_stock_history(code, start, end, **kwargs)
            except Exception as exc:
                errors.append(f"baostock: {type(exc).__name__}: {exc}")
        try:
            return self.web.fetch_stock_history(code, start, end, **kwargs)
        except Exception as exc:
            errors.append(f"web: {type(exc).__name__}: {exc}")
        raise RuntimeError("all stock-history sources failed: " + " | ".join(errors))

    def fetch_benchmark_history(self, secid: str, start: str, end: date) -> tuple[pd.DataFrame, str]:
        errors: list[str] = []
        if self.baostock.available:
            try:
                return self.baostock.fetch_benchmark_history(secid, start, end)
            except Exception as exc:
                errors.append(f"baostock: {type(exc).__name__}: {exc}")
        try:
            return self.web.fetch_benchmark_history(secid, start, end)
        except Exception as exc:
            errors.append(f"web: {type(exc).__name__}: {exc}")
        raise RuntimeError("all benchmark sources failed: " + " | ".join(errors))

    def fetch_stock_fund_flow(self, code: str, limit: int = 100) -> tuple[pd.DataFrame, str]:
        return self.web.fetch_stock_fund_flow(code, limit)
