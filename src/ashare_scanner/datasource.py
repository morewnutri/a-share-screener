from __future__ import annotations

import time
from datetime import date
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
EASTMONEY_FS = "m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23,m:0+t:81+s:2048"
MAINBOARD_PREFIXES = ("600", "601", "603", "605", "000", "001", "002", "003")


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


def is_mainboard_code(code: str) -> bool:
    return str(code).zfill(6).startswith(MAINBOARD_PREFIXES)


class EastmoneyDataSource:
    """Eastmoney-only strategy data source to keep adjustment semantics consistent."""

    def __init__(self, http: HttpClient, fqt: int = 1) -> None:
        if fqt != 1:
            raise ValueError("Only forward-adjusted stock bars (fqt=1) are supported.")
        self.http = http
        self.fqt = fqt

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

    def fetch_stock_history(self, code: str, start: str, end: date) -> tuple[pd.DataFrame, str]:
        return self._fetch_history(code_to_secid(code), code, start, end, self.fqt)

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
