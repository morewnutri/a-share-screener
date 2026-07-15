from __future__ import annotations

import json
import random
import threading
import time
from datetime import datetime

import numpy as np
import pandas as pd
import requests

from config import CONFIG


_thread_local = threading.local()


def get_session():
    if not hasattr(_thread_local, "session"):
        s = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=50, pool_maxsize=50)
        s.mount("http://", adapter)
        s.mount("https://", adapter)
        _thread_local.session = s
    return _thread_local.session


def safe_float(x, default=np.nan):
    try:
        if x is None:
            return default
        if isinstance(x, str):
            x = x.replace(",", "").replace("%", "").strip()
            if x == "":
                return default
        return float(x)
    except Exception:
        return default


def random_headers(referer="https://quote.eastmoney.com/"):
    return {
        "User-Agent": random.choice(CONFIG.user_agents),
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "Referer": referer,
        "Connection": "keep-alive",
    }


def parse_jsonp(text: str):
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return json.loads(text)
    start = text.find("(")
    end = text.rfind(")")
    if start == -1 or end == -1:
        raise ValueError("无法解析 JSONP")
    return json.loads(text[start + 1:end])


def request_text(url: str, params=None, headers=None, timeout=None, max_retries=None):
    timeout = timeout or CONFIG.request_timeout
    max_retries = CONFIG.max_retries if max_retries is None else max_retries

    last_exc = None
    for i in range(max_retries):
        try:
            session = get_session()
            resp = session.get(url, params=params, headers=headers or random_headers(), timeout=timeout)
            resp.raise_for_status()
            resp.encoding = resp.apparent_encoding or "utf-8"
            return resp.text
        except Exception as e:
            last_exc = e
            time.sleep(0.4 * (i + 1) + random.random() * 0.2)
    raise last_exc


def request_json_or_jsonp(url: str, params=None, headers=None, timeout=None, max_retries=None):
    text = request_text(url, params=params, headers=headers, timeout=timeout, max_retries=max_retries)
    return parse_jsonp(text)


def code_to_secid(code: str) -> str:
    code = str(code).strip().zfill(6)
    if code.startswith(("600", "601", "603", "605", "688", "689")):
        return f"1.{code}"
    return f"0.{code}"


def code_to_sina_symbol(code: str) -> str:
    code = str(code).strip().zfill(6)
    if code.startswith(("600", "601", "603", "605", "688", "689")):
        return f"sh{code}"
    return f"sz{code}"


def probe_eastmoney_hist():
    code = "000001"
    secid = code_to_secid(code)

    for host in CONFIG.eastmoney_push2his_hosts:
        try:
            url = f"{host}/api/qt/stock/kline/get"
            params = {
                "secid": secid,
                "ut": CONFIG.eastmoney_ut,
                "fields1": "f1,f2,f3,f4,f5,f6",
                "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
                "klt": 101,
                "fqt": CONFIG.fqt,
                "beg": CONFIG.start_date,
                "end": CONFIG.end_date,
                "lmt": 5,
                "_": str(int(time.time() * 1000)),
            }
            data = request_json_or_jsonp(url, params=params, headers=random_headers())
            payload = data.get("data")
            klines = payload.get("klines") if payload else None
            if klines:
                return True, host
        except Exception:
            continue
    return False, None


def probe_sina_hist():
    try:
        symbol = "sz000001"
        ts = int(time.time() * 1000)
        url = f"https://quotes.sina.cn/cn/api/jsonp_v2.php/var%20_{symbol}_240_{ts}=/CN_MarketDataService.getKLineData"
        params = {"symbol": symbol, "scale": 240, "ma": "no", "datalen": 5}
        text = request_text(
            url,
            params=params,
            headers={"User-Agent": random.choice(CONFIG.user_agents), "Referer": "https://finance.sina.com.cn"},
            max_retries=1,
        )
        return "(" in text and ")" in text, "sina"
    except Exception:
        return False, None


def eastmoney_get_stock_list():
    errors = []
    for host in CONFIG.eastmoney_push2_hosts:
        try:
            url = f"{host}/api/qt/clist/get"
            page = 1
            page_size = 500
            all_rows = []

            while True:
                params = {
                    "pn": page,
                    "pz": page_size,
                    "po": 1,
                    "np": 1,
                    "ut": CONFIG.eastmoney_ut,
                    "fltt": 2,
                    "invt": 2,
                    "fid": "f12",
                    "fs": CONFIG.eastmoney_fs,
                    "fields": "f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f12,f13,f14,f15,f16,f17,f18,f20,f21,f22,f23,f24,f25,f26",
                    "_": str(int(time.time() * 1000)),
                }
                data = request_json_or_jsonp(
                    url,
                    params=params,
                    headers=random_headers("https://quote.eastmoney.com/center/gridlist.html"),
                    max_retries=2,
                )
                payload = data.get("data") or {}
                diff = payload.get("diff") or []
                total = int(payload.get("total", 0))
                if not diff:
                    break

                for item in diff:
                    code = str(item.get("f12", "")).zfill(6)
                    name = str(item.get("f14", "")).strip()
                    all_rows.append({
                        "code": code,
                        "name": name,
                        "market": item.get("f13"),
                        "latest": safe_float(item.get("f2")),
                        "pct_chg": safe_float(item.get("f3")),
                        "turnover": safe_float(item.get("f8")),
                    })

                if len(all_rows) >= total and total > 0:
                    break
                page += 1
                time.sleep(0.05)

            if all_rows:
                df = pd.DataFrame(all_rows).drop_duplicates(subset=["code"]).sort_values("code").reset_index(drop=True)
                return df, f"eastmoney:{host}"
        except Exception as e:
            errors.append(f"{host} -> {type(e).__name__}: {e}")

    raise RuntimeError("东方财富股票列表全部失败: " + " | ".join(errors))


def sina_fetch_batch_quotes(symbols):
    if not symbols:
        return {}
    url = "https://hq.sinajs.cn/list=" + ",".join(symbols)
    text = request_text(
        url,
        headers={"User-Agent": random.choice(CONFIG.user_agents), "Referer": "https://finance.sina.com.cn"},
        max_retries=2,
    )

    result = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or "hq_str_" not in line:
            continue
        try:
            left, right = line.split("=", 1)
            symbol = left.split("hq_str_")[1]
            raw = right.strip().strip(";").strip('"')
            if not raw:
                continue
            parts = raw.split(",")
            name = parts[0].strip()
            if name:
                result[symbol] = {"name": name}
        except Exception:
            continue
    return result


def sina_scan_mainboard_universe():
    candidates = []
    for prefix in ["000", "001", "002", "003"]:
        for i in range(1000):
            candidates.append(f"{prefix}{i:03d}")
    for prefix in ["600", "601", "603", "605"]:
        for i in range(1000):
            candidates.append(f"{prefix}{i:03d}")

    symbols = [code_to_sina_symbol(c) for c in candidates]
    batch_size = 400
    rows = []
    total_batches = (len(symbols) + batch_size - 1) // batch_size

    for idx in range(total_batches):
        batch = symbols[idx * batch_size:(idx + 1) * batch_size]
        quotes = sina_fetch_batch_quotes(batch)
        for symbol, info in quotes.items():
            code = symbol[2:]
            name = info["name"]
            if name and name != "NULL":
                rows.append({
                    "code": code,
                    "name": name,
                    "market": symbol[:2],
                    "latest": np.nan,
                    "pct_chg": np.nan,
                    "turnover": np.nan,
                })
        time.sleep(0.03)

    if not rows:
        raise RuntimeError("新浪股票池扫描结果为空")

    df = pd.DataFrame(rows).drop_duplicates(subset=["code"]).sort_values("code").reset_index(drop=True)
    return df, "sina_scan"


def eastmoney_get_hist(code: str, beg: str, end: str, fqt: int = 1, lmt: int = 1500):
    errors = []
    secid = code_to_secid(code)

    for host in CONFIG.eastmoney_push2his_hosts:
        try:
            url = f"{host}/api/qt/stock/kline/get"
            params = {
                "secid": secid,
                "ut": CONFIG.eastmoney_ut,
                "fields1": "f1,f2,f3,f4,f5,f6",
                "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
                "klt": 101,
                "fqt": fqt,
                "beg": beg,
                "end": end,
                "lmt": lmt,
                "_": str(int(time.time() * 1000)),
            }
            data = request_json_or_jsonp(
                url,
                params=params,
                headers=random_headers("https://quote.eastmoney.com/"),
                max_retries=1,
            )
            payload = data.get("data")
            if not payload:
                raise RuntimeError("payload为空")
            klines = payload.get("klines") or []
            if not klines:
                return pd.DataFrame()

            rows = []
            for line in klines:
                parts = str(line).split(",")
                if len(parts) < 11:
                    continue
                rows.append({
                    "date": pd.to_datetime(parts[0]),
                    "open": safe_float(parts[1]),
                    "close": safe_float(parts[2]),
                    "high": safe_float(parts[3]),
                    "low": safe_float(parts[4]),
                    "volume": safe_float(parts[5]),
                    "amount": safe_float(parts[6]),
                    "amplitude": safe_float(parts[7]),
                    "pct_chg": safe_float(parts[8]),
                    "chg": safe_float(parts[9]),
                    "turnover": safe_float(parts[10]),
                    "code": str(code).zfill(6),
                })
            return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
        except Exception as e:
            errors.append(f"{host} -> {type(e).__name__}: {e}")

    raise RuntimeError("东方财富历史K线全部失败: " + " | ".join(errors))


def sina_get_hist(code: str, start_date: str, end_date: str, datalen: int = 1500):
    symbol = code_to_sina_symbol(code)
    ts = int(time.time() * 1000)
    url = f"https://quotes.sina.cn/cn/api/jsonp_v2.php/var%20_{symbol}_240_{ts}=/CN_MarketDataService.getKLineData"
    params = {"symbol": symbol, "scale": 240, "ma": "no", "datalen": datalen}

    text = request_text(
        url,
        params=params,
        headers={"User-Agent": random.choice(CONFIG.user_agents), "Referer": "https://finance.sina.com.cn"},
        max_retries=1,
    )

    start = text.find("(")
    end = text.rfind(")")
    if start == -1 or end == -1:
        raise ValueError(f"新浪K线解析失败: {code}")

    raw = text[start + 1:end].strip()
    if raw in ("", "null"):
        return pd.DataFrame()

    data = json.loads(raw)
    if not isinstance(data, list) or not data:
        return pd.DataFrame()

    rows = []
    for item in data:
        rows.append({
            "date": pd.to_datetime(item.get("day")),
            "open": safe_float(item.get("open")),
            "close": safe_float(item.get("close")),
            "high": safe_float(item.get("high")),
            "low": safe_float(item.get("low")),
            "volume": safe_float(item.get("volume")),
            "amount": np.nan,
            "amplitude": np.nan,
            "pct_chg": np.nan,
            "chg": np.nan,
            "turnover": np.nan,
            "code": str(code).zfill(6),
        })

    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    if not df.empty:
        sdt = pd.to_datetime(start_date)
        edt = pd.to_datetime(end_date)
        df = df[(df["date"] >= sdt) & (df["date"] <= edt)].copy().reset_index(drop=True)
        df["chg"] = df["close"].diff()
        prev_close = df["close"].shift(1)
        df["pct_chg"] = np.where(prev_close > 0, (df["close"] / prev_close - 1) * 100, np.nan)
    return df


def eastmoney_get_index_hist(code: str = "000001", beg: str | None = None, end: str | None = None, lmt: int = 1500):
    return eastmoney_get_hist(code=code, beg=beg or CONFIG.start_date, end=end or CONFIG.end_date, fqt=0, lmt=lmt)
