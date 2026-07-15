from __future__ import annotations

import json
import os
from datetime import datetime, timedelta

import pandas as pd

from calendar_utils import is_complete_daily_bar, latest_complete_trade_date


CACHE_SCHEMA_VERSION = 2


def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def is_cache_fresh(path: str, expire_hours: int) -> bool:
    if not os.path.exists(path):
        return False
    mtime = datetime.fromtimestamp(os.path.getmtime(path))
    return datetime.now() - mtime < timedelta(hours=expire_hours)


def atomic_save_csv(df: pd.DataFrame, path: str):
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False, encoding="utf-8-sig")
    os.replace(tmp, path)


def load_csv_if_exists(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


def get_hist_cache_path(hist_cache_dir: str, code: str) -> str:
    return os.path.join(hist_cache_dir, f"{code}.csv")


def get_hist_meta_path(hist_cache_dir: str, code: str) -> str:
    return os.path.join(hist_cache_dir, f"{code}.meta.json")


def read_hist_cache(hist_cache_dir: str, code: str) -> pd.DataFrame:
    path = get_hist_cache_path(hist_cache_dir, code)
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["date"], dtype={"code": str})
    return df.sort_values("date").reset_index(drop=True)


def read_hist_meta(hist_cache_dir: str, code: str) -> dict:
    path = get_hist_meta_path(hist_cache_dir, code)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_hist_cache(hist_cache_dir: str, code: str, df: pd.DataFrame, meta: dict):
    ensure_dir(hist_cache_dir)
    csv_path = get_hist_cache_path(hist_cache_dir, code)
    meta_path = get_hist_meta_path(hist_cache_dir, code)
    atomic_save_csv(df, csv_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def need_refresh_hist(hist_cache_dir: str, code: str, *, force_refresh: bool, expire_hours: int, expected_trade_date, fqt: int, allowed_source: str = "eastmoney") -> bool:
    path = get_hist_cache_path(hist_cache_dir, code)
    meta = read_hist_meta(hist_cache_dir, code)

    if force_refresh:
        return True
    if not os.path.exists(path):
        return True
    if not is_cache_fresh(path, expire_hours):
        return True
    if not meta:
        return True
    if meta.get("schema_version") != CACHE_SCHEMA_VERSION:
        return True
    if meta.get("fqt") != fqt:
        return True
    if meta.get("source") != allowed_source:
        return True

    try:
        df = pd.read_csv(path, usecols=["date"])
        if df.empty:
            return True
        last_date = pd.to_datetime(df["date"]).max().date()
        return last_date < expected_trade_date
    except Exception:
        return True


def is_hist_stale(df: pd.DataFrame, now_dt: datetime, market_close_time: str = "15:10") -> bool:
    if df is None or df.empty:
        return True
    last_date = pd.to_datetime(df["date"]).max().date()
    return not is_complete_daily_bar(last_date, now_dt, market_close_time)


def build_hist_meta(code: str, source: str, fqt: int, df: pd.DataFrame) -> dict:
    last_date = None
    if df is not None and not df.empty:
        last_date = pd.to_datetime(df["date"]).max().strftime("%Y-%m-%d")
    return {
        "code": code,
        "source": source,
        "fqt": fqt,
        "schema_version": CACHE_SCHEMA_VERSION,
        "last_complete_date": last_date,
        "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
