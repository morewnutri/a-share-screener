from __future__ import annotations

import json
import os
from datetime import datetime

import pandas as pd

from cache_manager import atomic_save_csv, ensure_dir, is_cache_fresh
from config import CONFIG
from data_sources import eastmoney_get_stock_list, sina_scan_mainboard_universe


def is_st_name(name: str) -> bool:
    if not isinstance(name, str):
        return False
    u = name.upper().strip()
    return ("ST" in u) or ("*ST" in u)


def is_mainboard_code(code: str) -> bool:
    code = str(code).strip().zfill(6)
    return code.startswith(CONFIG.mainboard_prefixes)


def get_stock_list_with_cache():
    ensure_dir(CONFIG.list_cache_dir)
    cache_path = os.path.join(CONFIG.list_cache_dir, "stock_list.csv")
    meta_path = os.path.join(CONFIG.list_cache_dir, "stock_list_meta.json")

    if (not CONFIG.force_refresh_list) and is_cache_fresh(cache_path, CONFIG.list_cache_expire_hours):
        df = pd.read_csv(cache_path, dtype={"code": str})
        source = "cache"
        if os.path.exists(meta_path):
            try:
                meta = json.load(open(meta_path, "r", encoding="utf-8"))
                source = meta.get("source", "cache")
            except Exception:
                pass
        return df, source + ":cached"

    try:
        df, source = eastmoney_get_stock_list()
    except Exception as e1:
        print(f"东方财富股票列表失败，尝试新浪兜底: {e1}")
        df, source = sina_scan_mainboard_universe()

    atomic_save_csv(df, cache_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"source": source, "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}, f, ensure_ascii=False, indent=2)
    return df, source


def build_universe(raw_df: pd.DataFrame):
    df = raw_df.copy()
    df["code"] = df["code"].astype(str).str.zfill(6)
    df["name"] = df["name"].astype(str).str.strip()

    df["is_mainboard"] = df["code"].apply(is_mainboard_code)
    df["is_st"] = df["name"].apply(is_st_name)
    df["exclude_reason"] = ""

    df.loc[~df["is_mainboard"], "exclude_reason"] = "not_mainboard"
    if CONFIG.exclude_st:
        df.loc[df["is_mainboard"] & df["is_st"], "exclude_reason"] = "ST"

    universe = df[df["is_mainboard"]].copy()
    if CONFIG.exclude_st:
        universe = universe[~universe["is_st"]].copy()

    universe = universe.sort_values("code").reset_index(drop=True)
    df = df.sort_values("code").reset_index(drop=True)
    return df, universe
