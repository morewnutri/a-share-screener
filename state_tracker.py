from __future__ import annotations

import os
from datetime import datetime, timedelta

import pandas as pd

from cache_manager import ensure_dir
from config import CONFIG


STATE_COLUMNS = [
    "code", "name", "state", "setup_date", "trigger_price", "support_price", "expire_date", "last_seen_date"
]


def state_file_path() -> str:
    ensure_dir(CONFIG.state_dir)
    return os.path.join(CONFIG.state_dir, "state_snapshot.csv")


def load_state() -> pd.DataFrame:
    path = state_file_path()
    if not os.path.exists(path):
        return pd.DataFrame(columns=STATE_COLUMNS)
    df = pd.read_csv(path, dtype={"code": str})
    for col in STATE_COLUMNS:
        if col not in df.columns:
            df[col] = None
    return df[STATE_COLUMNS].copy()


def save_state(df: pd.DataFrame):
    path = state_file_path()
    df.to_csv(path, index=False, encoding="utf-8-sig")


def update_state(setup_all: pd.DataFrame, breakout_all: pd.DataFrame, retest_all: pd.DataFrame, expected_trade_date):
    state_df = load_state()
    existing = {row["code"]: row for _, row in state_df.iterrows()}
    snapshot_rows = []

    for _, row in setup_all.iterrows():
        code = row["code"]
        old = existing.get(code, {})
        snapshot_rows.append({
            "code": code,
            "name": row["name"],
            "state": "SETUP",
            "setup_date": old.get("setup_date") or str(expected_trade_date),
            "trigger_price": row.get("prior_high_20"),
            "support_price": row.get("ma20"),
            "expire_date": str(expected_trade_date + timedelta(days=CONFIG.setup_expire_days)),
            "last_seen_date": row.get("date"),
        })

    for _, row in breakout_all.iterrows():
        code = row["code"]
        snapshot_rows.append({
            "code": code,
            "name": row["name"],
            "state": "CONFIRMED_BREAKOUT",
            "setup_date": str(expected_trade_date),
            "trigger_price": row.get("prior_high_20"),
            "support_price": row.get("ma20"),
            "expire_date": str(expected_trade_date + timedelta(days=CONFIG.setup_expire_days)),
            "last_seen_date": row.get("date"),
        })

    for _, row in retest_all.iterrows():
        code = row["code"]
        snapshot_rows.append({
            "code": code,
            "name": row["name"],
            "state": "RETEST",
            "setup_date": str(expected_trade_date),
            "trigger_price": row.get("prior_high_20"),
            "support_price": row.get("ma20"),
            "expire_date": str(expected_trade_date + timedelta(days=CONFIG.setup_expire_days)),
            "last_seen_date": row.get("date"),
        })

    result = pd.DataFrame(snapshot_rows, columns=STATE_COLUMNS).drop_duplicates(subset=["code", "state"], keep="first")
    save_state(result)
    return result
