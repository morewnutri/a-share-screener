from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd


FUND_FLOW_WINDOWS = (3, 5, 10, 20)
FUND_FLOW_FEATURE_COLUMNS = (
    "fund_flow_available",
    "fund_flow_is_current",
    "fund_flow_latest_date",
    "fund_flow_all_windows_positive",
    "fund_flow_positive_window_count",
    "fund_flow_rank_reason",
    *(f"main_net_inflow_{window}d_amount" for window in FUND_FLOW_WINDOWS),
    *(f"main_net_inflow_{window}d_yi" for window in FUND_FLOW_WINDOWS),
)


def summarize_fund_flow(history: pd.DataFrame, expected: date) -> dict:
    frame = history.copy()
    if frame.empty:
        return {}
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["main_net_inflow_amount"] = pd.to_numeric(
        frame["main_net_inflow_amount"], errors="coerce"
    )
    frame = frame.dropna(subset=["date", "main_net_inflow_amount"])
    frame = frame[frame["date"].dt.date <= expected]
    frame = frame.drop_duplicates("date", keep="last").sort_values("date")
    if frame.empty:
        return {}

    latest = frame["date"].iloc[-1].date()
    result: dict[str, object] = {
        "fund_flow_available": int(len(frame) >= max(FUND_FLOW_WINDOWS)),
        "fund_flow_is_current": int(latest == expected),
        "fund_flow_latest_date": latest.isoformat(),
    }
    positives = 0
    for window in FUND_FLOW_WINDOWS:
        amount = (
            float(frame["main_net_inflow_amount"].tail(window).sum())
            if len(frame) >= window
            else np.nan
        )
        result[f"main_net_inflow_{window}d_amount"] = amount
        result[f"main_net_inflow_{window}d_yi"] = amount / 100_000_000 if np.isfinite(amount) else np.nan
        positives += int(np.isfinite(amount) and amount > 0)
    available = bool(result["fund_flow_available"])
    all_positive = available and positives == len(FUND_FLOW_WINDOWS)
    result["fund_flow_all_windows_positive"] = int(all_positive)
    result["fund_flow_positive_window_count"] = positives
    if all_positive:
        reason = "3/5/10/20日均净流入"
    elif available:
        reason = f"{positives}/4个周期净流入"
    else:
        reason = "资金历史不足20日"
    if latest != expected:
        reason = f"资金数据滞后至{latest.isoformat()}"
    result["fund_flow_rank_reason"] = reason
    return result


def merge_fund_flow_features(frame: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    existing = [column for column in FUND_FLOW_FEATURE_COLUMNS if column in result]
    if existing:
        result = result.drop(columns=existing)
    if features.empty:
        for column in FUND_FLOW_FEATURE_COLUMNS:
            if column in ("fund_flow_latest_date", "fund_flow_rank_reason"):
                result[column] = ""
            else:
                result[column] = np.nan
        return result
    clean = features.copy()
    clean["code"] = clean["code"].astype(str).str.zfill(6)
    result["code"] = result["code"].astype(str).str.zfill(6)
    return result.merge(clean, on="code", how="left")


def rank_signal_by_fund_flow(frame: pd.DataFrame, model_score: str) -> pd.DataFrame:
    result = frame.copy()
    numeric_defaults = {
        "fund_flow_is_current": 0,
        "fund_flow_all_windows_positive": 0,
        "fund_flow_positive_window_count": 0,
        **{f"main_net_inflow_{window}d_amount": 0.0 for window in FUND_FLOW_WINDOWS},
    }
    sort_columns: list[str] = []
    for column, default in numeric_defaults.items():
        if column not in result:
            result[column] = default
        result[column] = pd.to_numeric(result[column], errors="coerce").fillna(default)
        sort_columns.append(column)
    sort_columns.append(model_score)
    result = result.sort_values(
        sort_columns,
        ascending=[False] * len(sort_columns),
    ).reset_index(drop=True)
    result["rank"] = np.arange(1, len(result) + 1)
    return result
