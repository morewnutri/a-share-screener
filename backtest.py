from __future__ import annotations

import pandas as pd


def backtest_label_forward_return(hist: pd.DataFrame, hold_days: int = 10, take_profit_pct: float = 12.0):
    df = hist.copy().sort_values("date").reset_index(drop=True)
    df["future_max_close"] = df["close"].shift(-1).rolling(hold_days).max()
    df["future_return_max_pct"] = (df["future_max_close"] / df["close"] - 1) * 100
    df["label_hit"] = (df["future_return_max_pct"] >= take_profit_pct).astype(int)
    return df


def evaluate_signals(signal_df: pd.DataFrame, label_df: pd.DataFrame):
    if signal_df.empty or label_df.empty:
        return {"signal_count": 0, "hit_count": 0, "hit_rate": 0.0}
    merged = signal_df.merge(label_df[["date", "label_hit"]], on="date", how="left")
    signal_count = len(merged)
    hit_count = int(merged["label_hit"].fillna(0).sum())
    hit_rate = round(hit_count / signal_count * 100, 2) if signal_count > 0 else 0.0
    return {
        "signal_count": signal_count,
        "hit_count": hit_count,
        "hit_rate": hit_rate,
    }
