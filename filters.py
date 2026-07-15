from __future__ import annotations

import pandas as pd

from config import CONFIG


def apply_base_filters(df: pd.DataFrame, expected_trade_date):
    work = df.copy()
    work["eligible"] = 1
    work["filter_reason"] = ""

    work.loc[work["bars"] < CONFIG.min_history_bars, ["eligible", "filter_reason"]] = [0, "insufficient_bars"]
    work.loc[pd.to_datetime(work["date"]).dt.date < expected_trade_date, ["eligible", "filter_reason"]] = [0, "stale_hist"]
    work.loc[work["close"] < CONFIG.min_price, ["eligible", "filter_reason"]] = [0, "low_price"]
    work.loc[work["amount_ma20"].fillna(0) < CONFIG.min_amount_ma20, ["eligible", "filter_reason"]] = [0, "low_liquidity"]

    return work
