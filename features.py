from __future__ import annotations

import numpy as np
import pandas as pd

from indicators import max_drawdown_from_high, slope_n


def add_relative_strength(stock_df: pd.DataFrame, benchmark_df: pd.DataFrame) -> pd.DataFrame:
    df = stock_df.copy()
    bm = benchmark_df[["date", "close"]].copy().rename(columns={"close": "benchmark_close"})
    bm = bm.sort_values("date").reset_index(drop=True)
    bm["benchmark_ret_20"] = bm["benchmark_close"].pct_change(20) * 100
    bm["benchmark_ret_60"] = bm["benchmark_close"].pct_change(60) * 100
    df = df.merge(bm[["date", "benchmark_ret_20", "benchmark_ret_60"]], on="date", how="left")
    df["rs_20"] = df["return_20d_pct"] - df["benchmark_ret_20"]
    df["stock_ret_60"] = df["close"].pct_change(60) * 100
    df["rs_60"] = df["stock_ret_60"] - df["benchmark_ret_60"]
    return df


def summarize_last_row(ind_df: pd.DataFrame, code: str, name: str, source: str) -> dict:
    last = ind_df.iloc[-1]
    close = float(last.get("close")) if pd.notna(last.get("close")) else np.nan
    prior_high_20 = last.get("prior_high_20")
    prior_high_60 = last.get("prior_high_60")
    volume = last.get("volume")
    vma20_prev = last.get("vma20_prev")

    dist_to_prior_high20 = (close / prior_high_20 - 1) * 100 if pd.notna(prior_high_20) and prior_high_20 != 0 else np.nan
    dist_to_prior_high60 = (close / prior_high_60 - 1) * 100 if pd.notna(prior_high_60) and prior_high_60 != 0 else np.nan
    vol_ratio_20 = volume / vma20_prev if pd.notna(vma20_prev) and vma20_prev not in [0, np.nan] else np.nan

    summary = {
        "code": code,
        "name": name,
        "source": source,
        "date": pd.to_datetime(last["date"]).strftime("%Y-%m-%d"),
        "close": close,
        "pct_chg": last.get("pct_chg"),
        "turnover": last.get("turnover"),
        "ma5": last.get("ma5"),
        "ma10": last.get("ma10"),
        "ma20": last.get("ma20"),
        "ma60": last.get("ma60"),
        "ma120": last.get("ma120"),
        "dif": last.get("dif"),
        "dea": last.get("dea"),
        "macd": last.get("macd"),
        "rsi14": last.get("rsi14"),
        "volume": volume,
        "vma20_prev": vma20_prev,
        "vol_ratio_20": vol_ratio_20,
        "prior_high_20": prior_high_20,
        "prior_high_60": prior_high_60,
        "dist_to_prior_high20_pct": dist_to_prior_high20,
        "dist_to_prior_high60_pct": dist_to_prior_high60,
        "is_breakout_20": int(last.get("is_breakout_20", 0)),
        "is_breakout_60": int(last.get("is_breakout_60", 0)),
        "ma20_slope_5": slope_n(ind_df["ma20"], 5),
        "ma60_slope_5": slope_n(ind_df["ma60"], 5),
        "close_slope_5": slope_n(ind_df["close"], 5),
        "drawdown_from_20d_high_pct": max_drawdown_from_high(ind_df["close"], 20),
        "bars": len(ind_df),
        "atr_pct": last.get("atr_pct"),
        "atr_pct_mean_20": last.get("atr_pct_mean_20"),
        "amount_ma20": last.get("amount_ma20"),
        "return_5d_pct": last.get("return_5d_pct"),
        "return_10d_pct": last.get("return_10d_pct"),
        "return_20d_pct": last.get("return_20d_pct"),
        "extension_ma20_pct": last.get("extension_ma20_pct"),
        "macd_rising": int(last.get("macd_rising", 0)),
        "dif_slope_3": last.get("dif_slope_3"),
        "rs_20": last.get("rs_20"),
        "rs_60": last.get("rs_60"),
    }

    summary["close_gt_ma20"] = int(summary["close"] > summary["ma20"]) if pd.notna(summary["ma20"]) else 0
    summary["close_gt_ma60"] = int(summary["close"] > summary["ma60"]) if pd.notna(summary["ma60"]) else 0
    summary["ma5_gt_ma10"] = int(summary["ma5"] > summary["ma10"]) if pd.notna(summary["ma5"]) and pd.notna(summary["ma10"]) else 0
    summary["ma10_gt_ma20"] = int(summary["ma10"] > summary["ma20"]) if pd.notna(summary["ma10"]) and pd.notna(summary["ma20"]) else 0
    summary["ma20_gt_ma60"] = int(summary["ma20"] > summary["ma60"]) if pd.notna(summary["ma20"]) and pd.notna(summary["ma60"]) else 0
    summary["dif_gt_dea"] = int(summary["dif"] > summary["dea"]) if pd.notna(summary["dif"]) and pd.notna(summary["dea"]) else 0
    summary["dif_gt_0"] = int(summary["dif"] > 0) if pd.notna(summary["dif"]) else 0
    summary["volatility_contraction"] = int(
        pd.notna(summary["atr_pct"]) and pd.notna(summary["atr_pct_mean_20"]) and summary["atr_pct"] < summary["atr_pct_mean_20"]
    )
    return summary


def finalize_cross_section(feature_table: pd.DataFrame) -> pd.DataFrame:
    df = feature_table.copy()
    if df.empty:
        return df
    df["rs_20_percentile"] = df["rs_20"].rank(pct=True)
    df["rs_60_percentile"] = df["rs_60"].rank(pct=True)
    return df
