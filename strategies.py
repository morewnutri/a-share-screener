from __future__ import annotations

import numpy as np
import pandas as pd

from config import CONFIG


def score_setup(row: pd.Series) -> float:
    score = 0.0
    if row["close_gt_ma20"]:
        score += 8
    if row["close_gt_ma60"]:
        score += 8
    if row["ma20_slope_5"] > 0:
        score += 10
    if row["ma60_slope_5"] > 0:
        score += 8
    if row["dif_gt_dea"]:
        score += 8
    if row["macd_rising"]:
        score += 6
    if pd.notna(row["dist_to_prior_high20_pct"]) and -3.0 <= row["dist_to_prior_high20_pct"] < -0.2:
        score += 16
    if pd.notna(row["vol_ratio_20"]) and 0.6 <= row["vol_ratio_20"] <= 1.2:
        score += 10
    if row.get("volatility_contraction", 0) == 1:
        score += 12
    if pd.notna(row["rsi14"]) and 50 <= row["rsi14"] <= 68:
        score += 8
    if pd.notna(row.get("rs_20_percentile", np.nan)) and row["rs_20_percentile"] >= CONFIG.min_rs_percentile:
        score += 14
    if pd.notna(row["extension_ma20_pct"]) and row["extension_ma20_pct"] > CONFIG.max_extension_ma20_pct:
        score -= 10
    if pd.notna(row["return_10d_pct"]) and row["return_10d_pct"] > CONFIG.max_return_10d_pct_for_setup:
        score -= 10
    return round(score, 2)


def score_breakout(row: pd.Series) -> float:
    score = 0.0
    if row["close_gt_ma20"]:
        score += 8
    if row["close_gt_ma60"]:
        score += 8
    if row["ma10_gt_ma20"]:
        score += 8
    if row["ma20_gt_ma60"]:
        score += 10
    if row["dif_gt_dea"]:
        score += 8
    if row["dif_gt_0"]:
        score += 8
    if row["macd_rising"]:
        score += 6
    if row["is_breakout_20"]:
        score += 16
    if pd.notna(row["vol_ratio_20"]) and 1.2 <= row["vol_ratio_20"] <= 2.8:
        score += 12
    if pd.notna(row["dist_to_prior_high20_pct"]) and 0 <= row["dist_to_prior_high20_pct"] <= 3:
        score += 10
    if pd.notna(row.get("rs_20_percentile", np.nan)) and row["rs_20_percentile"] >= CONFIG.min_rs_percentile:
        score += 12
    if pd.notna(row["extension_ma20_pct"]) and row["extension_ma20_pct"] > CONFIG.max_extension_ma20_pct:
        score -= 8
    return round(score, 2)


def score_retest(row: pd.Series) -> float:
    score = 0.0
    if row["close_gt_ma20"]:
        score += 8
    if row["close_gt_ma60"]:
        score += 8
    if row["ma20_gt_ma60"]:
        score += 10
    if row["dif_gt_dea"]:
        score += 8
    if pd.notna(row["dist_to_prior_high20_pct"]) and -2 <= row["dist_to_prior_high20_pct"] <= 1.5:
        score += 14
    if pd.notna(row["vol_ratio_20"]) and 0.5 <= row["vol_ratio_20"] <= 1.1:
        score += 12
    if row.get("volatility_contraction", 0) == 1:
        score += 10
    if pd.notna(row.get("rs_20_percentile", np.nan)) and row["rs_20_percentile"] >= CONFIG.min_rs_percentile:
        score += 10
    return round(score, 2)


def apply_strategies(feature_table: pd.DataFrame):
    df = feature_table.copy()
    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), df

    df["score_setup"] = df.apply(score_setup, axis=1)
    df["score_breakout"] = df.apply(score_breakout, axis=1)
    df["score_retest"] = df.apply(score_retest, axis=1)

    eligible = df[df["eligible"] == 1].copy()

    cond_setup = (
        (eligible["bars"] >= CONFIG.min_history_bars) &
        (eligible["close_gt_ma20"] == 1) &
        (eligible["close_gt_ma60"] == 1) &
        (eligible["ma20_slope_5"] > 0) &
        (eligible["dif_gt_dea"] == 1) &
        (eligible["rsi14"].between(50, 72, inclusive="both")) &
        (eligible["dist_to_prior_high20_pct"].between(-3.5, -0.1, inclusive="both")) &
        (eligible["extension_ma20_pct"] <= CONFIG.max_extension_ma20_pct) &
        (eligible["return_10d_pct"] <= CONFIG.max_return_10d_pct_for_setup)
    )

    cond_breakout = (
        (eligible["bars"] >= CONFIG.min_history_bars) &
        (eligible["close_gt_ma20"] == 1) &
        (eligible["close_gt_ma60"] == 1) &
        (eligible["ma10_gt_ma20"] == 1) &
        (eligible["ma20_gt_ma60"] == 1) &
        (eligible["dif_gt_dea"] == 1) &
        (eligible["dif_gt_0"] == 1) &
        (eligible["is_breakout_20"] == 1) &
        (eligible["dist_to_prior_high20_pct"].between(0, 3.0, inclusive="both"))
    )

    cond_retest = (
        (eligible["bars"] >= CONFIG.min_history_bars) &
        (eligible["close_gt_ma20"] == 1) &
        (eligible["close_gt_ma60"] == 1) &
        (eligible["ma20_gt_ma60"] == 1) &
        (eligible["dif_gt_dea"] == 1) &
        (eligible["dist_to_prior_high20_pct"].between(-2.5, 1.5, inclusive="both")) &
        (eligible["vol_ratio_20"].between(0.5, 1.2, inclusive="both"))
    )

    setup_all = eligible[cond_setup].copy().sort_values(
        ["score_setup", "rs_20_percentile", "vol_ratio_20"], ascending=[False, False, False]
    )
    breakout_all = eligible[cond_breakout].copy().sort_values(
        ["score_breakout", "rs_20_percentile", "vol_ratio_20"], ascending=[False, False, False]
    )
    retest_all = eligible[cond_retest].copy().sort_values(
        ["score_retest", "rs_20_percentile", "vol_ratio_20"], ascending=[False, False, False]
    )

    setup_top = setup_all.head(CONFIG.top_n)
    breakout_top = breakout_all.head(CONFIG.top_n)
    retest_top = retest_all.head(CONFIG.top_n)

    return setup_all, breakout_all, retest_all, df, setup_top, breakout_top, retest_top
