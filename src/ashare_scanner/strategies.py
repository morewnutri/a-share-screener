from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from .config import DataConfig, StrategyConfig


SIGNAL_ORDER = (
    "setup_contraction",
    "setup_accumulation",
    "breakout_today",
    "retest_after_breakout",
)


def add_relative_strength(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["rs20_excess_pct"] = result["return_20d_pct"] - result["benchmark_return_20d_pct"]
    result["rs60_excess_pct"] = result["return_60d_pct"] - result["benchmark_return_60d_pct"]
    groups = result.groupby("date", sort=False)
    result["rs20_percentile"] = groups["rs20_excess_pct"].rank(pct=True, method="average")
    result["rs60_percentile"] = groups["rs60_excess_pct"].rank(pct=True, method="average")
    return result


def add_factor_scores(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    trend = (
        (result["close"] > result["ma20"]).astype(int) * 6
        + (result["ma20"] > result["ma60"]).astype(int) * 6
        + (result["ma20_slope_5_pct"] > 0).astype(int) * 6
        + (result["ma60_slope_5_pct"] > 0).astype(int) * 4
        + (result["dif"] > result["dea"]).astype(int) * 3
    )
    position = (
        result["dist_to_prior_high20_pct"].between(-6, 3, inclusive="both").astype(int) * 10
        + result["extension_ma20_pct"].between(0, 8, inclusive="both").astype(int) * 5
        + (result["return_5d_pct"] <= 10).astype(int) * 5
    )
    volume_price = (
        result["vol_ratio_20"].between(0.6, 2.8, inclusive="both").astype(int) * 7
        + (result["up_down_volume_ratio_10"] >= 1.2).astype(int) * 7
        + (result["macd_rising"] == 1).astype(int) * 6
    )
    contraction = (
        (result["atr_contraction_ratio"] <= 0.90).astype(int) * 5
        + (result["volume_contraction_5_20"] <= 0.90).astype(int) * 5
        + (result["bb_contraction_ratio"] <= 0.90).astype(int) * 5
    )
    relative = (
        (result["rs20_percentile"] >= 0.80).astype(int) * 10
        + (result["rs60_percentile"] >= 0.70).astype(int) * 5
    )
    liquidity = (result["amount_ma20_prev"] >= 100_000_000).astype(int) * 5
    result["score_trend"] = trend.clip(upper=25)
    result["score_position"] = position.clip(upper=20)
    result["score_volume_price"] = volume_price.clip(upper=20)
    result["score_contraction"] = contraction.clip(upper=15)
    result["score_relative_strength"] = relative.clip(upper=15)
    result["score_liquidity"] = liquidity.clip(upper=5)
    score_columns = [column for column in result.columns if column.startswith("score_")]
    result["score_total"] = result[score_columns].sum(axis=1).clip(upper=100)
    return result


def signal_masks(
    frame: pd.DataFrame,
    data_config: DataConfig,
    strategy_config: StrategyConfig,
) -> Mapping[str, pd.Series]:
    enough_contraction = (
        (frame["atr_contraction_ratio"] <= 0.90).astype(int)
        + (frame["volume_contraction_5_20"] <= 0.90).astype(int)
        + (frame["bb_contraction_ratio"] <= 0.90).astype(int)
    ) >= 2
    trend_base = (
        (frame["close"] > frame["ma20"])
        & (frame["ma20"] > frame["ma60"])
        & (frame["ma20_slope_5_pct"] > 0)
    )
    eligible = (
        (frame["bars"] >= data_config.min_history_bars)
        & (frame["amount_ma20_prev"] >= strategy_config.min_amount_ma20)
        & (frame["one_price_limit"] == 0)
        & frame["prior_high_20"].notna()
        & frame["rs20_percentile"].notna()
    )
    setup_not_extended = (
        (frame["extension_ma20_pct"] <= strategy_config.max_setup_extension_ma20_pct)
        & (frame["return_5d_pct"] <= strategy_config.max_setup_return_5d_pct)
        & (frame["return_10d_pct"] <= 18)
    )

    setup_contraction = (
        eligible
        & trend_base
        & setup_not_extended
        & frame["dist_to_prior_high20_pct"].between(-6.0, -0.2, inclusive="both")
        & enough_contraction
        & (frame["rs20_percentile"] >= strategy_config.setup_min_rs_percentile)
    )
    setup_accumulation = (
        eligible
        & trend_base
        & setup_not_extended
        & frame["dist_to_prior_high20_pct"].between(-12.0, -0.2, inclusive="both")
        & (frame["up_down_volume_ratio_10"] >= 1.30)
        & (frame["obv_slope_10_pct"] > 0)
        & (frame["range_position_10"] >= 0.60)
        & (frame["rs20_percentile"] >= strategy_config.setup_min_rs_percentile)
    )
    breakout_today = (
        eligible
        & trend_base
        & (frame["close"] > frame["prior_high_20"])
        & frame["dist_to_prior_high20_pct"].between(0.0, 3.0, inclusive="both")
        & frame["vol_ratio_20"].between(1.20, 4.0, inclusive="both")
        & (frame["extension_ma20_pct"] <= strategy_config.max_breakout_extension_ma20_pct)
        & (frame["return_5d_pct"] <= strategy_config.max_breakout_return_5d_pct)
        & (frame["pct_chg"] < 9.5)
        & (frame["rsi14"] <= 88)
        & (frame["rs20_percentile"] >= strategy_config.breakout_min_rs_percentile)
    )
    retest_after_breakout = (
        eligible
        & trend_base
        & frame["bars_since_breakout"].between(2, 10, inclusive="both")
        & frame["retest_distance_pct"].between(-1.5, 3.0, inclusive="both")
        & (frame["retest_touch"] == 1)
        & (frame["vol_ratio_20"] <= 1.10)
        & (frame["close"] >= frame["open"])
        & (frame["extension_ma20_pct"] <= 10)
        & (frame["rs20_percentile"] >= strategy_config.setup_min_rs_percentile)
    )
    return {
        "setup_contraction": setup_contraction.fillna(False),
        "setup_accumulation": setup_accumulation.fillna(False),
        "breakout_today": breakout_today.fillna(False),
        "retest_after_breakout": retest_after_breakout.fillna(False),
    }


def apply_strategies(
    frame: pd.DataFrame,
    data_config: DataConfig,
    strategy_config: StrategyConfig,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    scored = add_factor_scores(frame)
    masks = signal_masks(scored, data_config, strategy_config)
    outputs: dict[str, pd.DataFrame] = {}
    for signal in SIGNAL_ORDER:
        selected = scored.loc[masks[signal]].copy()
        selected.insert(min(3, len(selected.columns)), "signal", signal)
        selected = selected.sort_values(
            ["score_total", "rs20_percentile", "amount_ma20_prev"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
        selected["rank"] = np.arange(1, len(selected) + 1)
        outputs[signal] = selected
    return scored, outputs

