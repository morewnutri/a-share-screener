from __future__ import annotations

from collections.abc import Mapping
from functools import reduce
from operator import and_

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


def signal_funnels(
    frame: pd.DataFrame,
    data_config: DataConfig,
    strategy_config: StrategyConfig,
) -> Mapping[str, list[tuple[str, pd.Series]]]:
    enough_contraction = (
        (frame["atr_contraction_ratio"] <= strategy_config.contraction_ratio_max).astype(int)
        + (frame["volume_contraction_5_20"] <= strategy_config.contraction_ratio_max).astype(int)
        + (frame["bb_contraction_ratio"] <= strategy_config.contraction_ratio_max).astype(int)
    ) >= strategy_config.contraction_min_count
    trend_base = (
        (frame["close"] > frame["ma20"])
        & (frame["ma20"] > frame["ma60"])
        & (frame["ma20_slope_5_pct"] > 0)
    )
    history_ready = frame["bars"] >= data_config.min_history_bars
    liquid = frame["amount_ma20_prev"] >= strategy_config.min_amount_ma20
    features_ready = (
        (frame["one_price_limit"] == 0)
        & frame["prior_high_20"].notna()
        & frame["rs20_percentile"].notna()
    )
    setup_not_extended = (
        (frame["extension_ma20_pct"] <= strategy_config.max_setup_extension_ma20_pct)
        & (frame["return_5d_pct"] <= strategy_config.max_setup_return_5d_pct)
        & (frame["return_10d_pct"] <= 18)
    )
    common = [
        ("history_bars", history_ready),
        ("liquidity", liquid),
        ("tradable_features", features_ready),
        ("trend_structure", trend_base),
    ]
    return {
        "setup_contraction": common
        + [
            ("not_extended", setup_not_extended),
            (
                "near_prior_high",
                frame["dist_to_prior_high20_pct"].between(
                    strategy_config.setup_contraction_distance_min_pct,
                    strategy_config.setup_distance_max_pct,
                    inclusive="both",
                ),
            ),
            (
                "relative_strength",
                frame["rs20_percentile"] >= strategy_config.setup_min_rs_percentile,
            ),
            ("contraction", enough_contraction),
        ],
        "setup_accumulation": common
        + [
            ("not_extended", setup_not_extended),
            (
                "near_prior_high",
                frame["dist_to_prior_high20_pct"].between(
                    strategy_config.accumulation_distance_min_pct,
                    strategy_config.setup_distance_max_pct,
                    inclusive="both",
                ),
            ),
            (
                "relative_strength",
                frame["rs20_percentile"] >= strategy_config.setup_min_rs_percentile,
            ),
            (
                "up_down_volume",
                frame["up_down_volume_ratio_10"]
                >= strategy_config.accumulation_up_down_volume_min,
            ),
            ("obv_improving", frame["obv_slope_10_pct"] > 0),
            (
                "range_position",
                frame["range_position_10"]
                >= strategy_config.accumulation_range_position_min,
            ),
        ],
        "breakout_today": common
        + [
            ("above_prior_high", frame["close"] > frame["prior_high_20"]),
            (
                "breakout_distance",
                frame["dist_to_prior_high20_pct"].between(
                    0.0,
                    strategy_config.breakout_distance_max_pct,
                    inclusive="both",
                ),
            ),
            (
                "breakout_volume",
                frame["vol_ratio_20"].between(
                    strategy_config.breakout_volume_ratio_min,
                    strategy_config.breakout_volume_ratio_max,
                    inclusive="both",
                ),
            ),
            (
                "not_overextended",
                (frame["extension_ma20_pct"] <= strategy_config.max_breakout_extension_ma20_pct)
                & (frame["return_5d_pct"] <= strategy_config.max_breakout_return_5d_pct),
            ),
            ("buyable_close", (frame["pct_chg"] < 9.5) & (frame["rsi14"] <= 88)),
            (
                "relative_strength",
                frame["rs20_percentile"] >= strategy_config.breakout_min_rs_percentile,
            ),
        ],
        "retest_after_breakout": common
        + [
            (
                "recent_breakout",
                frame["bars_since_breakout"].between(2, 10, inclusive="both"),
            ),
            (
                "near_breakout_level",
                frame["retest_distance_pct"].between(-1.5, 3.0, inclusive="both")
                & (frame["retest_touch"] == 1),
            ),
            (
                "quiet_retest",
                frame["vol_ratio_20"] <= strategy_config.retest_volume_ratio_max,
            ),
            (
                "close_recovered",
                (frame["close"] >= frame["open"])
                & (frame["extension_ma20_pct"] <= 10),
            ),
            (
                "relative_strength",
                frame["rs20_percentile"] >= strategy_config.setup_min_rs_percentile,
            ),
        ],
    }


def signal_masks(
    frame: pd.DataFrame,
    data_config: DataConfig,
    strategy_config: StrategyConfig,
) -> Mapping[str, pd.Series]:
    funnels = signal_funnels(frame, data_config, strategy_config)
    return {
        signal: reduce(and_, (condition.fillna(False) for _, condition in steps))
        for signal, steps in funnels.items()
    }


def screening_diagnostics(
    frame: pd.DataFrame,
    data_config: DataConfig,
    strategy_config: StrategyConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    funnels = signal_funnels(frame, data_config, strategy_config)
    masks = signal_masks(frame, data_config, strategy_config)
    funnel_rows: list[dict] = []
    total = len(frame)

    best_ratio = np.full(total, -1.0)
    best_signal = np.full(total, "", dtype=object)
    best_failed_at = np.full(total, "", dtype=object)
    best_passed = np.zeros(total, dtype=int)
    best_total = np.zeros(total, dtype=int)

    for signal, steps in funnels.items():
        running = np.ones(total, dtype=bool)
        first_failure = np.full(total, "", dtype=object)
        passed = np.zeros(total, dtype=int)
        previous_count = total
        for step_number, (step, condition) in enumerate(steps, start=1):
            values = condition.fillna(False).to_numpy(dtype=bool)
            first_failure[running & ~values] = step
            running &= values
            passed += running.astype(int)
            count = int(running.sum())
            funnel_rows.append(
                {
                    "signal": signal,
                    "step_number": step_number,
                    "step": step,
                    "remaining_count": count,
                    "retention_from_previous_pct": round(count / previous_count * 100, 2)
                    if previous_count
                    else 0.0,
                    "retention_from_all_pct": round(count / total * 100, 2) if total else 0.0,
                }
            )
            previous_count = count
        ratio = passed / len(steps)
        better = ratio > best_ratio
        best_ratio[better] = ratio[better]
        best_signal[better] = signal
        best_failed_at[better] = first_failure[better]
        best_passed[better] = passed[better]
        best_total[better] = len(steps)

    matched = np.zeros(total, dtype=bool)
    for mask in masks.values():
        matched |= mask.to_numpy(dtype=bool)
    near_misses = frame.loc[~matched].copy()
    near_misses["closest_signal"] = best_signal[~matched]
    near_misses["failed_at"] = best_failed_at[~matched]
    near_misses["passed_steps"] = best_passed[~matched]
    near_misses["total_steps"] = best_total[~matched]
    near_misses["completion_ratio"] = best_ratio[~matched]
    near_misses = near_misses.sort_values(
        ["completion_ratio", "score_total", "rs20_percentile"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    near_misses["near_miss_rank"] = np.arange(1, len(near_misses) + 1)
    return pd.DataFrame(funnel_rows), near_misses


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
