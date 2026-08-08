from __future__ import annotations

from collections.abc import Mapping
from functools import reduce
from operator import and_

import numpy as np
import pandas as pd

from .config import DataConfig, StrategyConfig


SIGNAL_ORDER = ("chip_base_ready", "chip_base_launch")


def _numeric(result: pd.DataFrame, column: str, default: float = np.nan) -> pd.Series:
    if column not in result:
        result[column] = default
    result[column] = pd.to_numeric(result[column], errors="coerce")
    return result[column]


def add_relative_strength(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    benchmark20 = _numeric(result, "benchmark_return_20d_pct", 0.0).fillna(0.0)
    benchmark60 = _numeric(result, "benchmark_return_60d_pct", 0.0).fillna(0.0)
    result["rs20_excess_pct"] = result["return_20d_pct"] - benchmark20
    result["rs60_excess_pct"] = result["return_60d_pct"] - benchmark60
    groups = result.groupby("date", sort=False)
    result["rs20_percentile"] = groups["rs20_excess_pct"].rank(pct=True, method="average")
    result["rs60_percentile"] = groups["rs60_excess_pct"].rank(pct=True, method="average")
    if "float_market_cap" in result:
        result["float_market_cap"] = pd.to_numeric(
            result["float_market_cap"], errors="coerce"
        )
        result["float_cap_percentile"] = groups["float_market_cap"].rank(
            pct=True, method="average"
        )
    else:
        result["float_cap_percentile"] = np.nan
    return result


def add_factor_scores(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    benchmark_ok = _numeric(result, "benchmark_risk_ok", 1.0).fillna(1.0) >= 1

    deep_drawdown = result["base_drawdown_from_120_high_pct"] <= -20
    prior_decline = result["pre_base_decline_60_pct"] <= -12
    result["chip_score_base"] = (
        deep_drawdown.astype(int) * 8
        + prior_decline.astype(int) * 6
        + (result["base_position_120_pre3"] <= 0.35).astype(int) * 7
        + (result["base_width_20_pre3_pct"] <= 22).astype(int) * 8
        + (result["base_return_20_pre3_pct"].abs() <= 8).astype(int) * 4
        + (result["base_turnover_sum_20_pre3_pct"] >= 20).astype(int) * 2
    ).clip(upper=35)

    tight70 = result["chip_70_width_pct"] <= 18
    acceptable70 = result["chip_70_width_pct"].between(18, 26, inclusive="right")
    very_low_peak = result["chip_peak_position"] <= 0.35
    low_peak = result["chip_peak_position"].between(0.35, 0.55, inclusive="right")
    strong_peak = result["chip_peak_band_share_pct"] >= 28
    visible_peak = result["chip_peak_band_share_pct"].between(18, 28, inclusive="left")
    dense_low_zone = result["chip_low_zone_share_pct"] >= 55
    acceptable_low_zone = result["chip_low_zone_share_pct"].between(
        42, 55, inclusive="left"
    )
    result["chip_score_profile"] = (
        tight70.astype(int) * 10
        + acceptable70.astype(int) * 6
        + very_low_peak.astype(int) * 8
        + low_peak.astype(int) * 5
        + strong_peak.astype(int) * 8
        + visible_peak.astype(int) * 5
        + dense_low_zone.astype(int) * 5
        + acceptable_low_zone.astype(int) * 3
        + (result["chip_peak_dominance"] >= 1.25).astype(int) * 2
        + (result["chip_significant_peak_count"] <= 2).astype(int) * 2
        + (result["chip_overhead_ratio_pct"] <= 60).astype(int) * 2
    ).clip(upper=40)

    result["chip_score_launch"] = (
        (result["early_launch_price_action"] == 1).astype(int) * 8
        + (result["close"] >= result["chip_peak_price"]).astype(int) * 2
        + (result["vol_ratio_20"] >= 1.05).astype(int) * 3
        + (result["macd_rising"] == 1).astype(int) * 2
    ).clip(upper=15)
    result["chip_score_risk"] = (
        (result["distribution_day_count_5"] == 0).astype(int) * 3
        + (result["breakout_failed_fast"] == 0).astype(int) * 2
        + (result["return_10d_pct"] <= 35).astype(int) * 2
        + (result["chip_overhead_ratio_pct"] <= 60).astype(int) * 2
        + benchmark_ok.astype(int)
    ).clip(upper=10)

    result["chip_structure_score"] = (
        result[["chip_score_base", "chip_score_profile", "chip_score_risk"]]
        .sum(axis=1)
        .clip(upper=85)
    )
    result["chip_base_ready_score"] = result["chip_structure_score"]
    result["chip_base_launch_score"] = (
        result["chip_structure_score"] + result["chip_score_launch"]
    ).clip(upper=100)
    result["score_total"] = result[
        ["chip_base_ready_score", "chip_base_launch_score"]
    ].max(axis=1)
    return result


def signal_funnels(
    frame: pd.DataFrame,
    data_config: DataConfig,
    strategy_config: StrategyConfig,
) -> Mapping[str, list[tuple[str, pd.Series]]]:
    history_ready = frame["bars"] >= data_config.min_history_bars
    liquid = frame["amount_ma20_prev"] >= strategy_config.min_amount_ma20
    features_ready = (
        (frame["one_price_limit"] == 0)
        & frame["base_high_20_pre3"].notna()
        & frame["chip_peak_price"].notna()
        & (frame["chip_model_coverage_pct"] >= 70)
    )
    common = [
        ("history_bars", history_ready),
        ("liquidity", liquid),
        ("chip_features_ready", features_ready),
    ]

    bottom_location = (
        (frame["position_250"] <= strategy_config.chip_max_position_250)
        & (frame["base_position_120_pre3"] <= 0.55)
        & (
            (frame["base_drawdown_from_120_high_pct"] <= -18)
            | (frame["pre_base_decline_60_pct"] <= -12)
        )
    )
    platform = (
        (frame["base_width_20_pre3_pct"] <= strategy_config.chip_max_base_width_pct)
        & (
            frame["base_return_20_pre3_pct"].abs()
            <= strategy_config.chip_max_base_abs_return_pct
        )
    )
    concentrated_low_peak = (
        (frame["chip_70_width_pct"] <= strategy_config.chip_max_70_width_pct)
        & (frame["chip_peak_position"] <= strategy_config.chip_max_peak_position)
        & (
            frame["chip_peak_band_share_pct"]
            >= strategy_config.chip_min_peak_band_share_pct
        )
        & (
            frame["chip_low_zone_share_pct"]
            >= strategy_config.chip_min_low_zone_share_pct
        )
    )
    risk_ok = (
        (frame["distribution_day_count_5"] <= strategy_config.chip_max_distribution_days_5)
        & (frame["breakout_failed_fast"] == 0)
        & (frame["breakout_stall_distribution"] == 0)
    )
    ready_position = (
        (frame["early_launch_price_action"] == 0)
        & frame["chip_peak_distance_pct"].between(
            -6,
            strategy_config.chip_ready_max_peak_distance_pct,
            inclusive="both",
        )
    )
    launch = (
        (frame["early_launch_price_action"] == 1)
        & frame["chip_peak_distance_pct"].between(
            -3,
            strategy_config.chip_launch_max_peak_distance_pct,
            inclusive="both",
        )
        & (frame["return_10d_pct"] <= strategy_config.chip_launch_max_return_10d_pct)
    )
    return {
        "chip_base_ready": common
        + [
            ("bottom_location", bottom_location),
            ("sideways_platform", platform),
            ("concentrated_low_chip_peak", concentrated_low_peak),
            ("risk_control", risk_ok),
            ("waiting_near_peak", ready_position),
            (
                "chip_base_ready_score",
                frame["chip_base_ready_score"]
                >= strategy_config.chip_base_ready_score_min,
            ),
        ],
        "chip_base_launch": common
        + [
            ("bottom_location", bottom_location),
            ("sideways_platform", platform),
            ("concentrated_low_chip_peak", concentrated_low_peak),
            ("risk_control", risk_ok),
            ("early_launch", launch),
            (
                "chip_base_launch_score",
                frame["chip_base_launch_score"]
                >= strategy_config.chip_base_launch_score_min,
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
        ["completion_ratio", "score_total", "chip_peak_band_share_pct"],
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
        model_score = (
            "chip_base_ready_score"
            if signal == "chip_base_ready"
            else "chip_base_launch_score"
        )
        selected = selected.sort_values(
            [model_score, "chip_peak_band_share_pct", "amount_ma20_prev"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
        selected["rank"] = np.arange(1, len(selected) + 1)
        outputs[signal] = selected
    return scored, outputs
