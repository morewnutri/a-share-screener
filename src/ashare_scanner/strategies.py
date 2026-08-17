from __future__ import annotations

from collections.abc import Mapping
from functools import reduce
from operator import and_

import numpy as np
import pandas as pd

from .config import DataConfig, StrategyConfig


SIGNAL_ORDER = ("chip_base_ready", "chip_base_launch", "chip_base_rebound")


def _numeric(result: pd.DataFrame, column: str, default: float = np.nan) -> pd.Series:
    if column not in result:
        result[column] = default
    result[column] = pd.to_numeric(result[column], errors="coerce")
    return result[column]


def _fallback_column(
    result: pd.DataFrame,
    target: str,
    source: str,
    default: float = np.nan,
) -> None:
    if target not in result:
        result[target] = result[source] if source in result else default


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
    aliases = {
        "adaptive_base_high": "base_high_20_pre3",
        "adaptive_base_width_pct": "base_width_20_pre3_pct",
        "adaptive_base_return_pct": "base_return_20_pre3_pct",
        "adaptive_base_turnover_sum_pct": "base_turnover_sum_20_pre3_pct",
        "adaptive_base_drawdown_120_pct": "base_drawdown_from_120_high_pct",
        "adaptive_base_position_120": "base_position_120_pre3",
        "adaptive_pre_base_decline_60_pct": "pre_base_decline_60_pct",
        "adaptive_launch_price_action": "early_launch_price_action",
        "adaptive_rebound_price_action": "rebound_price_action",
    }
    for target, source in aliases.items():
        _fallback_column(result, target, source)
    if "adaptive_ready_price_action" not in result:
        launch_flag = pd.to_numeric(
            result.get(
                "early_launch_price_action",
                pd.Series(0, index=result.index),
            ),
            errors="coerce",
        ).fillna(0)
        result["adaptive_ready_price_action"] = (launch_flag == 0).astype(int)
    _fallback_column(result, "adaptive_base_window", "", 20.0)
    _fallback_column(result, "adaptive_base_offset", "", 3.0)
    if "adaptive_trend_votes" not in result:
        launch_flag = pd.to_numeric(
            result.get(
                "early_launch_price_action",
                pd.Series(0, index=result.index),
            ),
            errors="coerce",
        ).fillna(0)
        result["adaptive_trend_votes"] = launch_flag * 3
    if "distance_from_adaptive_base_high_pct" not in result:
        result["distance_from_adaptive_base_high_pct"] = (
            result["close"] / result["adaptive_base_high"] - 1
        ) * 100

    numeric_columns = (
        "adaptive_base_window",
        "adaptive_base_offset",
        "adaptive_base_width_pct",
        "adaptive_base_return_pct",
        "adaptive_base_turnover_sum_pct",
        "adaptive_base_drawdown_120_pct",
        "adaptive_base_position_120",
        "adaptive_pre_base_decline_60_pct",
        "adaptive_ready_price_action",
        "adaptive_launch_price_action",
        "adaptive_rebound_price_action",
        "adaptive_trend_votes",
        "distance_from_adaptive_base_high_pct",
        "chip_peak_band_share_pct",
        "chip_70_width_pct",
        "chip_peak_position",
        "chip_peak_dominance",
        "chip_significant_peak_count",
        "chip_low_zone_share_pct",
        "chip_overhead_ratio_pct",
        "chip_peak_distance_pct",
        "distribution_day_count_5",
        "breakout_failed_fast",
        "breakout_stall_distribution",
        "return_5d_pct",
        "return_10d_pct",
        "return_20d_pct",
        "vol_ratio_20",
        "up_down_volume_ratio_10",
        "macd_rising",
    )
    for column in numeric_columns:
        _numeric(result, column)

    width = result["adaptive_base_width_pct"]
    base_return = result["adaptive_base_return_pct"].abs()
    result["chip_score_base"] = (
        (width <= 16).astype(int) * 12
        + width.between(16, 28, inclusive="right").astype(int) * 8
        + width.between(28, 42, inclusive="right").astype(int) * 4
        + (base_return <= 6).astype(int) * 8
        + base_return.between(6, 14, inclusive="right").astype(int) * 5
        + base_return.between(14, 22, inclusive="right").astype(int) * 2
        + (result["adaptive_base_window"] >= 40).astype(int) * 4
        + result["adaptive_base_window"].between(20, 39, inclusive="both").astype(int) * 2
        + (result["adaptive_base_turnover_sum_pct"] >= 15).astype(int) * 3
    ).clip(upper=35)

    peak_share = result["chip_peak_band_share_pct"]
    chip_width = result["chip_70_width_pct"]
    result["chip_score_profile"] = (
        (peak_share >= 25).astype(int) * 12
        + peak_share.between(18, 25, inclusive="left").astype(int) * 9
        + peak_share.between(12, 18, inclusive="left").astype(int) * 6
        + peak_share.between(10, 12, inclusive="left").astype(int) * 4
        + (chip_width <= 18).astype(int) * 10
        + chip_width.between(18, 30, inclusive="right").astype(int) * 7
        + chip_width.between(30, 45, inclusive="right").astype(int) * 4
        + (result["chip_peak_dominance"] >= 1.50).astype(int) * 5
        + result["chip_peak_dominance"].between(1.15, 1.50, inclusive="left").astype(int) * 3
        + (result["chip_significant_peak_count"] <= 2).astype(int) * 4
        + result["chip_significant_peak_count"].between(3, 4, inclusive="both").astype(int) * 2
        + (result["chip_low_zone_share_pct"] >= 35).astype(int) * 4
        + result["chip_low_zone_share_pct"].between(20, 35, inclusive="left").astype(int) * 2
    ).clip(upper=35)

    drawdown = result["adaptive_base_drawdown_120_pct"]
    position = result["adaptive_base_position_120"]
    prior_decline = result["adaptive_pre_base_decline_60_pct"]
    result["chip_score_context"] = (
        (drawdown <= -20).astype(int) * 6
        + drawdown.between(-20, -8, inclusive="left").astype(int) * 4
        + (position <= 0.40).astype(int) * 5
        + position.between(0.40, 0.75, inclusive="right").astype(int) * 3
        + position.between(0.75, 0.88, inclusive="right").astype(int)
        + (prior_decline <= -12).astype(int) * 4
        + prior_decline.between(-12, -5, inclusive="left").astype(int) * 2
    ).clip(upper=15)

    result["chip_score_launch"] = (
        result["adaptive_trend_votes"].clip(lower=0, upper=5) * 2
        + (result["vol_ratio_20"] >= 1.05).astype(int) * 2
        + (result["up_down_volume_ratio_10"] >= 1.15).astype(int) * 2
        + (result["macd_rising"] == 1).astype(int)
    ).clip(upper=15)
    result["chip_score_risk"] = (
        (result["distribution_day_count_5"] <= 2).astype(int) * 2
        + (result["breakout_failed_fast"] == 0).astype(int)
        + (result["breakout_stall_distribution"] == 0).astype(int)
        + benchmark_ok.astype(int)
    ).clip(upper=5)

    structure = result[
        ["chip_score_base", "chip_score_profile", "chip_score_context", "chip_score_risk"]
    ].sum(axis=1)
    result["chip_structure_score"] = structure.clip(upper=90)
    result["chip_base_ready_score"] = (
        structure + (result["adaptive_ready_price_action"] == 1).astype(int) * 10
    ).clip(upper=100)
    result["chip_base_launch_score"] = (
        structure
        + result["chip_score_launch"]
        + (result["adaptive_launch_price_action"] == 1).astype(int) * 5
    ).clip(upper=100)
    result["chip_base_rebound_score"] = (
        structure
        + result["chip_score_launch"]
        + (result["adaptive_rebound_price_action"] == 1).astype(int) * 5
    ).clip(upper=100)
    result["score_total"] = result[
        ["chip_base_ready_score", "chip_base_launch_score", "chip_base_rebound_score"]
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
        & (
            (frame["adaptive_base_position_120"] <= 0.88)
            | (frame["adaptive_base_drawdown_120_pct"] <= -8)
            | (frame["adaptive_pre_base_decline_60_pct"] <= -5)
        )
    )
    platform = (
        frame["adaptive_base_high"].notna()
        & (frame["adaptive_base_width_pct"] <= strategy_config.chip_max_base_width_pct)
        & (
            frame["adaptive_base_return_pct"].abs()
            <= strategy_config.chip_max_base_abs_return_pct
        )
    )
    strong_visible_peak = (
        (
            frame["chip_peak_band_share_pct"]
            >= strategy_config.chip_strong_peak_band_share_pct
        )
        & (
            frame["chip_70_width_pct"]
            <= strategy_config.chip_strong_peak_max_70_width_pct
        )
    )
    concentrated_low_peak = (
        (
            (frame["chip_70_width_pct"] <= strategy_config.chip_max_70_width_pct)
            | strong_visible_peak
        )
        & (frame["chip_peak_position"] <= strategy_config.chip_max_peak_position)
        & (
            frame["chip_peak_band_share_pct"]
            >= strategy_config.chip_min_peak_band_share_pct
        )
    )
    risk_ok = (
        (frame["distribution_day_count_5"] <= strategy_config.chip_max_distribution_days_5 + 1)
        & ~(
            (frame["breakout_failed_fast"] == 1)
            & (frame["distance_from_adaptive_base_high_pct"] < -8)
        )
    )
    recent_base = frame["adaptive_base_offset"] <= 8
    early_launch_evidence = (frame["adaptive_launch_price_action"] == 1) | (
        recent_base
        & (frame["return_5d_pct"] >= 3)
        & (frame["adaptive_trend_votes"] >= 2)
    )
    ready_position = (
        recent_base
        & ~early_launch_evidence
        & frame["chip_peak_distance_pct"].between(
            -12,
            strategy_config.chip_ready_max_peak_distance_pct,
            inclusive="both",
        )
    )
    launch = (
        early_launch_evidence
        & frame["chip_peak_distance_pct"].between(
            -8,
            strategy_config.chip_launch_max_peak_distance_pct,
            inclusive="both",
        )
        & (frame["return_10d_pct"] <= strategy_config.chip_launch_max_return_10d_pct)
    )
    rebound_location = (
        (frame["position_250"] <= strategy_config.chip_rebound_max_position_250)
        & (
            (frame["adaptive_base_position_120"] <= 0.92)
            | (frame["adaptive_base_drawdown_120_pct"] <= -5)
            | (frame["adaptive_pre_base_decline_60_pct"] <= -3)
        )
    )
    rebound_platform = (
        frame["adaptive_base_high"].notna()
        & (frame["adaptive_base_offset"] >= 8)
        & (frame["adaptive_base_width_pct"] <= strategy_config.chip_rebound_max_base_width_pct)
        & (frame["adaptive_base_return_pct"].abs() <= 25)
    )
    rebound_chip_peak = (
        (frame["chip_70_width_pct"] <= strategy_config.chip_rebound_max_70_width_pct)
        & (frame["chip_peak_position"] <= strategy_config.chip_rebound_max_peak_position)
        & (
            frame["chip_peak_band_share_pct"]
            >= strategy_config.chip_rebound_min_peak_band_share_pct
        )
    )
    rebound_move = (
        (
            (frame["adaptive_rebound_price_action"] == 1)
            | (
                (frame["adaptive_base_offset"] >= 8)
                & (
                    frame["return_5d_pct"]
                    >= strategy_config.chip_rebound_min_return_5d_pct
                )
                & (frame["adaptive_trend_votes"] >= 2)
            )
        )
        & frame["chip_peak_distance_pct"].between(
            -10,
            strategy_config.chip_rebound_max_peak_distance_pct,
            inclusive="both",
        )
        & (frame["return_20d_pct"] <= strategy_config.chip_rebound_max_return_20d_pct)
    )
    rebound_risk_ok = (
        (
            frame["distribution_day_count_5"]
            <= strategy_config.chip_max_distribution_days_5 + 1
        )
        & ~(
            (frame["breakout_failed_fast"] == 1)
            & (frame["distance_from_adaptive_base_high_pct"] < -10)
        )
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
        "chip_base_rebound": common
        + [
            ("rebound_location", rebound_location),
            ("recent_historical_platform", rebound_platform),
            ("rebound_chip_peak", rebound_chip_peak),
            ("rebound_risk_control", rebound_risk_ok),
            ("post_platform_rebound", rebound_move),
            (
                "chip_base_rebound_score",
                frame["chip_base_rebound_score"]
                >= strategy_config.chip_base_rebound_score_min,
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
        model_score = {
            "chip_base_ready": "chip_base_ready_score",
            "chip_base_launch": "chip_base_launch_score",
            "chip_base_rebound": "chip_base_rebound_score",
        }[signal]
        selected = selected.sort_values(
            [model_score, "chip_peak_band_share_pct", "amount_ma20_prev"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
        selected["rank"] = np.arange(1, len(selected) + 1)
        outputs[signal] = selected
    return scored, outputs
