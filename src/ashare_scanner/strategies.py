from __future__ import annotations

from collections.abc import Mapping
from functools import reduce
from operator import and_

import numpy as np
import pandas as pd

from .config import DataConfig, StrategyConfig


SIGNAL_ORDER = ("accumulation_late", "main_wave")
OPTIONAL_SCORE_FIELDS = (
    "main_net_inflow_ratio_pct",
    "main_net_inflow_amount",
    "main_net_inflow_positive_days_5",
    "institution_net_buy_amount",
    "margin_balance_growth_3d_pct",
    "shareholder_count_change_pct",
    "sector_profit_growth_median_pct",
    "has_reduction_plan",
    "industry_risk_ok",
)


def _numeric(result: pd.DataFrame, column: str, default: float = np.nan) -> pd.Series:
    if column not in result:
        result[column] = default
    return pd.to_numeric(result[column], errors="coerce")


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
        result["float_cap_percentile"] = groups["float_market_cap"].rank(pct=True, method="average")
    else:
        result["float_cap_percentile"] = np.nan
    return result


def _funding_score(result: pd.DataFrame) -> pd.Series:
    for field in OPTIONAL_SCORE_FIELDS:
        _numeric(result, field)
    positive = (
        (result["main_net_inflow_ratio_pct"] >= 20).astype(int) * 4
        + (result["institution_net_buy_amount"] >= 50_000_000).astype(int) * 3
        + (result["margin_balance_growth_3d_pct"] >= 10).astype(int) * 2
        + (result["shareholder_count_change_pct"] <= -5).astype(int) * 2
        + (result["sector_profit_growth_median_pct"] >= 20).astype(int) * 2
        + (result["main_net_inflow_positive_days_5"] >= 5).astype(int) * 2
        + (result["industry_risk_ok"] >= 1).astype(int)
    ).clip(upper=10)
    reduction_penalty = (result["has_reduction_plan"] >= 1).astype(int) * 5
    return (positive - reduction_penalty).clip(lower=-5, upper=10)


def add_factor_scores(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    _numeric(result, "float_cap_percentile")
    result["score_funding"] = _funding_score(result)

    result["acc_score_structure"] = (
        (result["position_250"] <= 0.75).astype(int) * 6
        + (result["range_convergence_ratio_20_60"] <= 0.65).astype(int) * 6
        + (result["range_width_60_pct"] <= 35).astype(int) * 4
        + (result["resistance_test_count_60"] >= 2).astype(int) * 5
        + result["return_120d_pct"].between(-15, 55, inclusive="both").astype(int) * 4
    ).clip(upper=25)
    result["acc_score_volume_chip"] = (
        (result["up_down_volume_ratio_10"] >= 1.15).astype(int) * 5
        + (result["obv_slope_10_pct"] > 0).astype(int) * 4
        + result["vol_ratio_20"].between(0.65, 2.50, inclusive="both").astype(int) * 3
        + result["turnover_ratio_20"].between(0.70, 2.50, inclusive="both").astype(int) * 3
        + (result["low_position_turnover_peak_count_60"] >= 1).astype(int) * 4
        + (result["cost_concentration_60_pct"] <= 12).astype(int) * 4
        + result["distance_to_cost_center_pct"].between(-8, 12, inclusive="both").astype(int)
        + (result["float_cap_percentile"] <= 0.70).astype(int)
    ).clip(upper=25)
    contraction_count = (
        (result["atr_contraction_ratio"] <= 0.92).astype(int)
        + (result["volume_contraction_5_20"] <= 0.92).astype(int)
        + (result["bb_contraction_ratio"] <= 0.92).astype(int)
    )
    result["acc_score_behavior"] = (
        (contraction_count >= 2).astype(int) * 8
        + (result["false_break_recovery_count_60"] >= 1).astype(int) * 4
        + (result["support_acceptance_count_60"] >= 1).astype(int) * 4
        + (result["upper_probe_count_60"] >= 1).astype(int) * 2
        + (result["contraction_then_expansion"] == 1).astype(int) * 2
    ).clip(upper=20)
    result["acc_score_independence"] = (
        (result["rs20_percentile"] >= 0.65).astype(int) * 5
        + (result["rs60_percentile"] >= 0.60).astype(int) * 4
        + (result["rs20_excess_pct"] > 3).astype(int) * 3
    ).clip(upper=12)
    result["acc_score_risk"] = (
        (result["distribution_day_count_5"] == 0).astype(int) * 5
        + (result["breakout_failed_fast"] == 0).astype(int) * 3
    ).clip(upper=8)
    result["accumulation_evidence_groups"] = (
        (result["acc_score_structure"] >= 10).astype(int)
        + (result["acc_score_volume_chip"] >= 9).astype(int)
        + (result["acc_score_behavior"] >= 7).astype(int)
        + (result["acc_score_independence"] >= 4).astype(int)
        + (result["score_funding"] >= 3).astype(int)
    )
    result["accumulation_score"] = (
        result[
            [
                "acc_score_structure",
                "acc_score_volume_chip",
                "acc_score_behavior",
                "acc_score_independence",
                "acc_score_risk",
                "score_funding",
            ]
        ]
        .sum(axis=1)
        .clip(lower=0, upper=100)
    )

    result["wave_score_trend"] = (
        (result["close"] > result["ma60"]).astype(int) * 5
        + (result["ma60_slope_5_pct"] > 0).astype(int) * 5
        + (result["ma20"] > result["ma30"]).astype(int) * 5
        + (result["ma30_slope_5_pct"] > 0).astype(int) * 4
        + (result["bull_alignment_5_20_30"] == 1).astype(int) * 7
        + (result["ma5_slope_3_pct"] > 0).astype(int) * 4
    ).clip(upper=30)
    result["wave_score_trigger"] = (
        (result["cross_ma5_up"] == 1).astype(int) * 9
        + ((result["close"] > result["open"]) & (result["close"] > result["ma5"])).astype(int) * 4
        + (result["close_above_prior_high5"] == 1).astype(int) * 4
        + (result["macd_rising"] == 1).astype(int) * 3
    ).clip(upper=20)
    result["wave_score_volume"] = (
        (result["up_down_volume_ratio_10"] >= 1.20).astype(int) * 5
        + result["vol_ratio_20"].between(0.90, 3.00, inclusive="both").astype(int) * 4
        + (result["pullback_volume_ratio_5"] <= 0.90).astype(int) * 5
        + (result["obv_slope_10_pct"] > 0).astype(int) * 4
    ).clip(upper=18)
    result["wave_score_strength"] = (
        (result["rs20_percentile"] >= 0.70).astype(int) * 5
        + (result["rs60_percentile"] >= 0.65).astype(int) * 4
        + (result["rs20_excess_pct"] > 4).astype(int) * 3
    ).clip(upper=12)
    benchmark_ok = _numeric(result, "benchmark_risk_ok", 1.0).fillna(1.0) >= 1
    result["wave_score_market_risk"] = (
        benchmark_ok.astype(int) * 4
        + (result["extension_ma20_pct"] <= 12).astype(int) * 3
        + (result["return_10d_pct"] <= 20).astype(int) * 2
        + (result["distribution_day_count_5"] == 0).astype(int)
    ).clip(upper=10)
    result["main_wave_score"] = (
        result[
            [
                "wave_score_trend",
                "wave_score_trigger",
                "wave_score_volume",
                "wave_score_strength",
                "wave_score_market_risk",
                "score_funding",
            ]
        ]
        .sum(axis=1)
        .clip(lower=0, upper=100)
    )
    result["score_total"] = result[["accumulation_score", "main_wave_score"]].max(axis=1)
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
        & frame["prior_high_20"].notna()
        & frame["ma60"].notna()
        & frame["position_250"].notna()
        & frame["rs20_percentile"].notna()
    )
    common = [
        ("history_bars", history_ready),
        ("liquidity", liquid),
        ("tradable_features", features_ready),
    ]
    accumulation_risk = (
        (frame["breakout_failed_fast"] == 0)
        & (frame["breakout_stall_distribution"] == 0)
        & (
            frame["distribution_day_count_5"]
            <= strategy_config.accumulation_max_distribution_days_5
        )
    )
    accumulation_position = (
        (frame["position_250"] <= strategy_config.accumulation_max_position_250)
        & (frame["return_20d_pct"] <= strategy_config.accumulation_max_return_20d_pct)
        & (
            frame["extension_ma60_pct"]
            <= strategy_config.accumulation_max_extension_ma60_pct
        )
    )
    trend_core = (
        (frame["close"] > frame["ma60"])
        & (frame["ma60_slope_5_pct"] > 0)
        & (frame["close"] > frame["ma20"])
        & (frame["ma20"] > frame["ma30"])
    )
    trigger = (
        (frame["close"] > frame["ma5"])
        & (
            (frame["cross_ma5_up"] == 1)
            | ((frame["close"] > frame["open"]) & (frame["macd_rising"] == 1))
        )
    )
    wave_risk = (
        (frame["extension_ma20_pct"] <= strategy_config.main_wave_max_extension_ma20_pct)
        & (frame["return_10d_pct"] <= strategy_config.main_wave_max_return_10d_pct)
        & (frame["distribution_day_count_5"] <= 2)
        & (frame["breakout_failed_fast"] == 0)
    )
    benchmark_ok = _numeric(frame, "benchmark_risk_ok", 1.0).fillna(1.0) >= 1
    return {
        "accumulation_late": common
        + [
            ("accumulation_position", accumulation_position),
            ("accumulation_risk", accumulation_risk),
            (
                "evidence_groups",
                frame["accumulation_evidence_groups"]
                >= strategy_config.accumulation_min_evidence_groups,
            ),
            ("accumulation_score", frame["accumulation_score"] >= strategy_config.accumulation_score_min),
        ],
        "main_wave": common
        + [
            ("trend_resonance", trend_core),
            ("ma5_funding_trigger", trigger),
            ("market_risk", benchmark_ok),
            ("wave_risk", wave_risk),
            ("main_wave_score", frame["main_wave_score"] >= strategy_config.main_wave_score_min),
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
        model_score = "accumulation_score" if signal == "accumulation_late" else "main_wave_score"
        selected = selected.sort_values(
            [model_score, "rs20_percentile", "amount_ma20_prev"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
        selected["rank"] = np.arange(1, len(selected) + 1)
        outputs[signal] = selected
    return scored, outputs