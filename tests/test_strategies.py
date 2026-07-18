import pandas as pd

from ashare_scanner.config import DataConfig, StrategyConfig
from ashare_scanner.strategies import (
    add_factor_scores,
    apply_strategies,
    screening_diagnostics,
)


def _base_row(code: str) -> dict:
    return {
        "code": code,
        "name": code,
        "date": pd.Timestamp("2026-07-15"),
        "bars": 220,
        "open": 9.9,
        "close": 10.0,
        "ma5": 10.1,
        "ma20": 10.1,
        "ma30": 10.15,
        "ma60": 10.2,
        "ma5_slope_3_pct": -0.2,
        "ma20_slope_5_pct": 0.1,
        "ma30_slope_5_pct": 0.1,
        "ma60_slope_5_pct": 0.1,
        "bull_alignment_5_20_30": 0,
        "cross_ma5_up": 0,
        "macd_rising": 0,
        "close_above_prior_high5": 0,
        "prior_high_20": 10.3,
        "position_250": 0.50,
        "range_convergence_ratio_20_60": 0.55,
        "range_width_60_pct": 25.0,
        "resistance_test_count_60": 3,
        "return_120d_pct": 18.0,
        "return_20d_pct": 8.0,
        "return_60d_pct": 15.0,
        "return_10d_pct": 5.0,
        "extension_ma20_pct": -1.0,
        "extension_ma60_pct": -2.0,
        "up_down_volume_ratio_10": 1.40,
        "obv_slope_10_pct": 2.0,
        "vol_ratio_20": 0.85,
        "turnover_ratio_20": 1.20,
        "low_position_turnover_peak_count_60": 2,
        "cost_concentration_60_pct": 8.0,
        "distance_to_cost_center_pct": 3.0,
        "atr_contraction_ratio": 0.82,
        "volume_contraction_5_20": 0.84,
        "bb_contraction_ratio": 0.80,
        "false_break_recovery_count_60": 1,
        "support_acceptance_count_60": 2,
        "upper_probe_count_60": 2,
        "contraction_then_expansion": 0,
        "pullback_volume_ratio_5": 0.75,
        "rs20_percentile": 0.80,
        "rs60_percentile": 0.75,
        "rs20_excess_pct": 5.0,
        "amount_ma20_prev": 120_000_000.0,
        "one_price_limit": 0,
        "distribution_day_count_5": 0,
        "breakout_failed_fast": 0,
        "breakout_stall_distribution": 0,
        "benchmark_risk_ok": 1,
        "pct_chg": 1.0,
    }


def _main_wave_row(code: str) -> dict:
    row = _base_row(code)
    row.update(
        {
            "open": 11.6,
            "close": 12.0,
            "ma5": 11.8,
            "ma20": 11.2,
            "ma30": 10.8,
            "ma60": 10.0,
            "ma5_slope_3_pct": 2.0,
            "ma20_slope_5_pct": 2.0,
            "ma30_slope_5_pct": 1.5,
            "ma60_slope_5_pct": 1.0,
            "bull_alignment_5_20_30": 1,
            "cross_ma5_up": 1,
            "macd_rising": 1,
            "close_above_prior_high5": 1,
            "position_250": 0.90,
            "return_10d_pct": 16.0,
            "return_20d_pct": 24.0,
            "extension_ma20_pct": 7.1,
            "extension_ma60_pct": 20.0,
            "vol_ratio_20": 1.6,
        }
    )
    return row


def test_two_models_select_intended_rows_without_external_data():
    frame = pd.DataFrame([_base_row("000001"), _main_wave_row("000002")])
    _, signals = apply_strategies(frame, DataConfig(), StrategyConfig())
    assert signals["accumulation_late"]["code"].tolist() == ["000001"]
    assert signals["main_wave"]["code"].tolist() == ["000002"]


def test_optional_funding_evidence_adds_score_and_reduction_penalizes():
    positive = _base_row("000001")
    positive.update(
        {
            "main_net_inflow_ratio_pct": 25.0,
            "institution_net_buy_amount": 60_000_000,
            "margin_balance_growth_3d_pct": 12.0,
            "shareholder_count_change_pct": -6.0,
            "sector_profit_growth_median_pct": 25.0,
            "industry_risk_ok": 1,
            "has_reduction_plan": 0,
        }
    )
    reduction = {**positive, "code": "000002", "has_reduction_plan": 1}
    scored = add_factor_scores(pd.DataFrame([positive, reduction]))
    assert scored.loc[0, "score_funding"] == 10
    assert scored.loc[1, "score_funding"] == 5
    assert scored.loc[0, "accumulation_score"] > scored.loc[1, "accumulation_score"]


def test_scores_are_capped_at_100():
    row = _main_wave_row("000001")
    row.update(
        {
            "main_net_inflow_ratio_pct": 99,
            "institution_net_buy_amount": 1_000_000_000,
            "margin_balance_growth_3d_pct": 30,
            "shareholder_count_change_pct": -20,
            "sector_profit_growth_median_pct": 50,
            "industry_risk_ok": 1,
        }
    )
    scored = add_factor_scores(pd.DataFrame([row]))
    assert 0 <= scored.loc[0, "score_total"] <= 100
    assert 0 <= scored.loc[0, "main_wave_score"] <= 100


def test_funnel_final_count_matches_each_signal_output():
    frame = pd.DataFrame([_base_row("000001"), _main_wave_row("000002")])
    data_config = DataConfig()
    strategy_config = StrategyConfig()
    scored, signals = apply_strategies(frame, data_config, strategy_config)
    funnel, _ = screening_diagnostics(scored, data_config, strategy_config)
    final_counts = funnel.groupby("signal").tail(1).set_index("signal")["remaining_count"]
    for signal, selected in signals.items():
        assert final_counts[signal] == len(selected)