import pandas as pd

from ashare_scanner.config import DataConfig, StrategyConfig
from ashare_scanner.strategies import apply_strategies, screening_diagnostics


def _row(code: str, launch: bool) -> dict:
    return {
        "code": code,
        "name": code,
        "date": pd.Timestamp("2026-08-07"),
        "bars": 220,
        "open": 17.2 if launch else 15.0,
        "close": 18.0 if launch else 15.2,
        "amount_ma20_prev": 100_000_000.0,
        "one_price_limit": 0,
        "position_250": 0.25 if launch else 0.10,
        "base_high_20_pre3": 15.6,
        "base_drawdown_from_120_high_pct": -42.0,
        "pre_base_decline_60_pct": -28.0,
        "base_position_120_pre3": 0.08,
        "base_width_20_pre3_pct": 12.0,
        "base_return_20_pre3_pct": -2.0,
        "base_turnover_sum_20_pre3_pct": 55.0,
        "chip_model_coverage_pct": 97.0,
        "chip_peak_price": 15.1,
        "chip_peak_distance_pct": 19.2 if launch else 0.7,
        "chip_peak_band_share_pct": 48.0,
        "chip_70_width_pct": 18.0,
        "chip_low_zone_share_pct": 72.0,
        "chip_overhead_ratio_pct": 18.0 if launch else 48.0,
        "chip_peak_position": 0.10,
        "chip_peak_dominance": 2.5,
        "chip_significant_peak_count": 1,
        "early_launch_price_action": int(launch),
        "return_10d_pct": 22.0 if launch else 2.0,
        "return_5d_pct": 18.0 if launch else 1.0,
        "vol_ratio_20": 1.6 if launch else 0.9,
        "macd_rising": int(launch),
        "distribution_day_count_5": 0,
        "breakout_failed_fast": 0,
        "breakout_stall_distribution": 0,
        "benchmark_risk_ok": 1,
    }


def test_two_chip_base_phases_select_intended_rows():
    frame = pd.DataFrame([_row("000001", False), _row("000002", True)])
    _, signals = apply_strategies(frame, DataConfig(), StrategyConfig())
    assert signals["chip_base_ready"]["code"].tolist() == ["000001"]
    assert signals["chip_base_launch"]["code"].tolist() == ["000002"]


def test_long_term_ma_alignment_is_not_required_for_early_launch():
    frame = pd.DataFrame([_row("000001", True)])
    _, signals = apply_strategies(frame, DataConfig(), StrategyConfig())
    assert signals["chip_base_launch"]["code"].tolist() == ["000001"]


def test_flat_stock_without_prior_decline_is_rejected():
    row = _row("000001", False)
    row["base_drawdown_from_120_high_pct"] = -5.0
    row["pre_base_decline_60_pct"] = 1.0
    _, signals = apply_strategies(pd.DataFrame([row]), DataConfig(), StrategyConfig())
    assert signals["chip_base_ready"].empty


def test_diffuse_or_high_chip_profile_is_rejected():
    row = _row("000001", True)
    row["chip_70_width_pct"] = 40.0
    row["chip_peak_position"] = 0.75
    _, signals = apply_strategies(pd.DataFrame([row]), DataConfig(), StrategyConfig())
    assert signals["chip_base_launch"].empty


def test_late_extended_launch_is_rejected():
    row = _row("000001", True)
    row["chip_peak_distance_pct"] = 70.0
    row["return_10d_pct"] = 60.0
    _, signals = apply_strategies(pd.DataFrame([row]), DataConfig(), StrategyConfig())
    assert signals["chip_base_launch"].empty


def test_scores_are_capped_and_funnel_matches_outputs():
    frame = pd.DataFrame([_row("000001", False), _row("000002", True)])
    data_config = DataConfig()
    strategy_config = StrategyConfig()
    scored, signals = apply_strategies(frame, data_config, strategy_config)
    assert scored["score_total"].between(0, 100).all()
    funnel, _ = screening_diagnostics(scored, data_config, strategy_config)
    final_counts = funnel.groupby("signal").tail(1).set_index("signal")["remaining_count"]
    for signal, selected in signals.items():
        assert final_counts[signal] == len(selected)
