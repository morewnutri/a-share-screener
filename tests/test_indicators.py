import numpy as np
import pandas as pd

from ashare_scanner.indicators import calc_rsi, compute_indicators


def test_rsi_handles_monotonic_and_flat_series():
    rising = calc_rsi(pd.Series(np.arange(1.0, 40.0)))
    flat = calc_rsi(pd.Series(np.ones(40)))
    assert rising.iloc[-1] == 100.0
    assert flat.iloc[-1] == 50.0


def test_prior_high_and_volume_baseline_exclude_current_bar(trending_history):
    frame = trending_history.iloc[:40].copy()
    frame.loc[39, "high"] = 999.0
    frame.loc[39, "volume"] = 999_000_000.0
    indicators = compute_indicators(frame)
    expected_high = frame.loc[19:38, "high"].max()
    expected_volume = frame.loc[19:38, "volume"].mean()
    assert indicators.loc[39, "prior_high_20"] == expected_high
    assert np.isclose(indicators.loc[39, "vma20_prev"], expected_volume)


def test_indicator_frame_has_chip_base_features(trending_history):
    indicators = compute_indicators(trending_history)
    required = {
        "ma30",
        "position_250",
        "range_convergence_ratio_20_60",
        "cost_concentration_60_pct",
        "false_break_recovery_count_60",
        "support_acceptance_count_60",
        "low_position_turnover_peak_count_60",
        "cross_ma5_up",
        "breakout_failed_fast",
        "distribution_day_count_5",
        "base_width_20_pre3_pct",
        "base_drawdown_from_120_high_pct",
        "early_launch_price_action",
        "rebound_base_offset",
        "rebound_base_width_pct",
        "rebound_base_drawdown_120_pct",
        "distance_from_rebound_base_high_pct",
        "rebound_price_action",
        "chip_peak_price",
        "chip_peak_band_share_pct",
        "chip_70_width_pct",
        "chip_low_zone_share_pct",
        "chip_overhead_ratio_pct",
        "chip_model",
    }
    assert required.issubset(indicators.columns)
    assert np.isfinite(indicators.iloc[-1]["cost_concentration_60_pct"])


def test_false_breakdown_recovery_is_detected():
    count = 80
    frame = pd.DataFrame(
        {
            "date": pd.bdate_range("2026-01-01", periods=count),
            "open": np.full(count, 10.0),
            "high": np.full(count, 10.3),
            "low": np.full(count, 9.5),
            "close": np.full(count, 10.0),
            "volume": np.full(count, 1_000_000.0),
            "amount": np.full(count, 10_000_000.0),
            "turnover": np.full(count, 2.0),
            "pct_chg": np.zeros(count),
            "amplitude": np.full(count, 8.0),
        }
    )
    frame.loc[count - 1, ["open", "high", "low", "close"]] = [9.5, 10.2, 8.8, 10.0]
    indicators = compute_indicators(frame)
    assert indicators.iloc[-1]["false_break_recovery"] == 1
