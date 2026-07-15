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


def test_indicator_frame_has_retest_state(trending_history):
    indicators = compute_indicators(trending_history)
    required = {
        "recent_breakout_level",
        "bars_since_breakout",
        "retest_distance_pct",
        "atr_contraction_ratio",
        "bb_contraction_ratio",
    }
    assert required.issubset(indicators.columns)

