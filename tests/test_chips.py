import numpy as np
import pandas as pd

from ashare_scanner.chips import CHIP_MODEL_NAME, compute_modeled_cyq
from ashare_scanner.indicators import compute_indicators


def _decline_base_launch(launch: bool) -> pd.DataFrame:
    decline_count = 125
    base_count = 25
    decline = np.linspace(30.0, 15.3, decline_count)
    base = 15.0 + np.sin(np.arange(base_count) * 1.7) * 0.35
    tail = np.array([15.8, 16.8, 18.0]) if launch else np.array([15.1, 15.0, 15.2])
    close = np.r_[decline, base, tail]
    open_ = close.copy()
    if launch:
        open_[-3:] = close[-3:] * 0.98
    high = np.maximum(open_, close) * 1.02
    low = np.minimum(open_, close) * 0.98
    volume = np.r_[
        np.full(decline_count, 1_000_000.0),
        np.full(base_count, 2_000_000.0),
        np.full(3, 3_000_000.0),
    ]
    turnover = np.r_[
        np.full(decline_count, 2.0),
        np.full(base_count, 4.0),
        np.full(3, 6.0),
    ]
    return pd.DataFrame(
        {
            "date": pd.bdate_range("2025-01-01", periods=len(close)),
            "open": open_,
            "close": close,
            "high": high,
            "low": low,
            "volume": volume,
            "amount": volume * close,
            "turnover": turnover,
            "pct_chg": pd.Series(close).pct_change().fillna(0) * 100,
            "amplitude": (high - low) / close * 100,
        }
    )


def test_modeled_cyq_forms_visible_low_peak_after_base_turnover():
    profile = compute_modeled_cyq(_decline_base_launch(False)).iloc[-1]
    assert profile["chip_model"] == CHIP_MODEL_NAME
    assert 14.0 <= profile["chip_peak_price"] <= 16.0
    assert profile["chip_peak_band_share_pct"] >= 40
    assert profile["chip_low_zone_share_pct"] >= 60
    assert profile["chip_peak_position"] <= 0.20
    assert profile["chip_model_coverage_pct"] >= 90


def test_latest_only_keeps_the_same_final_profile():
    history = _decline_base_launch(False)
    full = compute_modeled_cyq(history)
    latest_only = compute_modeled_cyq(history, latest_only=True)
    assert latest_only.iloc[:-1]["chip_peak_price"].isna().all()
    assert latest_only.iloc[:-1]["chip_model"].eq("").all()
    pd.testing.assert_series_equal(
        full.iloc[-1],
        latest_only.iloc[-1],
        check_names=False,
    )


def test_rolling_window_removes_prices_older_than_lookback():
    common = _decline_base_launch(False).tail(120).reset_index(drop=True)
    old_low = common.head(30).copy()
    old_high = common.head(30).copy()
    for frame, price in ((old_low, 5.0), (old_high, 80.0)):
        frame[["open", "close"]] = price
        frame["high"] = price * 1.02
        frame["low"] = price * 0.98
        frame["amount"] = frame["volume"] * price
    low_history = pd.concat([old_low, common], ignore_index=True)
    high_history = pd.concat([old_high, common], ignore_index=True)
    low_profile = compute_modeled_cyq(
        low_history,
        lookback=120,
        latest_only=True,
    ).iloc[-1]
    high_profile = compute_modeled_cyq(
        high_history,
        lookback=120,
        latest_only=True,
    ).iloc[-1]
    pd.testing.assert_series_equal(low_profile, high_profile, check_names=False)


def test_launch_is_measured_against_prelaunch_platform_and_peak():
    indicators = compute_indicators(_decline_base_launch(True))
    latest = indicators.iloc[-1]
    assert latest["base_width_20_pre3_pct"] <= 15
    assert latest["base_drawdown_from_120_high_pct"] <= -30
    assert latest["early_launch_price_action"] == 1
    assert 10 <= latest["chip_peak_distance_pct"] <= 30


def test_indicators_can_skip_chip_model_for_benchmark_history():
    indicators = compute_indicators(_decline_base_launch(False), include_chips=False)
    assert "return_20d_pct" in indicators
    assert "chip_peak_price" not in indicators
