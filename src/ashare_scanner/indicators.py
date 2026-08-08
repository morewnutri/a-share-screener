from __future__ import annotations

import numpy as np
import pandas as pd

from .chips import compute_modeled_cyq


PRICE_COLUMNS = ("open", "high", "low", "close", "volume", "amount")


def calc_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - 100 / (1 + rs)
    rsi = rsi.mask((avg_loss == 0) & (avg_gain > 0), 100.0)
    rsi = rsi.mask((avg_loss == 0) & (avg_gain == 0), 50.0)
    return rsi


def _recent_breakout_features(frame: pd.DataFrame, lookback: int = 10) -> None:
    levels = np.full(len(frame), np.nan)
    bars_since = np.full(len(frame), np.nan)
    event_lows = np.full(len(frame), np.nan)
    event_closes = np.full(len(frame), np.nan)
    last_index: int | None = None
    last_level = np.nan
    last_low = np.nan
    last_close = np.nan
    events = frame["breakout_20"].fillna(False).to_numpy(dtype=bool)
    prior_highs = frame["prior_high_20"].to_numpy(dtype=float)
    lows = frame["low"].to_numpy(dtype=float)
    closes = frame["close"].to_numpy(dtype=float)
    for index, event in enumerate(events):
        if event:
            last_index = index
            last_level = prior_highs[index]
            last_low = lows[index]
            last_close = closes[index]
        if last_index is not None and index - last_index <= lookback:
            levels[index] = last_level
            bars_since[index] = index - last_index
            event_lows[index] = last_low
            event_closes[index] = last_close
    frame["recent_breakout_level"] = levels
    frame["bars_since_breakout"] = bars_since
    frame["breakout_event_low"] = event_lows
    frame["breakout_event_close"] = event_closes
    frame["retest_distance_pct"] = (frame["close"] / frame["recent_breakout_level"] - 1) * 100
    frame["retest_touch"] = (
        (frame["low"] <= frame["recent_breakout_level"] * 1.02)
        & (frame["close"] >= frame["recent_breakout_level"] * 0.985)
    ).astype(int)
    early_retest = frame["bars_since_breakout"].between(1, 3, inclusive="both")
    frame["breakout_failed_fast"] = (
        early_retest
        & (
            (frame["close"] < frame["recent_breakout_level"] * 0.98)
            | (frame["close"] < frame["breakout_event_low"])
        )
    ).astype(int)


def _cost_concentration_proxy(frame: pd.DataFrame, window: int = 60) -> None:
    typical = (frame["high"] + frame["low"] + frame["close"]) / 3
    rolling_volume = frame["volume"].rolling(window, min_periods=30).sum()
    first_moment = (typical * frame["volume"]).rolling(window, min_periods=30).sum()
    second_moment = ((typical**2) * frame["volume"]).rolling(window, min_periods=30).sum()
    center = first_moment / rolling_volume.replace(0, np.nan)
    variance = (second_moment / rolling_volume.replace(0, np.nan) - center**2).clip(lower=0)
    frame["cost_center_60"] = center
    frame["cost_concentration_60_pct"] = variance.pow(0.5) / center * 100
    frame["distance_to_cost_center_pct"] = (frame["close"] / center - 1) * 100


def compute_indicators(
    history: pd.DataFrame,
    *,
    include_chips: bool = True,
    chip_latest_only: bool = False,
) -> pd.DataFrame:
    frame = history.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    for column in PRICE_COLUMNS:
        if column not in frame:
            frame[column] = np.nan
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in ("pct_chg", "turnover", "amplitude"):
        if column not in frame:
            frame[column] = np.nan
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["date", "open", "high", "low", "close", "volume"])
    frame = frame.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)

    previous_close = frame["close"].shift(1)
    calculated_pct = frame["close"].pct_change(fill_method=None) * 100
    calculated_amplitude = (frame["high"] - frame["low"]) / previous_close * 100
    frame["pct_chg"] = frame["pct_chg"].fillna(calculated_pct)
    frame["amplitude"] = frame["amplitude"].fillna(calculated_amplitude)

    for window in (5, 10, 20, 30, 60, 120):
        frame[f"ma{window}"] = frame["close"].rolling(window, min_periods=window).mean()

    ema_fast = frame["close"].ewm(span=12, adjust=False).mean()
    ema_slow = frame["close"].ewm(span=26, adjust=False).mean()
    frame["dif"] = ema_fast - ema_slow
    frame["dea"] = frame["dif"].ewm(span=9, adjust=False).mean()
    frame["macd_hist"] = (frame["dif"] - frame["dea"]) * 2
    frame["macd_rising"] = (frame["macd_hist"] > frame["macd_hist"].shift(1)).astype(int)
    frame["dif_slope_3_pct"] = (frame["dif"] - frame["dif"].shift(3)) / frame["close"] * 100
    frame["rsi14"] = calc_rsi(frame["close"], 14)

    frame["prior_high_5"] = frame["high"].rolling(5, min_periods=5).max().shift(1)
    frame["prior_high_20"] = frame["high"].rolling(20, min_periods=20).max().shift(1)
    frame["prior_low_20"] = frame["low"].rolling(20, min_periods=20).min().shift(1)
    frame["prior_high_60"] = frame["high"].rolling(60, min_periods=60).max().shift(1)
    frame["dist_to_prior_high20_pct"] = (frame["close"] / frame["prior_high_20"] - 1) * 100
    frame["dist_to_prior_high60_pct"] = (frame["close"] / frame["prior_high_60"] - 1) * 100
    frame["breakout_20"] = (
        (frame["close"] > frame["prior_high_20"])
        & (frame["close"].shift(1) <= frame["prior_high_20"].shift(1))
    ).astype(int)
    frame["close_above_prior_high5"] = (frame["close"] > frame["prior_high_5"]).astype(int)

    frame["vma20_prev"] = frame["volume"].rolling(20, min_periods=20).mean().shift(1)
    frame["amount_ma20_prev"] = frame["amount"].rolling(20, min_periods=20).mean().shift(1)
    frame["vol_ratio_20"] = frame["volume"] / frame["vma20_prev"]
    recent_volume = frame["volume"].rolling(5, min_periods=5).mean()
    preceding_volume = frame["volume"].shift(5).rolling(20, min_periods=20).mean()
    frame["volume_contraction_5_20"] = recent_volume / preceding_volume
    frame["turnover_ma20_prev"] = frame["turnover"].rolling(20, min_periods=10).mean().shift(1)
    frame["turnover_ratio_20"] = frame["turnover"] / frame["turnover_ma20_prev"].replace(0, np.nan)

    true_range = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - previous_close).abs(),
            (frame["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    frame["atr14"] = true_range.rolling(14, min_periods=14).mean()
    frame["atr_pct"] = frame["atr14"] / frame["close"] * 100
    frame["atr_pct_mean20_prev"] = frame["atr_pct"].rolling(20, min_periods=20).mean().shift(1)
    frame["atr_contraction_ratio"] = frame["atr_pct"] / frame["atr_pct_mean20_prev"]

    std20 = frame["close"].rolling(20, min_periods=20).std(ddof=0)
    frame["bb_width_pct"] = std20 * 4 / frame["ma20"] * 100
    frame["bb_width_mean20_prev"] = frame["bb_width_pct"].rolling(20, min_periods=20).mean().shift(1)
    frame["bb_contraction_ratio"] = frame["bb_width_pct"] / frame["bb_width_mean20_prev"]
    prior_compressed = frame["atr_contraction_ratio"].shift(1) <= 0.90
    expanding_now = frame["atr_pct"] >= frame["atr_pct"].shift(1) * 1.10
    frame["contraction_then_expansion"] = (prior_compressed & expanding_now).astype(int)

    direction = np.sign(frame["close"].diff()).fillna(0)
    up_volume = frame["volume"].where(direction > 0, 0.0).rolling(10, min_periods=10).sum()
    down_volume = frame["volume"].where(direction < 0, 0.0).rolling(10, min_periods=10).sum()
    frame["up_down_volume_ratio_10"] = np.where(
        down_volume > 0,
        up_volume / down_volume,
        np.where(up_volume > 0, np.inf, np.nan),
    )
    obv = (direction * frame["volume"]).cumsum()
    volume20_sum = frame["volume"].rolling(20, min_periods=20).sum()
    frame["obv_slope_10_pct"] = (obv - obv.shift(10)) / volume20_sum * 100
    down_day = frame["close"] < previous_close
    down_volume_5 = frame["volume"].where(down_day).rolling(5, min_periods=1).mean()
    frame["pullback_volume_ratio_5"] = down_volume_5 / frame["vma20_prev"]

    frame["return_5d_pct"] = frame["close"].pct_change(5, fill_method=None) * 100
    frame["return_10d_pct"] = frame["close"].pct_change(10, fill_method=None) * 100
    frame["return_20d_pct"] = frame["close"].pct_change(20, fill_method=None) * 100
    frame["return_60d_pct"] = frame["close"].pct_change(60, fill_method=None) * 100
    frame["return_120d_pct"] = frame["close"].pct_change(120, fill_method=None) * 100
    frame["extension_ma20_pct"] = (frame["close"] / frame["ma20"] - 1) * 100
    frame["extension_ma60_pct"] = (frame["close"] / frame["ma60"] - 1) * 100
    frame["ma5_slope_3_pct"] = frame["ma5"].pct_change(3, fill_method=None) * 100
    frame["ma20_slope_5_pct"] = frame["ma20"].pct_change(5, fill_method=None) * 100
    frame["ma30_slope_5_pct"] = frame["ma30"].pct_change(5, fill_method=None) * 100
    frame["ma60_slope_5_pct"] = frame["ma60"].pct_change(5, fill_method=None) * 100
    frame["bull_alignment_5_20_30"] = (
        (frame["close"] > frame["ma5"])
        & (frame["ma5"] > frame["ma20"])
        & (frame["ma20"] > frame["ma30"])
        & (frame["close"] > frame["ma60"])
    ).astype(int)
    frame["cross_ma5_up"] = (
        (frame["close"] > frame["ma5"])
        & (frame["close"].shift(1) <= frame["ma5"].shift(1))
        & (frame["close"] > frame["open"])
    ).astype(int)

    # Measure the platform before the latest three bars so an early launch does
    # not inflate the consolidation range it is supposed to break from.
    base_close = frame["close"].shift(3)
    base_high = frame["high"].shift(3).rolling(20, min_periods=20).max()
    base_low = frame["low"].shift(3).rolling(20, min_periods=20).min()
    base_mean = frame["close"].shift(3).rolling(20, min_periods=20).mean()
    base_std = frame["close"].shift(3).rolling(20, min_periods=20).std(ddof=0)
    frame["base_high_20_pre3"] = base_high
    frame["base_low_20_pre3"] = base_low
    frame["base_mid_20_pre3"] = (base_high + base_low) / 2
    frame["base_width_20_pre3_pct"] = (base_high / base_low.replace(0, np.nan) - 1) * 100
    frame["base_close_cv_20_pre3_pct"] = base_std / base_mean * 100
    frame["base_return_20_pre3_pct"] = (base_close / frame["close"].shift(22) - 1) * 100
    frame["base_turnover_sum_20_pre3_pct"] = frame["turnover"].shift(3).rolling(
        20, min_periods=15
    ).sum()
    prior_high_120_pre3 = frame["high"].shift(3).rolling(120, min_periods=80).max()
    prior_low_120_pre3 = frame["low"].shift(3).rolling(120, min_periods=80).min()
    frame["base_drawdown_from_120_high_pct"] = (base_close / prior_high_120_pre3 - 1) * 100
    frame["base_position_120_pre3"] = (
        frame["base_mid_20_pre3"] - prior_low_120_pre3
    ) / (prior_high_120_pre3 - prior_low_120_pre3).replace(0, np.nan)
    frame["pre_base_decline_60_pct"] = (
        frame["close"].shift(22) / frame["close"].shift(82) - 1
    ) * 100
    frame["distance_from_base_high_pct"] = (frame["close"] / base_high - 1) * 100
    frame["distance_from_base_mid_pct"] = (
        frame["close"] / frame["base_mid_20_pre3"] - 1
    ) * 100
    frame["early_launch_price_action"] = (
        (frame["return_5d_pct"].between(3, 35, inclusive="both"))
        & (frame["close"] > frame["ma5"])
        & (frame["ma5_slope_3_pct"] > 0)
        & (frame["distance_from_base_mid_pct"] >= 4)
        & (frame["distance_from_base_high_pct"].between(-4, 35, inclusive="both"))
    ).astype(int)

    rolling_ranges: dict[int, pd.Series] = {}
    for window in (10, 20, 60, 120):
        high = frame["high"].rolling(window, min_periods=window).max()
        low = frame["low"].rolling(window, min_periods=window).min()
        rolling_ranges[window] = (high / low.replace(0, np.nan) - 1) * 100
        frame[f"range_width_{window}_pct"] = rolling_ranges[window]
    frame["range_convergence_ratio_20_60"] = rolling_ranges[20] / rolling_ranges[60].replace(0, np.nan)

    for window, minimum in ((120, 80), (250, 120)):
        high = frame["high"].rolling(window, min_periods=minimum).max()
        low = frame["low"].rolling(window, min_periods=minimum).min()
        frame[f"position_{window}"] = (frame["close"] - low) / (high - low).replace(0, np.nan)

    bar_range = (frame["high"] - frame["low"]).replace(0, np.nan)
    body_high = pd.concat([frame["open"], frame["close"]], axis=1).max(axis=1)
    body_low = pd.concat([frame["open"], frame["close"]], axis=1).min(axis=1)
    frame["upper_wick_ratio"] = (frame["high"] - body_high) / bar_range
    frame["lower_wick_ratio"] = (body_low - frame["low"]) / bar_range
    upper_probe = (
        (frame["upper_wick_ratio"] >= 0.40)
        & (frame["high"] >= frame["prior_high_20"] * 0.985)
        & (frame["vol_ratio_20"] >= 0.80)
    )
    false_break_recovery = (
        (frame["low"] < frame["prior_low_20"] * 0.995)
        & (frame["close"] > frame["prior_low_20"])
        & (frame["lower_wick_ratio"] >= 0.30)
    )
    support_acceptance = (
        (frame["low"] <= frame["prior_low_20"] * 1.03)
        & (frame["lower_wick_ratio"] >= 0.30)
        & (frame["close"] >= frame["open"])
        & (frame["vol_ratio_20"] >= 0.75)
    )
    resistance_test = (
        (frame["high"] >= frame["prior_high_20"] * 0.985)
        & (frame["high"] <= frame["prior_high_20"] * 1.03)
    )
    frame["upper_probe"] = upper_probe.astype(int)
    frame["false_break_recovery"] = false_break_recovery.astype(int)
    frame["support_acceptance"] = support_acceptance.astype(int)
    frame["resistance_test"] = resistance_test.astype(int)
    frame["upper_probe_count_60"] = upper_probe.rolling(60, min_periods=20).sum()
    frame["false_break_recovery_count_60"] = false_break_recovery.rolling(60, min_periods=20).sum()
    frame["support_acceptance_count_60"] = support_acceptance.rolling(60, min_periods=20).sum()
    frame["resistance_test_count_60"] = resistance_test.rolling(60, min_periods=20).sum()

    low_turnover_peak = (
        (frame["position_120"] <= 0.60)
        & (frame["turnover_ratio_20"] >= 1.50)
        & (frame["close"] >= frame["open"])
    )
    frame["low_position_turnover_peak"] = low_turnover_peak.astype(int)
    frame["low_position_turnover_peak_count_60"] = low_turnover_peak.rolling(60, min_periods=20).sum()

    frame["amplitude_ma20_prev"] = frame["amplitude"].rolling(20, min_periods=10).mean().shift(1)
    distribution = (
        (frame["turnover_ratio_20"] >= 1.80)
        & (frame["amplitude"] >= frame["amplitude_ma20_prev"] * 1.40)
        & (frame["pct_chg"] <= 1.0)
        & (frame["upper_wick_ratio"] >= 0.25)
    )
    frame["distribution_day"] = distribution.astype(int)
    frame["distribution_day_count_5"] = distribution.rolling(5, min_periods=1).sum()

    frame = frame.copy()
    _cost_concentration_proxy(frame)
    _recent_breakout_features(frame)
    frame["breakout_stall_distribution"] = (
        frame["bars_since_breakout"].between(1, 3, inclusive="both")
        & (frame["turnover_ratio_20"] >= 1.50)
        & (frame["amplitude"] >= frame["amplitude_ma20_prev"] * 1.30)
        & (frame["close"] <= frame["breakout_event_close"] * 1.01)
    ).astype(int)

    frame["range_position_10"] = (
        frame["close"] - frame["low"].rolling(10, min_periods=10).min()
    ) / (
        frame["high"].rolling(10, min_periods=10).max()
        - frame["low"].rolling(10, min_periods=10).min()
    ).replace(0, np.nan)
    frame["one_price_limit"] = (
        np.isclose(frame["high"], frame["low"], rtol=0, atol=0.001)
        & (frame["pct_chg"] >= 9.5)
    ).astype(int)
    if not include_chips:
        return frame
    chip_features = compute_modeled_cyq(frame, latest_only=chip_latest_only)
    return pd.concat(
        [frame.reset_index(drop=True), chip_features.reset_index(drop=True)],
        axis=1,
    )


def latest_snapshot(indicators: pd.DataFrame, code: str, name: str, source: str) -> dict:
    last = indicators.iloc[-1].to_dict()
    last.update(
        {
            "code": str(code).zfill(6),
            "name": name,
            "source": source,
            "date": pd.Timestamp(last["date"]).strftime("%Y-%m-%d"),
            "bars": int(len(indicators)),
        }
    )
    return last
