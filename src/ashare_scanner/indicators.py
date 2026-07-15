from __future__ import annotations

import numpy as np
import pandas as pd


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
    last_index: int | None = None
    last_level = np.nan
    events = frame["breakout_20"].fillna(False).to_numpy(dtype=bool)
    prior_highs = frame["prior_high_20"].to_numpy(dtype=float)
    for index, event in enumerate(events):
        if event:
            last_index = index
            last_level = prior_highs[index]
        if last_index is not None and index - last_index <= lookback:
            levels[index] = last_level
            bars_since[index] = index - last_index
    frame["recent_breakout_level"] = levels
    frame["bars_since_breakout"] = bars_since
    frame["retest_distance_pct"] = (frame["close"] / frame["recent_breakout_level"] - 1) * 100
    frame["retest_touch"] = (
        (frame["low"] <= frame["recent_breakout_level"] * 1.02)
        & (frame["close"] >= frame["recent_breakout_level"] * 0.985)
    ).astype(int)


def compute_indicators(history: pd.DataFrame) -> pd.DataFrame:
    frame = history.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    for column in PRICE_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["date", "open", "high", "low", "close", "volume"])
    frame = frame.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)

    for window in (5, 10, 20, 60, 120):
        frame[f"ma{window}"] = frame["close"].rolling(window, min_periods=window).mean()

    ema_fast = frame["close"].ewm(span=12, adjust=False).mean()
    ema_slow = frame["close"].ewm(span=26, adjust=False).mean()
    frame["dif"] = ema_fast - ema_slow
    frame["dea"] = frame["dif"].ewm(span=9, adjust=False).mean()
    frame["macd_hist"] = (frame["dif"] - frame["dea"]) * 2
    frame["macd_rising"] = (frame["macd_hist"] > frame["macd_hist"].shift(1)).astype(int)
    frame["dif_slope_3_pct"] = (frame["dif"] - frame["dif"].shift(3)) / frame["close"] * 100
    frame["rsi14"] = calc_rsi(frame["close"], 14)

    frame["prior_high_20"] = frame["high"].rolling(20, min_periods=20).max().shift(1)
    frame["prior_high_60"] = frame["high"].rolling(60, min_periods=60).max().shift(1)
    frame["dist_to_prior_high20_pct"] = (frame["close"] / frame["prior_high_20"] - 1) * 100
    frame["dist_to_prior_high60_pct"] = (frame["close"] / frame["prior_high_60"] - 1) * 100
    frame["breakout_20"] = (
        (frame["close"] > frame["prior_high_20"])
        & (frame["close"].shift(1) <= frame["prior_high_20"].shift(1))
    ).astype(int)

    frame["vma20_prev"] = frame["volume"].rolling(20, min_periods=20).mean().shift(1)
    frame["amount_ma20_prev"] = frame["amount"].rolling(20, min_periods=20).mean().shift(1)
    frame["vol_ratio_20"] = frame["volume"] / frame["vma20_prev"]
    recent_volume = frame["volume"].rolling(5, min_periods=5).mean()
    preceding_volume = frame["volume"].shift(5).rolling(20, min_periods=20).mean()
    frame["volume_contraction_5_20"] = recent_volume / preceding_volume

    previous_close = frame["close"].shift(1)
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

    frame["return_5d_pct"] = frame["close"].pct_change(5, fill_method=None) * 100
    frame["return_10d_pct"] = frame["close"].pct_change(10, fill_method=None) * 100
    frame["return_20d_pct"] = frame["close"].pct_change(20, fill_method=None) * 100
    frame["return_60d_pct"] = frame["close"].pct_change(60, fill_method=None) * 100
    frame["extension_ma20_pct"] = (frame["close"] / frame["ma20"] - 1) * 100
    frame["ma20_slope_5_pct"] = frame["ma20"].pct_change(5, fill_method=None) * 100
    frame["ma60_slope_5_pct"] = frame["ma60"].pct_change(5, fill_method=None) * 100

    high10 = frame["high"].rolling(10, min_periods=10).max()
    low10 = frame["low"].rolling(10, min_periods=10).min()
    frame["range_position_10"] = (frame["close"] - low10) / (high10 - low10).replace(0, np.nan)
    frame["one_price_limit"] = (
        np.isclose(frame["high"], frame["low"], rtol=0, atol=0.001)
        & (frame["pct_chg"] >= 9.5)
    ).astype(int)
    _recent_breakout_features(frame)
    return frame


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
