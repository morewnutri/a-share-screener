from __future__ import annotations

import numpy as np
import pandas as pd


def calc_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - 100 / (1 + rs)
    rsi = rsi.mask((avg_loss == 0) & (avg_gain > 0), 100.0)
    rsi = rsi.mask((avg_loss == 0) & (avg_gain == 0), 50.0)
    return rsi


def calc_macd(close: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=signal, adjust=False).mean()
    macd = (dif - dea) * 2
    return dif, dea, macd


def slope_n(series: pd.Series, n: int = 5) -> float:
    s = series.dropna()
    if len(s) < n:
        return np.nan
    y = s.iloc[-n:].values.astype(float)
    x = np.arange(len(y))
    try:
        return float(np.polyfit(x, y, 1)[0])
    except Exception:
        return np.nan


def compute_atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def max_drawdown_from_high(close: pd.Series, lookback: int = 20) -> float:
    s = close.dropna()
    if len(s) < lookback:
        return np.nan
    recent = s.iloc[-lookback:]
    high = recent.max()
    last = recent.iloc[-1]
    if high == 0:
        return np.nan
    return (last / high - 1.0) * 100


def compute_indicators(hist: pd.DataFrame) -> pd.DataFrame:
    df = hist.copy().sort_values("date").reset_index(drop=True)

    for n in [5, 10, 20, 60, 120]:
        df[f"ma{n}"] = df["close"].rolling(n).mean()

    for n in [5, 10, 20]:
        df[f"vma{n}"] = df["volume"].rolling(n).mean()

    df["vma20_prev"] = df["volume"].rolling(20, min_periods=20).mean().shift(1)

    df["dif"], df["dea"], df["macd"] = calc_macd(df["close"])
    df["rsi14"] = calc_rsi(df["close"], 14)

    df["prior_high_20"] = df["high"].rolling(20, min_periods=20).max().shift(1)
    df["prior_high_60"] = df["high"].rolling(60, min_periods=60).max().shift(1)
    df["prior_low_20"] = df["low"].rolling(20, min_periods=20).min().shift(1)

    df["is_breakout_20"] = (df["close"] > df["prior_high_20"]).astype(int)
    df["is_breakout_60"] = (df["close"] > df["prior_high_60"]).astype(int)

    df["atr14"] = compute_atr(df, 14)
    df["atr_pct"] = np.where(df["close"] > 0, df["atr14"] / df["close"] * 100, np.nan)
    df["atr_pct_mean_20"] = df["atr_pct"].rolling(20).mean()

    df["amount_ma20"] = df["amount"].rolling(20).mean()
    df["return_5d_pct"] = df["close"].pct_change(5) * 100
    df["return_10d_pct"] = df["close"].pct_change(10) * 100
    df["return_20d_pct"] = df["close"].pct_change(20) * 100
    df["extension_ma20_pct"] = np.where(df["ma20"] > 0, (df["close"] / df["ma20"] - 1) * 100, np.nan)
    df["macd_rising"] = (df["macd"] > df["macd"].shift(1)).astype(int)
    df["dif_slope_3"] = df["dif"].diff(3)

    return df
