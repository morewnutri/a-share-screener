import pandas as pd

from ashare_scanner.config import DataConfig, StrategyConfig
from ashare_scanner.strategies import apply_strategies


def _row(code: str, close: float, prior_high: float, vol_ratio: float) -> dict:
    return {
        "code": code,
        "name": code,
        "date": pd.Timestamp("2026-07-15"),
        "bars": 200,
        "open": close - 0.2,
        "close": close,
        "ma20": 95.0,
        "ma60": 90.0,
        "ma20_slope_5_pct": 1.0,
        "ma60_slope_5_pct": 0.5,
        "dif": 0.5,
        "dea": 0.4,
        "macd_rising": 1,
        "prior_high_20": prior_high,
        "dist_to_prior_high20_pct": (close / prior_high - 1) * 100,
        "extension_ma20_pct": (close / 95.0 - 1) * 100,
        "return_5d_pct": 5.0,
        "return_10d_pct": 8.0,
        "vol_ratio_20": vol_ratio,
        "up_down_volume_ratio_10": 1.5,
        "obv_slope_10_pct": 2.0,
        "range_position_10": 0.8,
        "atr_contraction_ratio": 0.8,
        "volume_contraction_5_20": 0.8,
        "bb_contraction_ratio": 0.8,
        "rs20_percentile": 0.9,
        "rs60_percentile": 0.8,
        "amount_ma20_prev": 120_000_000.0,
        "one_price_limit": 0,
        "pct_chg": 3.0,
        "rsi14": 65.0,
        "bars_since_breakout": 0.0,
        "retest_distance_pct": 0.0,
        "retest_touch": 0,
    }


def test_setup_does_not_include_already_broken_out_stock():
    frame = pd.DataFrame(
        [
            _row("000001", close=99.0, prior_high=100.0, vol_ratio=0.8),
            _row("000002", close=101.0, prior_high=100.0, vol_ratio=1.5),
        ]
    )
    _, signals = apply_strategies(frame, DataConfig(), StrategyConfig())
    assert signals["setup_contraction"]["code"].tolist() == ["000001"]
    assert signals["breakout_today"]["code"].tolist() == ["000002"]


def test_scores_are_capped_at_100():
    frame = pd.DataFrame([_row("000001", close=99.0, prior_high=100.0, vol_ratio=0.8)])
    scored, _ = apply_strategies(frame, DataConfig(), StrategyConfig())
    assert 0 <= scored.loc[0, "score_total"] <= 100

