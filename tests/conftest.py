from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def trending_history() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    count = 240
    dates = pd.bdate_range("2025-01-01", periods=count)
    close = 10 + np.linspace(0, 8, count) + rng.normal(0, 0.08, count)
    open_ = close + rng.normal(0, 0.04, count)
    high = np.maximum(open_, close) + 0.12
    low = np.minimum(open_, close) - 0.12
    volume = 12_000_000 + rng.normal(0, 500_000, count)
    amount = volume * close
    pct_chg = pd.Series(close).pct_change().fillna(0).to_numpy() * 100
    return pd.DataFrame(
        {
            "date": dates,
            "open": open_,
            "close": close,
            "high": high,
            "low": low,
            "volume": volume,
            "amount": amount,
            "amplitude": (high - low) / close * 100,
            "pct_chg": pct_chg,
            "chg": pd.Series(close).diff().fillna(0),
            "turnover": 2.0,
            "code": "000001",
        }
    )

