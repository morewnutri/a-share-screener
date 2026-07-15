import numpy as np
import pandas as pd

from ashare_scanner.backtest import _outcomes


def test_outcome_enters_at_next_open_and_respects_drawdown():
    history = pd.DataFrame(
        {
            "open": [10.0, 10.0, 10.5, 11.0, 11.0],
            "high": [10.0, 10.5, 11.3, 11.0, 11.0],
            "low": [10.0, 9.7, 10.0, 10.5, 10.5],
        }
    )
    result = _outcomes(history, horizon=2, target_pct=12, max_drawdown_pct=6)
    assert result.loc[0, "outcome_success"] == 1
    assert np.isclose(result.loc[0, "forward_max_return_pct"], 13.0)

