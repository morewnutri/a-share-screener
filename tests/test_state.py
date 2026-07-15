from datetime import date

import pandas as pd

from ashare_scanner.state import update_watchlist


def test_watchlist_update_is_idempotent_for_same_session(tmp_path):
    row = {
        "code": "000001",
        "name": "sample",
        "score_total": 80,
        "close": 10.0,
        "prior_high_20": 10.2,
        "recent_breakout_level": float("nan"),
        "ma20": 9.5,
    }
    signals = {
        "setup_contraction": pd.DataFrame([row]),
        "setup_accumulation": pd.DataFrame(),
        "breakout_today": pd.DataFrame(),
        "retest_after_breakout": pd.DataFrame(),
    }
    indicators = pd.DataFrame([row])
    session = date(2026, 7, 15)
    first, active, transitions = update_watchlist(tmp_path, signals, indicators, session, 10)
    second, _, second_transitions = update_watchlist(tmp_path, signals, indicators, session, 10)
    assert first.loc[0, "state"] == "SETUP"
    assert len(active) == 1
    assert len(transitions) == 1
    assert second.loc[0, "age_sessions"] == 0
    assert second_transitions.empty

