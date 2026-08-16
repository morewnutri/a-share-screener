from datetime import date

import pandas as pd

from ashare_scanner.fund_flow import (
    merge_fund_flow_features,
    rank_signal_by_fund_flow,
    select_fund_flow_candidates,
    summarize_fund_flow,
)


def test_summarize_fund_flow_calculates_nested_positive_windows():
    expected = date(2026, 8, 14)
    frame = pd.DataFrame(
        {
            "date": pd.bdate_range(end=expected.isoformat(), periods=20),
            "main_net_inflow_amount": [10_000_000.0] * 20,
        }
    )
    result = summarize_fund_flow(frame, expected)
    assert result["fund_flow_is_current"] == 1
    assert result["fund_flow_all_windows_positive"] == 1
    assert result["fund_flow_positive_window_count"] == 4
    assert result["main_net_inflow_3d_amount"] == 30_000_000
    assert result["main_net_inflow_20d_yi"] == 2.0


def test_fund_flow_ranking_puts_all_positive_windows_first():
    candidates = pd.DataFrame(
        [
            {"code": "000001", "chip_base_rebound_score": 65},
            {"code": "000002", "chip_base_rebound_score": 95},
            {"code": "000003", "chip_base_rebound_score": 99},
        ]
    )
    features = pd.DataFrame(
        [
            {
                "code": "000001",
                "fund_flow_is_current": 1,
                "fund_flow_all_windows_positive": 1,
                "fund_flow_positive_window_count": 4,
                "main_net_inflow_3d_amount": 1,
                "main_net_inflow_5d_amount": 1,
                "main_net_inflow_10d_amount": 1,
                "main_net_inflow_20d_amount": 1,
            },
            {
                "code": "000002",
                "fund_flow_is_current": 1,
                "fund_flow_all_windows_positive": 0,
                "fund_flow_positive_window_count": 3,
                "main_net_inflow_3d_amount": 100,
                "main_net_inflow_5d_amount": 100,
                "main_net_inflow_10d_amount": 100,
                "main_net_inflow_20d_amount": -1,
            },
        ]
    )
    enriched = merge_fund_flow_features(candidates, features)
    ranked = rank_signal_by_fund_flow(enriched, "chip_base_rebound_score")
    assert ranked["code"].tolist() == ["000001", "000002", "000003"]
    assert ranked["rank"].tolist() == [1, 2, 3]


def test_stale_fund_flow_is_marked_not_current():
    expected = date(2026, 8, 14)
    frame = pd.DataFrame(
        {
            "date": pd.bdate_range(end="2026-08-13", periods=20),
            "main_net_inflow_amount": [1_000_000.0] * 20,
        }
    )
    result = summarize_fund_flow(frame, expected)
    assert result["fund_flow_is_current"] == 0
    assert "滞后" in result["fund_flow_rank_reason"]


def test_fund_flow_candidate_cap_balances_signal_lists():
    signals = {
        "ready": pd.DataFrame(
            [
                {"code": "000001", "name": "a", "ready_score": 99},
                {"code": "000002", "name": "b", "ready_score": 98},
                {"code": "000003", "name": "c", "ready_score": 97},
            ]
        ),
        "launch": pd.DataFrame(
            [
                {"code": "600001", "name": "d", "launch_score": 96},
                {"code": "600002", "name": "e", "launch_score": 95},
                {"code": "600003", "name": "f", "launch_score": 94},
            ]
        ),
    }
    selected, total = select_fund_flow_candidates(
        signals,
        {"ready": "ready_score", "launch": "launch_score"},
        limit=4,
    )
    assert total == 6
    assert selected["code"].tolist() == ["000001", "600001", "000002", "600002"]
