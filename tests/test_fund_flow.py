from datetime import date

import pandas as pd
import pytest

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
    assert result["participant_structure_available"] == 0
    assert result["participant_structure_label"] == "数据不足"


def _participant_history(expected: date, institutional_positive: bool) -> pd.DataFrame:
    direction = 1 if institutional_positive else -1
    return pd.DataFrame(
        {
            "date": pd.bdate_range(end=expected.isoformat(), periods=20),
            "main_net_inflow_amount": [direction * 20_000_000.0] * 20,
            "main_net_inflow_ratio_pct": [direction * 5.0] * 20,
            "small_net_inflow_amount": [-direction * 10_000_000.0] * 20,
            "medium_net_inflow_amount": [-direction * 5_000_000.0] * 20,
            "large_net_inflow_amount": [direction * 8_000_000.0] * 20,
            "super_large_net_inflow_amount": [direction * 7_000_000.0] * 20,
        }
    )


def test_participant_structure_detects_institutional_dominance():
    expected = date(2026, 8, 14)
    result = summarize_fund_flow(_participant_history(expected, True), expected)

    assert result["participant_structure_available"] == 1
    assert result["institutional_favorable_days_20"] == 20
    assert result["institutional_favorable_day_ratio_20_pct"] == 100
    assert result["institutional_dominance_score"] == pytest.approx(93.33, abs=0.01)
    assert result["retail_pressure_index"] == pytest.approx(6.67, abs=0.01)
    assert result["participant_structure_label"] == "机构主导明显"
    assert result["fund_flow_strength_score"] == pytest.approx(89.5)


def test_participant_structure_detects_retail_dominance_risk():
    expected = date(2026, 8, 14)
    result = summarize_fund_flow(_participant_history(expected, False), expected)

    assert result["participant_structure_available"] == 1
    assert result["institutional_favorable_days_20"] == 0
    assert result["institutional_dominance_score"] == pytest.approx(6.67, abs=0.01)
    assert result["retail_pressure_index"] == pytest.approx(93.33, abs=0.01)
    assert result["participant_structure_label"] == "散户主导风险"


def test_zero_order_imbalance_is_neutral_participant_structure():
    expected = date(2026, 8, 14)
    frame = pd.DataFrame(
        {
            "date": pd.bdate_range(end=expected.isoformat(), periods=20),
            "main_net_inflow_amount": [0.0] * 20,
            "small_net_inflow_amount": [0.0] * 20,
            "medium_net_inflow_amount": [0.0] * 20,
            "large_net_inflow_amount": [0.0] * 20,
            "super_large_net_inflow_amount": [0.0] * 20,
        }
    )
    result = summarize_fund_flow(frame, expected)

    assert result["institutional_dominance_score"] == 50
    assert result["retail_pressure_index"] == 50
    assert result["participant_structure_label"] == "机构散户均衡"


def test_weighted_ranking_keeps_morphology_and_fund_flow_dominant():
    candidates = pd.DataFrame(
        [
            {"code": "000001", "chip_base_rebound_score": 90},
            {"code": "000002", "chip_base_rebound_score": 80},
            {"code": "000003", "chip_base_rebound_score": 85},
        ]
    )
    features = pd.DataFrame(
        [
            {
                "code": "000001",
                "fund_flow_available": 1,
                "fund_flow_is_current": 1,
                "fund_flow_strength_score": 20,
                "participant_structure_available": 1,
                "institutional_dominance_score": 20,
            },
            {
                "code": "000002",
                "fund_flow_available": 1,
                "fund_flow_is_current": 1,
                "fund_flow_strength_score": 90,
                "participant_structure_available": 1,
                "institutional_dominance_score": 90,
            },
        ]
    )
    enriched = merge_fund_flow_features(candidates, features)
    ranked = rank_signal_by_fund_flow(enriched, "chip_base_rebound_score")

    assert ranked["code"].tolist() == ["000002", "000003", "000001"]
    assert ranked["rank"].tolist() == [1, 2, 3]
    scores = ranked.set_index("code")["final_selection_score"]
    assert scores["000002"] == 85
    assert scores["000003"] == 67.5
    assert scores["000001"] == 55
    coverage = ranked.set_index("code")["selection_evidence_coverage_pct"]
    assert coverage["000002"] == 100
    assert coverage["000003"] == 50


def test_missing_participant_structure_uses_neutral_score_without_hard_filter():
    candidates = pd.DataFrame(
        [{"code": "000001", "chip_base_rebound_score": 80}]
    )
    features = pd.DataFrame(
        [
            {
                "code": "000001",
                "fund_flow_available": 1,
                "fund_flow_is_current": 1,
                "fund_flow_strength_score": 80,
                "participant_structure_available": 0,
            }
        ]
    )
    ranked = rank_signal_by_fund_flow(
        merge_fund_flow_features(candidates, features),
        "chip_base_rebound_score",
    )

    assert ranked.loc[0, "final_selection_score"] == 75.5
    assert ranked.loc[0, "selection_evidence_coverage_pct"] == 85


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
