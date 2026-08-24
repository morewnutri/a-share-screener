from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd


FUND_FLOW_WINDOWS = (3, 5, 10, 20)
FUND_FLOW_FEATURE_COLUMNS = (
    "fund_flow_available",
    "fund_flow_is_current",
    "fund_flow_latest_date",
    "fund_flow_all_windows_positive",
    "fund_flow_positive_window_count",
    "fund_flow_strength_score",
    "fund_flow_rank_reason",
    "participant_structure_available",
    "institutional_dominance_score",
    "retail_pressure_index",
    "participant_structure_label",
    "institutional_favorable_days_20",
    "institutional_favorable_day_ratio_20_pct",
    *(f"main_net_inflow_{window}d_amount" for window in FUND_FLOW_WINDOWS),
    *(f"main_net_inflow_{window}d_yi" for window in FUND_FLOW_WINDOWS),
    *(f"institutional_net_inflow_{window}d_amount" for window in FUND_FLOW_WINDOWS),
    *(f"institutional_net_inflow_{window}d_yi" for window in FUND_FLOW_WINDOWS),
    *(f"retail_net_inflow_{window}d_amount" for window in FUND_FLOW_WINDOWS),
    *(f"retail_net_inflow_{window}d_yi" for window in FUND_FLOW_WINDOWS),
    *(f"institutional_dominance_{window}d_score" for window in FUND_FLOW_WINDOWS),
)

FUND_FLOW_TEXT_COLUMNS = (
    "fund_flow_latest_date",
    "fund_flow_rank_reason",
    "participant_structure_label",
)

PARTICIPANT_AMOUNT_COLUMNS = (
    "small_net_inflow_amount",
    "medium_net_inflow_amount",
    "large_net_inflow_amount",
    "super_large_net_inflow_amount",
)

WINDOW_WEIGHTS = {3: 0.35, 5: 0.30, 10: 0.20, 20: 0.15}


def select_fund_flow_candidates(
    signals: dict[str, pd.DataFrame],
    model_scores: dict[str, str],
    limit: int,
) -> tuple[pd.DataFrame, int]:
    """Select a balanced morphology-first subset for optional fund-flow enrichment."""
    all_rows = [
        frame[["code", "name"]]
        for frame in signals.values()
        if not frame.empty
    ]
    all_candidates = (
        pd.concat(all_rows, ignore_index=True).drop_duplicates("code")
        if all_rows
        else pd.DataFrame(columns=["code", "name"])
    )
    total = len(all_candidates)
    if total == 0:
        return all_candidates, 0

    prepared: list[pd.DataFrame] = []
    for signal, frame in signals.items():
        if frame.empty:
            continue
        ranked = frame[["code", "name"]].copy()
        score_column = model_scores[signal]
        ranked["_score"] = pd.to_numeric(frame[score_column], errors="coerce").fillna(-np.inf)
        ranked["code"] = ranked["code"].astype(str).str.zfill(6)
        prepared.append(
            ranked.sort_values(["_score", "code"], ascending=[False, True])
            .drop_duplicates("code")
            .reset_index(drop=True)
        )

    selected: list[dict[str, object]] = []
    seen: set[str] = set()
    row_index = 0
    target = min(limit, total)
    while len(selected) < target:
        found_row = False
        for frame in prepared:
            if row_index >= len(frame):
                continue
            found_row = True
            row = frame.iloc[row_index]
            code = str(row["code"]).zfill(6)
            if code in seen:
                continue
            seen.add(code)
            selected.append({"code": code, "name": row["name"]})
            if len(selected) == target:
                break
        if not found_row:
            break
        row_index += 1
    return pd.DataFrame(selected, columns=["code", "name"]), total


def summarize_fund_flow(history: pd.DataFrame, expected: date) -> dict:
    frame = history.copy()
    if frame.empty:
        return {}
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["main_net_inflow_amount"] = pd.to_numeric(
        frame["main_net_inflow_amount"], errors="coerce"
    )
    for column in (*PARTICIPANT_AMOUNT_COLUMNS, "main_net_inflow_ratio_pct"):
        if column not in frame:
            frame[column] = np.nan
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["date", "main_net_inflow_amount"])
    frame = frame[frame["date"].dt.date <= expected]
    frame = frame.drop_duplicates("date", keep="last").sort_values("date")
    if frame.empty:
        return {}

    latest = frame["date"].iloc[-1].date()
    result: dict[str, object] = {
        "fund_flow_available": int(len(frame) >= max(FUND_FLOW_WINDOWS)),
        "fund_flow_is_current": int(latest == expected),
        "fund_flow_latest_date": latest.isoformat(),
    }
    positives = 0
    window_main_ratios: dict[int, float] = {}
    for window in FUND_FLOW_WINDOWS:
        amount = (
            float(frame["main_net_inflow_amount"].tail(window).sum())
            if len(frame) >= window
            else np.nan
        )
        result[f"main_net_inflow_{window}d_amount"] = amount
        result[f"main_net_inflow_{window}d_yi"] = amount / 100_000_000 if np.isfinite(amount) else np.nan
        positives += int(np.isfinite(amount) and amount > 0)
        ratio = (
            float(frame["main_net_inflow_ratio_pct"].tail(window).mean())
            if len(frame) >= window
            else np.nan
        )
        window_main_ratios[window] = ratio
    available = bool(result["fund_flow_available"])
    all_positive = available and positives == len(FUND_FLOW_WINDOWS)
    result["fund_flow_all_windows_positive"] = int(all_positive)
    result["fund_flow_positive_window_count"] = positives
    direction_score = positives / len(FUND_FLOW_WINDOWS) * 100
    available_ratios = {
        window: value for window, value in window_main_ratios.items() if np.isfinite(value)
    }
    if available_ratios:
        ratio_weight = sum(WINDOW_WEIGHTS[window] for window in available_ratios)
        weighted_ratio = sum(
            WINDOW_WEIGHTS[window] * value for window, value in available_ratios.items()
        ) / ratio_weight
        ratio_score = float(np.clip(50 + weighted_ratio * 3, 0, 100))
    else:
        ratio_score = 50.0
    result["fund_flow_strength_score"] = round(
        direction_score * 0.70 + ratio_score * 0.30,
        2,
    )

    participant_frame = frame.dropna(subset=list(PARTICIPANT_AMOUNT_COLUMNS)).copy()
    participant_available = len(participant_frame) >= max(FUND_FLOW_WINDOWS)
    result["participant_structure_available"] = int(participant_available)
    dominance_scores: dict[int, float] = {}
    for window in FUND_FLOW_WINDOWS:
        institutional_amount = np.nan
        retail_amount = np.nan
        dominance_score = np.nan
        if len(participant_frame) >= window:
            tail = participant_frame.tail(window)
            institutional_amount = float(
                (tail["large_net_inflow_amount"] + tail["super_large_net_inflow_amount"]).sum()
            )
            retail_amount = float(tail["small_net_inflow_amount"].sum())
            gross_order_imbalance = float(tail[list(PARTICIPANT_AMOUNT_COLUMNS)].abs().sum().sum())
            if gross_order_imbalance > 0:
                contrast = np.clip(
                    (institutional_amount - retail_amount) / gross_order_imbalance,
                    -1,
                    1,
                )
                dominance_score = float(50 + 50 * contrast)
            else:
                dominance_score = 50.0
        result[f"institutional_net_inflow_{window}d_amount"] = institutional_amount
        result[f"institutional_net_inflow_{window}d_yi"] = (
            institutional_amount / 100_000_000
            if np.isfinite(institutional_amount)
            else np.nan
        )
        result[f"retail_net_inflow_{window}d_amount"] = retail_amount
        result[f"retail_net_inflow_{window}d_yi"] = (
            retail_amount / 100_000_000 if np.isfinite(retail_amount) else np.nan
        )
        result[f"institutional_dominance_{window}d_score"] = dominance_score
        dominance_scores[window] = dominance_score

    if participant_available:
        recent = participant_frame.tail(20)
        institutional_daily = (
            recent["large_net_inflow_amount"] + recent["super_large_net_inflow_amount"]
        )
        favorable_days = int(
            ((institutional_daily > 0) & (recent["small_net_inflow_amount"] < 0)).sum()
        )
        adverse_days = int(
            ((institutional_daily < 0) & (recent["small_net_inflow_amount"] > 0)).sum()
        )
        favorable_ratio = favorable_days / len(recent) * 100
        consistency_score = float(
            np.clip(50 + 50 * (favorable_days - adverse_days) / len(recent), 0, 100)
        )
        weighted_dominance = sum(
            WINDOW_WEIGHTS[window] * dominance_scores[window]
            for window in FUND_FLOW_WINDOWS
        )
        institutional_score = float(
            np.clip(weighted_dominance * 0.80 + consistency_score * 0.20, 0, 100)
        )
        retail_index = 100 - institutional_score
        if institutional_score >= 70:
            structure_label = "机构主导明显"
        elif institutional_score >= 58:
            structure_label = "机构偏强"
        elif institutional_score >= 42:
            structure_label = "机构散户均衡"
        elif institutional_score >= 30:
            structure_label = "散户参与偏高"
        else:
            structure_label = "散户主导风险"
        result["institutional_favorable_days_20"] = favorable_days
        result["institutional_favorable_day_ratio_20_pct"] = round(favorable_ratio, 2)
        result["institutional_dominance_score"] = round(institutional_score, 2)
        result["retail_pressure_index"] = round(retail_index, 2)
        result["participant_structure_label"] = structure_label
    else:
        result["institutional_favorable_days_20"] = np.nan
        result["institutional_favorable_day_ratio_20_pct"] = np.nan
        result["institutional_dominance_score"] = np.nan
        result["retail_pressure_index"] = np.nan
        result["participant_structure_label"] = "数据不足"
    if all_positive:
        reason = "3/5/10/20日均净流入"
    elif available:
        reason = f"{positives}/4个周期净流入"
    else:
        reason = "资金历史不足20日"
    if latest != expected:
        reason = f"资金数据滞后至{latest.isoformat()}"
    result["fund_flow_rank_reason"] = reason
    return result


def merge_fund_flow_features(frame: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    existing = [column for column in FUND_FLOW_FEATURE_COLUMNS if column in result]
    if existing:
        result = result.drop(columns=existing)
    if features.empty:
        for column in FUND_FLOW_FEATURE_COLUMNS:
            if column in FUND_FLOW_TEXT_COLUMNS:
                result[column] = ""
            else:
                result[column] = np.nan
        return result
    clean = features.copy()
    clean["code"] = clean["code"].astype(str).str.zfill(6)
    result["code"] = result["code"].astype(str).str.zfill(6)
    return result.merge(clean, on="code", how="left")


def rank_signal_by_fund_flow(
    frame: pd.DataFrame,
    model_score: str,
    morphology_weight: float = 0.50,
    fund_flow_weight: float = 0.35,
    institutional_weight: float = 0.15,
) -> pd.DataFrame:
    result = frame.copy()
    numeric_defaults = {
        "fund_flow_available": 0,
        "fund_flow_is_current": 0,
        "fund_flow_all_windows_positive": 0,
        "fund_flow_positive_window_count": 0,
        "fund_flow_strength_score": 50.0,
        "participant_structure_available": 0,
        "institutional_dominance_score": 50.0,
        **{f"main_net_inflow_{window}d_amount": 0.0 for window in FUND_FLOW_WINDOWS},
    }
    for column, default in numeric_defaults.items():
        if column not in result:
            result[column] = default
        result[column] = pd.to_numeric(result[column], errors="coerce").fillna(default)
    model_values = pd.to_numeric(result[model_score], errors="coerce").fillna(0).clip(0, 100)
    fund_current = result["fund_flow_is_current"] >= 1
    fund_available = result["fund_flow_available"] >= 1
    participant_available = result["participant_structure_available"] >= 1
    fund_score = result["fund_flow_strength_score"].where(
        fund_current & fund_available,
        50.0,
    )
    institutional_score = result["institutional_dominance_score"].where(
        fund_current & participant_available,
        50.0,
    )
    result["final_selection_score"] = (
        model_values * morphology_weight
        + fund_score * fund_flow_weight
        + institutional_score * institutional_weight
    ).round(2)
    result["selection_evidence_coverage_pct"] = (
        morphology_weight * 100
        + (fund_current & fund_available).astype(int) * fund_flow_weight * 100
        + (fund_current & participant_available).astype(int) * institutional_weight * 100
    ).round(2)
    result["selection_weight_policy"] = (
        f"形态{morphology_weight:.0%}/主力资金{fund_flow_weight:.0%}/机构主导{institutional_weight:.0%}"
    )
    sort_columns = [
        "final_selection_score",
        "selection_evidence_coverage_pct",
        model_score,
        "fund_flow_strength_score",
        "institutional_dominance_score",
    ]
    result = result.sort_values(
        sort_columns,
        ascending=[False] * len(sort_columns),
    ).reset_index(drop=True)
    result["rank"] = np.arange(1, len(result) + 1)
    return result
