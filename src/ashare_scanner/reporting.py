from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


SIGNAL_LABELS = {
    "setup_contraction": "突破前收缩",
    "setup_accumulation": "量价积累",
    "breakout_today": "当日突破",
    "retest_after_breakout": "突破后回踩",
}

STEP_LABELS = {
    "history_bars": "历史长度",
    "liquidity": "流动性",
    "tradable_features": "数据/可交易",
    "trend_structure": "趋势结构",
    "not_extended": "未过度上涨",
    "near_prior_high": "接近前高",
    "relative_strength": "相对强度",
    "contraction": "波动/量能收缩",
    "up_down_volume": "上涨量占优",
    "obv_improving": "OBV改善",
    "range_position": "区间位置",
    "above_prior_high": "突破前高",
    "breakout_distance": "突破幅度",
    "breakout_volume": "突破放量",
    "not_overextended": "未过度延伸",
    "buyable_close": "收盘可交易",
    "recent_breakout": "近期有突破",
    "near_breakout_level": "回踩突破位",
    "quiet_retest": "缩量回踩",
    "close_recovered": "收盘转强",
}

RESULT_COLUMNS = [
    "rank",
    "code",
    "name",
    "close",
    "pct_chg",
    "score_total",
    "rs20_percentile",
    "dist_to_prior_high20_pct",
    "vol_ratio_20",
    "extension_ma20_pct",
]

RESULT_RENAMES = {
    "rank": "排名",
    "code": "代码",
    "name": "名称",
    "close": "收盘",
    "pct_chg": "涨跌%",
    "score_total": "评分",
    "rs20_percentile": "相对强度",
    "dist_to_prior_high20_pct": "距前高%",
    "vol_ratio_20": "量比20",
    "extension_ma20_pct": "距MA20%",
}


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, dtype={"code": str})
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def _print_table(frame: pd.DataFrame, columns: list[str], renames: dict[str, str]) -> None:
    available = [column for column in columns if column in frame.columns]
    view = frame[available].rename(columns=renames).copy()
    if "代码" in view:
        view["代码"] = view["代码"].astype(str).str.zfill(6)
    if "相对强度" in view:
        view["相对强度"] = (view["相对强度"] * 100).round(1)
    numeric_columns = view.select_dtypes(include="number").columns
    view[numeric_columns] = view[numeric_columns].round(2)
    with pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        180,
        "display.max_colwidth",
        20,
    ):
        print(view.to_string(index=False))


def print_run_summary(run_dir: str | Path, top_n: int = 20) -> None:
    run_dir = Path(run_dir)
    report = json.loads((run_dir / "coverage_report.json").read_text(encoding="utf-8"))
    screening = report.get("screening", {})
    fetch = report.get("fetch", {})

    print("\n" + "=" * 96)
    print(f"筛选日期: {report.get('expected_complete_session', '')}")
    print(
        f"股票池: {report.get('universe', {}).get('count', 0)} | "
        f"有效指标: {fetch.get('status_counts', {}).get('ok', 0)} | "
        f"覆盖率: {fetch.get('coverage_pct', 0)}%"
    )
    print(f"抓取状态: {fetch.get('status_counts', {})}")
    print(
        f"唯一候选: {screening.get('unique_candidate_count', 0)} | "
        f"候选率: {screening.get('candidate_rate_pct', 0)}%"
    )
    print(f"诊断: {screening.get('assessment', '')}")
    print("=" * 96)

    counts = report.get("signals", {})
    for signal, label in SIGNAL_LABELS.items():
        frame = _read_csv(run_dir / f"{signal}_all.csv")
        print(f"\n[{label}] 共 {counts.get(signal, len(frame))} 只")
        if frame.empty:
            print("无符合条件股票。")
        else:
            _print_table(frame.head(top_n), RESULT_COLUMNS, RESULT_RENAMES)

    funnel = _read_csv(run_dir / "screening_funnel.csv")
    if not funnel.empty:
        print("\n[筛选漏斗]")
        for signal, label in SIGNAL_LABELS.items():
            rows = funnel[funnel["signal"] == signal]
            stages = " -> ".join(
                f"{STEP_LABELS.get(row.step, row.step)}:{int(row.remaining_count)}"
                for row in rows.itertuples(index=False)
            )
            print(f"{label}: {stages}")

    if screening.get("unique_candidate_count", 0) <= 2:
        near = _read_csv(run_dir / "near_miss_top100.csv")
        if not near.empty:
            print(f"\n[最接近入选的前 {min(top_n, len(near))} 只]")
            near_columns = [
                "near_miss_rank",
                "code",
                "name",
                "closest_signal",
                "failed_at",
                "score_total",
                "close",
                "rs20_percentile",
                "dist_to_prior_high20_pct",
                "vol_ratio_20",
            ]
            near_renames = {
                "near_miss_rank": "排名",
                "code": "代码",
                "name": "名称",
                "closest_signal": "最接近策略",
                "failed_at": "未通过步骤",
                "score_total": "评分",
                "close": "收盘",
                "rs20_percentile": "相对强度",
                "dist_to_prior_high20_pct": "距前高%",
                "vol_ratio_20": "量比20",
            }
            view = near.head(top_n).copy()
            view["closest_signal"] = view["closest_signal"].map(SIGNAL_LABELS).fillna(
                view["closest_signal"]
            )
            view["failed_at"] = view["failed_at"].map(STEP_LABELS).fillna(view["failed_at"])
            _print_table(view, near_columns, near_renames)

    print(f"\n结果目录: {run_dir}")
