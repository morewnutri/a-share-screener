from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


SIGNAL_LABELS = {
    "accumulation_late": "强资金运作型（埋伏吸筹末期）",
    "main_wave": "强资金运作型（主升浪阶段）",
}

STEP_LABELS = {
    "history_bars": "历史长度",
    "liquidity": "流动性",
    "tradable_features": "数据/可交易",
    "accumulation_position": "低位与未过度上涨",
    "accumulation_risk": "突破失败/派发风险",
    "evidence_groups": "证据组数量",
    "accumulation_score": "埋伏型评分",
    "trend_resonance": "20/30/60日趋势共振",
    "ma5_funding_trigger": "MA5资金触发",
    "market_risk": "大盘系统性风险",
    "wave_risk": "追高/派发风险",
    "main_wave_score": "主升浪评分",
}

RESULT_COLUMNS = [
    "rank",
    "code",
    "name",
    "close",
    "pct_chg",
    "accumulation_score",
    "main_wave_score",
    "accumulation_evidence_groups",
    "rs20_percentile",
    "vol_ratio_20",
    "turnover_ratio_20",
    "cost_concentration_60_pct",
    "main_net_inflow_ratio_pct",
    "distribution_day_count_5",
]

RESULT_RENAMES = {
    "rank": "排名",
    "code": "代码",
    "name": "名称",
    "close": "收盘",
    "pct_chg": "涨跌%",
    "accumulation_score": "埋伏分",
    "main_wave_score": "主升分",
    "accumulation_evidence_groups": "证据组",
    "rs20_percentile": "相对强度",
    "vol_ratio_20": "量比20",
    "turnover_ratio_20": "换手比20",
    "cost_concentration_60_pct": "成本集中代理%",
    "main_net_inflow_ratio_pct": "主力净流入占比%",
    "distribution_day_count_5": "近5日派发风险",
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
        220,
        "display.max_colwidth",
        24,
    ):
        print(view.to_string(index=False))


def print_run_summary(run_dir: str | Path, top_n: int = 20) -> None:
    run_dir = Path(run_dir)
    report = json.loads((run_dir / "coverage_report.json").read_text(encoding="utf-8"))
    screening = report.get("screening", {})
    fetch = report.get("fetch", {})

    print("\n" + "=" * 104)
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
    external = report.get("external_evidence", {})
    print(
        f"外部资金证据: {'已加载' if external.get('loaded') else '未加载'} | "
        f"匹配股票: {external.get('matched_codes', 0)} | "
        f"说明: {external.get('note', '')}"
    )
    print("=" * 104)

    counts = report.get("signals", {})
    for signal, label in SIGNAL_LABELS.items():
        frame = _read_csv(run_dir / f"{signal}_all.csv")
        print(f"\n[{label}] 共 {counts.get(signal, len(frame))} 只")
        if frame.empty:
            print("无符合条件股票。该文件只有表头属于正常结果，不代表保存失败。")
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

    if screening.get("unique_candidate_count", 0) <= 5:
        near = _read_csv(run_dir / "near_miss_top100.csv")
        if not near.empty:
            print(f"\n[最接近入选的前 {min(top_n, len(near))} 只]")
            near_columns = [
                "near_miss_rank",
                "code",
                "name",
                "closest_signal",
                "failed_at",
                "accumulation_score",
                "main_wave_score",
                "accumulation_evidence_groups",
                "close",
                "rs20_percentile",
                "vol_ratio_20",
                "distribution_day_count_5",
            ]
            near_renames = {
                "near_miss_rank": "排名",
                "code": "代码",
                "name": "名称",
                "closest_signal": "最接近策略",
                "failed_at": "首个未通过",
                "accumulation_score": "埋伏分",
                "main_wave_score": "主升分",
                "accumulation_evidence_groups": "证据组",
                "close": "收盘",
                "rs20_percentile": "相对强度",
                "vol_ratio_20": "量比20",
                "distribution_day_count_5": "近5日派发风险",
            }
            view = near.head(top_n).copy()
            view["closest_signal"] = view["closest_signal"].map(SIGNAL_LABELS).fillna(
                view["closest_signal"]
            )
            view["failed_at"] = view["failed_at"].map(STEP_LABELS).fillna(view["failed_at"])
            _print_table(view, near_columns, near_renames)

    print(f"\n结果目录: {run_dir}")