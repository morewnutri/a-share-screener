from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from . import __version__

SIGNAL_LABELS = {
    "chip_base_ready": "低位横盘+筹码峰（待启动）",
    "chip_base_launch": "低位横盘+筹码峰（刚启动）",
    "chip_base_rebound": "横盘后反弹（启动确认）",
}

STEP_LABELS = {
    "history_bars": "历史长度",
    "liquidity": "流动性",
    "chip_features_ready": "筹码模型覆盖",
    "bottom_location": "下跌后低位",
    "sideways_platform": "启动前横盘",
    "concentrated_low_chip_peak": "低位集中筹码峰",
    "risk_control": "失败/派发风险",
    "waiting_near_peak": "等待启动位置",
    "early_launch": "刚启动价格行为",
    "chip_base_ready_score": "待启动评分",
    "chip_base_launch_score": "刚启动评分",
    "rebound_location": "反弹平台位置",
    "recent_historical_platform": "近期历史平台",
    "rebound_chip_peak": "反弹型筹码峰",
    "rebound_risk_control": "反弹风险控制",
    "post_platform_rebound": "平台后反弹",
    "chip_base_rebound_score": "反弹确认评分",
}

RESULT_COLUMNS = [
    "rank",
    "code",
    "name",
    "close",
    "pct_chg",
    "final_selection_score",
    "chip_base_ready_score",
    "chip_base_launch_score",
    "chip_base_rebound_score",
    "fund_flow_strength_score",
    "institutional_dominance_score",
    "retail_pressure_index",
    "participant_structure_label",
    "institutional_favorable_day_ratio_20_pct",
    "selection_evidence_coverage_pct",
    "fund_flow_rank_reason",
    "main_net_inflow_3d_yi",
    "main_net_inflow_5d_yi",
    "main_net_inflow_10d_yi",
    "main_net_inflow_20d_yi",
    "adaptive_base_window",
    "adaptive_base_offset",
    "adaptive_base_drawdown_120_pct",
    "adaptive_base_width_pct",
    "adaptive_base_return_pct",
    "chip_peak_price",
    "chip_peak_distance_pct",
    "chip_peak_band_share_pct",
    "chip_70_width_pct",
    "chip_low_zone_share_pct",
    "chip_overhead_ratio_pct",
    "chip_significant_peak_count",
    "return_5d_pct",
]

RESULT_RENAMES = {
    "rank": "排名",
    "code": "代码",
    "name": "名称",
    "close": "收盘",
    "pct_chg": "涨跌%",
    "final_selection_score": "综合分",
    "chip_base_ready_score": "待启动分",
    "chip_base_launch_score": "刚启动分",
    "chip_base_rebound_score": "反弹确认分",
    "fund_flow_strength_score": "主力资金分",
    "institutional_dominance_score": "机构主导指数",
    "retail_pressure_index": "散户压力指数",
    "participant_structure_label": "参与者结构",
    "institutional_favorable_day_ratio_20_pct": "20日机构占优天数%",
    "selection_evidence_coverage_pct": "排序证据覆盖%",
    "fund_flow_rank_reason": "资金排序依据",
    "main_net_inflow_3d_yi": "3日主力净流入(亿)",
    "main_net_inflow_5d_yi": "5日主力净流入(亿)",
    "main_net_inflow_10d_yi": "10日主力净流入(亿)",
    "main_net_inflow_20d_yi": "20日主力净流入(亿)",
    "adaptive_base_window": "横盘周期",
    "adaptive_base_offset": "距平台结束日",
    "adaptive_base_drawdown_120_pct": "平台前高回撤%",
    "adaptive_base_width_pct": "横盘宽度%",
    "adaptive_base_return_pct": "横盘涨跌%",
    "chip_peak_price": "主筹码峰价",
    "chip_peak_distance_pct": "距主峰%",
    "chip_peak_band_share_pct": "峰带筹码%",
    "chip_70_width_pct": "70%成本宽度",
    "chip_low_zone_share_pct": "低位筹码%",
    "chip_overhead_ratio_pct": "上方套牢盘%",
    "chip_significant_peak_count": "显著峰数",
    "return_5d_pct": "5日涨幅%",
}

REFERENCE_COLUMNS = [
    "code",
    "name",
    "acceptance_example",
    "selected_any",
    "fetch_status",
    "last_date",
    "chip_base_ready",
    "chip_base_launch",
    "chip_base_rebound",
    "closest_signal",
    "failed_at",
    "final_selection_score",
    "chip_base_ready_score",
    "chip_base_launch_score",
    "chip_base_rebound_score",
    "fund_flow_strength_score",
    "institutional_dominance_score",
    "retail_pressure_index",
    "participant_structure_label",
    "adaptive_base_window",
    "adaptive_base_offset",
    "adaptive_base_width_pct",
    "adaptive_base_return_pct",
    "adaptive_base_drawdown_120_pct",
    "chip_peak_price",
    "chip_peak_band_share_pct",
    "chip_70_width_pct",
    "chip_low_zone_share_pct",
    "return_5d_pct",
]

REFERENCE_RENAMES = {
    "code": "代码",
    "name": "名称",
    "acceptance_example": "标准答案",
    "selected_any": "是否命中",
    "fetch_status": "日线状态",
    "last_date": "最新日线",
    "chip_base_ready": "待启动",
    "chip_base_launch": "刚启动",
    "chip_base_rebound": "反弹",
    "closest_signal": "最接近策略",
    "failed_at": "首个未通过",
    **RESULT_RENAMES,
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
    numeric_columns = view.select_dtypes(include="number").columns
    view[numeric_columns] = view[numeric_columns].round(2)
    with pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        260,
        "display.max_colwidth",
        24,
    ):
        print(view.to_string(index=False))


def _print_reference_audit(run_dir: Path) -> None:
    audit = _read_csv(run_dir / "reference_examples_audit.csv")
    if audit.empty:
        return
    indicator_status = audit.get("indicators_ready", pd.Series(0, index=audit.index))
    ready_count = int(pd.to_numeric(indicator_status, errors="coerce").fillna(0).sum())
    print(f"\n[参考样本逐股审计] 指标可用 {ready_count}/{len(audit)} 只")
    if "acceptance_example" in audit and "selected_any" in audit:
        acceptance = audit[
            pd.to_numeric(audit["acceptance_example"], errors="coerce").fillna(0) == 1
        ]
        hit_count = int(
            pd.to_numeric(acceptance["selected_any"], errors="coerce").fillna(0).sum()
        )
        print(
            f"[标准答案回归检查] 命中 {hit_count}/{len(acceptance)} 只；"
            "名单只用于审计，不会绕过策略条件。"
        )
    view = audit.copy()
    if "closest_signal" in view:
        view["closest_signal"] = view["closest_signal"].map(SIGNAL_LABELS).fillna(
            view["closest_signal"]
        )
    if "failed_at" in view:
        view["failed_at"] = view["failed_at"].map(STEP_LABELS).fillna(view["failed_at"])
    _print_table(view, REFERENCE_COLUMNS, REFERENCE_RENAMES)


def _print_failure_diagnostics(run_dir: Path, limit: int = 8) -> None:
    fetch = _read_csv(run_dir / "fetch_status.csv")
    if fetch.empty or "status" not in fetch:
        return
    problems = fetch[fetch["status"] != "ok"].copy()
    if problems.empty:
        return
    print(f"\n[抓取失败样例] 共 {len(problems)} 只，显示前 {min(limit, len(problems))} 只")
    columns = [
        column
        for column in ("code", "name", "status", "source", "error")
        if column in problems
    ]
    view = problems[columns].head(limit).rename(
        columns={
            "code": "代码",
            "name": "名称",
            "status": "状态",
            "source": "数据源",
            "error": "失败原因",
        }
    )
    if "失败原因" in view:
        view["失败原因"] = view["失败原因"].astype(str).str.slice(0, 180)
    with pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        260,
        "display.max_colwidth",
        180,
    ):
        print(view.to_string(index=False))


def print_run_summary(run_dir: str | Path, top_n: int = 20) -> None:
    run_dir = Path(run_dir)
    report = json.loads((run_dir / "coverage_report.json").read_text(encoding="utf-8"))
    screening = report.get("screening", {})
    fetch = report.get("fetch", {})

    print("\n" + "=" * 112)
    print(f"程序版本: {__version__}")
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
    print("筹码口径: modeled_cyq（换手衰减+日内三角分布估算，不是真实账户持仓）")
    print("散户口径: 超大/大/中/小单净额构造的交易行为代理（不是账户或股东身份）")
    fund_flow = report.get("fund_flow", {})
    weights = fund_flow.get("selection_weights", {})
    print(
        f"资金排序: 请求{fund_flow.get('requested_candidate_count', 0)}只 | "
        f"当日有效{fund_flow.get('current_count', 0)}只 | "
        f"3/5/10/20日均净流入{fund_flow.get('all_windows_positive_count', 0)}只 | "
        f"机构结构有效{fund_flow.get('participant_structure_count', 0)}只 | "
        f"机构偏强{fund_flow.get('institutional_preferred_count', 0)}只"
    )
    if weights:
        print(
            "综合排序权重: "
            f"形态{float(weights.get('morphology', 0)):.0%} | "
            f"主力资金{float(weights.get('fund_flow', 0)):.0%} | "
            f"机构主导{float(weights.get('institutional_dominance', 0)):.0%}"
        )
    print("=" * 112)

    if not screening.get("valid", True):
        provisional = screening.get("provisional_unique_candidate_count", 0)
        minimum = screening.get("minimum_coverage_pct", 90)
        print(
            f"\n[本次扫描无效] 覆盖率未达到 {minimum}%。"
            f"{provisional} 只临时候选不作为全市场筛选结果，不打印、不更新观察池。"
        )
        print("请先解决日线抓取；临时诊断保存在 provisional_*.csv。")
        _print_failure_diagnostics(run_dir)
        _print_reference_audit(run_dir)
        print(f"\n结果目录: {run_dir}")
        return

    counts = report.get("signals", {})
    for signal, label in SIGNAL_LABELS.items():
        frame = _read_csv(run_dir / f"{signal}_all.csv")
        print(f"\n[{label}] 共 {counts.get(signal, len(frame))} 只")
        if frame.empty:
            print("无符合条件股票。CSV 只有表头属于正常结果，不代表保存失败。")
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

    if screening.get("unique_candidate_count", 0) <= 10:
        near = _read_csv(run_dir / "near_miss_top100.csv")
        if not near.empty:
            print(f"\n[最接近入选的前 {min(top_n, len(near))} 只]")
            near_columns = [
                "near_miss_rank",
                "code",
                "name",
                "closest_signal",
                "failed_at",
                *RESULT_COLUMNS[5:],
            ]
            near_renames = {
                "near_miss_rank": "排名",
                "code": "代码",
                "name": "名称",
                "closest_signal": "最接近策略",
                "failed_at": "首个未通过",
                **RESULT_RENAMES,
            }
            view = near.head(top_n).copy()
            view["closest_signal"] = view["closest_signal"].map(SIGNAL_LABELS).fillna(
                view["closest_signal"]
            )
            view["failed_at"] = view["failed_at"].map(STEP_LABELS).fillna(
                view["failed_at"]
            )
            _print_table(view, near_columns, near_renames)

    _print_reference_audit(run_dir)

    print(f"\n结果目录: {run_dir}")
