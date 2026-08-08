from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


SIGNAL_LABELS = {
    "chip_base_ready": "低位横盘+筹码峰（待启动）",
    "chip_base_launch": "低位横盘+筹码峰（刚启动）",
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
}

RESULT_COLUMNS = [
    "rank",
    "code",
    "name",
    "close",
    "pct_chg",
    "chip_base_ready_score",
    "chip_base_launch_score",
    "base_drawdown_from_120_high_pct",
    "base_width_20_pre3_pct",
    "base_return_20_pre3_pct",
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
    "chip_base_ready_score": "待启动分",
    "chip_base_launch_score": "刚启动分",
    "base_drawdown_from_120_high_pct": "平台前高回撤%",
    "base_width_20_pre3_pct": "平台宽度%",
    "base_return_20_pre3_pct": "平台涨跌%",
    "chip_peak_price": "主筹码峰价",
    "chip_peak_distance_pct": "距主峰%",
    "chip_peak_band_share_pct": "峰带筹码%",
    "chip_70_width_pct": "70%成本宽度",
    "chip_low_zone_share_pct": "低位筹码%",
    "chip_overhead_ratio_pct": "上方套牢盘%",
    "chip_significant_peak_count": "显著峰数",
    "return_5d_pct": "5日涨幅%",
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


def print_run_summary(run_dir: str | Path, top_n: int = 20) -> None:
    run_dir = Path(run_dir)
    report = json.loads((run_dir / "coverage_report.json").read_text(encoding="utf-8"))
    screening = report.get("screening", {})
    fetch = report.get("fetch", {})

    print("\n" + "=" * 112)
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
    print("=" * 112)

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

    print(f"\n结果目录: {run_dir}")
