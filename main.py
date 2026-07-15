# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import pandas as pd

from cache_manager import (
    build_hist_meta,
    ensure_dir,
    load_csv_if_exists,
    need_refresh_hist,
    read_hist_cache,
    save_hist_cache,
)
from calendar_utils import latest_complete_trade_date
from config import CONFIG
from data_sources import eastmoney_get_hist, eastmoney_get_index_hist, probe_eastmoney_hist, probe_sina_hist
from features import add_relative_strength, finalize_cross_section, summarize_last_row
from filters import apply_base_filters
from indicators import compute_indicators
from report import save_df, save_excel_bundle, save_report
from state_tracker import update_state
from strategies import apply_strategies
from universe import build_universe, get_stock_list_with_cache


def fetch_hist_with_policy(code: str, beg: str, end: str, source_flags: dict):
    if source_flags.get("eastmoney_hist_ok", False):
        df = eastmoney_get_hist(code, beg, end, fqt=CONFIG.fqt, lmt=1500)
        if df is not None and not df.empty:
            return df, "eastmoney"
    return pd.DataFrame(), "none"


def process_one_stock(code: str, name: str, source_flags: dict, expected_trade_date):
    rec = {
        "code": code,
        "name": name,
        "status": "",
        "source": "",
        "bars": 0,
        "last_date": "",
        "error": "",
        "from_cache": 0,
    }

    try:
        refresh_needed = need_refresh_hist(
            CONFIG.hist_cache_dir,
            code,
            force_refresh=CONFIG.force_refresh_hist,
            expire_hours=CONFIG.cache_expire_hours,
            expected_trade_date=expected_trade_date,
            fqt=CONFIG.fqt,
            allowed_source="eastmoney",
        )

        if not refresh_needed:
            hist = read_hist_cache(CONFIG.hist_cache_dir, code)
            if hist is not None and not hist.empty:
                rec["from_cache"] = 1
                rec["source"] = "cache"
            else:
                hist = pd.DataFrame()
        else:
            hist = pd.DataFrame()

        if hist.empty:
            hist, source = fetch_hist_with_policy(code, CONFIG.start_date, CONFIG.end_date, source_flags)
            rec["source"] = source
            if hist is not None and not hist.empty:
                last_date = pd.to_datetime(hist["date"]).max().date()
                if last_date > expected_trade_date:
                    hist = hist[pd.to_datetime(hist["date"]).dt.date <= expected_trade_date].copy().reset_index(drop=True)
                meta = build_hist_meta(code, source, CONFIG.fqt, hist)
                save_hist_cache(CONFIG.hist_cache_dir, code, hist, meta)

        if hist.empty:
            rec["status"] = "empty_hist"
            return rec, None

        rec["bars"] = len(hist)
        rec["last_date"] = pd.to_datetime(hist["date"]).max().strftime("%Y-%m-%d")

        if pd.to_datetime(hist["date"]).max().date() < expected_trade_date:
            rec["status"] = "stale_hist"
            return rec, None

        if len(hist) < CONFIG.min_history_bars:
            rec["status"] = "insufficient_bars"
            return rec, None

        ind_df = compute_indicators(hist)
        rec["status"] = "ok"

        if CONFIG.sleep_seconds > 0:
            time.sleep(CONFIG.sleep_seconds)

        return rec, ind_df
    except Exception as e:
        rec["status"] = "failed"
        rec["error"] = f"{type(e).__name__}: {str(e)}"
        return rec, None


def main():
    ensure_dir(CONFIG.output_dir)
    ensure_dir(CONFIG.cache_dir)
    ensure_dir(CONFIG.list_cache_dir)
    ensure_dir(CONFIG.hist_cache_dir)
    ensure_dir(CONFIG.state_dir)

    now_dt = datetime.now()
    expected_trade_date = latest_complete_trade_date(now_dt, CONFIG.market_close_time)

    print("=" * 80)
    print("A股主板日线筛选系统开始运行")
    print(f"开始时间: {now_dt.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"区间: {CONFIG.start_date} ~ {CONFIG.end_date}")
    print(f"最新完整交易日: {expected_trade_date}")
    print("=" * 80)

    print("\n[0/6] 数据源健康检查...")
    east_ok, east_host = probe_eastmoney_hist()
    sina_ok, _ = probe_sina_hist()
    source_flags = {
        "eastmoney_hist_ok": east_ok,
        "eastmoney_hist_host": east_host,
        "sina_hist_ok": sina_ok,
    }
    print(f"东方财富历史接口可用: {east_ok}, host={east_host}")
    print(f"新浪历史接口可用: {sina_ok}")

    print("\n[1/6] 获取A股列表并过滤主板...")
    raw_list, list_source = get_stock_list_with_cache()
    raw_all, universe = build_universe(raw_list)
    save_df(raw_all, "all_ashare_raw")
    save_df(universe, "stock_universe")
    print(f"股票列表来源: {list_source}")
    print(f"原始A股列表数量: {len(raw_all)}")
    print(f"主板股票数量: {len(universe)}")

    print("\n[2/6] 并发抓取历史数据并计算指标...")
    fetch_status_records = []
    indicator_frames = []

    to_process = [(row["code"], row["name"]) for _, row in universe.iterrows()]
    total = len(to_process)
    processed_count = 0

    with ThreadPoolExecutor(max_workers=CONFIG.max_workers) as executor:
        future_map = {
            executor.submit(process_one_stock, code, name, source_flags, expected_trade_date): (code, name)
            for code, name in to_process
        }

        for future in as_completed(future_map):
            code, name = future_map[future]
            processed_count += 1
            try:
                rec, ind_df = future.result()
            except Exception as e:
                rec = {
                    "code": code,
                    "name": name,
                    "status": "failed",
                    "source": "",
                    "bars": 0,
                    "last_date": "",
                    "error": f"{type(e).__name__}: {str(e)}",
                    "from_cache": 0,
                }
                ind_df = None

            fetch_status_records.append(rec)
            if ind_df is not None and not ind_df.empty:
                last_ind = ind_df.copy()
                last_ind["code"] = code
                last_ind["name"] = name
                indicator_frames.append(last_ind)

            if processed_count % 20 == 0 or processed_count == total:
                ok_count = sum(1 for x in fetch_status_records if x["status"] == "ok")
                print(f"进度: {processed_count}/{total} | ok={ok_count} | 当前完成: {code} {name} [{rec['status']}, source={rec['source']}, cache={rec.get('from_cache', 0)}]")

    fetch_status_df = pd.DataFrame(fetch_status_records).sort_values(["code"]).reset_index(drop=True)
    save_df(fetch_status_df, "fetch_status")

    print("\n[3/6] 构建指标与特征表...")
    benchmark_df = eastmoney_get_index_hist(CONFIG.benchmark_index_code, CONFIG.start_date, CONFIG.end_date, lmt=1500)
    feature_rows = []
    indicator_last_rows = []

    for ind_df in indicator_frames:
        code = str(ind_df["code"].iloc[-1]).zfill(6)
        name = str(ind_df["name"].iloc[-1])
        source = "eastmoney"
        with_rs = add_relative_strength(ind_df, benchmark_df)
        indicator_last_rows.append(with_rs.iloc[-1:].copy())
        feature_rows.append(summarize_last_row(with_rs, code, name, source))

    indicator_table = pd.concat(indicator_last_rows, ignore_index=True) if indicator_last_rows else pd.DataFrame()
    feature_table = pd.DataFrame(feature_rows)
    feature_table = finalize_cross_section(feature_table)
    filtered_table = apply_base_filters(feature_table, expected_trade_date)

    if not indicator_table.empty:
        save_df(indicator_table, "indicator_table")
    if not filtered_table.empty:
        save_df(filtered_table, "feature_table")

    print("\n[4/6] 应用筛选策略...")
    if filtered_table.empty:
        setup_all = pd.DataFrame()
        breakout_all = pd.DataFrame()
        retest_all = pd.DataFrame()
        scored_table = pd.DataFrame()
        setup_top = pd.DataFrame()
        breakout_top = pd.DataFrame()
        retest_top = pd.DataFrame()
    else:
        setup_all, breakout_all, retest_all, scored_table, setup_top, breakout_top, retest_top = apply_strategies(filtered_table)

    save_df(scored_table, "scored_table")
    save_df(setup_all, "setup_candidates_all")
    save_df(setup_top, "setup_candidates_top")
    save_df(breakout_all, "breakout_candidates_all")
    save_df(breakout_top, "breakout_candidates_top")
    save_df(retest_all, "retest_candidates_all")
    save_df(retest_top, "retest_candidates_top")

    print(f"setup 候选数量: {len(setup_all)}")
    print(f"breakout 候选数量: {len(breakout_all)}")
    print(f"retest 候选数量: {len(retest_all)}")

    print("\n[5/6] 更新状态与输出报告...")
    state_snapshot = update_state(setup_all, breakout_all, retest_all, expected_trade_date)
    save_df(state_snapshot, "state_snapshot")

    excluded_df = raw_all[raw_all["exclude_reason"] != ""].copy()
    report = {
        "run_time": now_dt.strftime("%Y-%m-%d %H:%M:%S"),
        "expected_trade_date": str(expected_trade_date),
        "config": {
            "start_date": CONFIG.start_date,
            "end_date": CONFIG.end_date,
            "fqt": CONFIG.fqt,
            "min_history_bars": CONFIG.min_history_bars,
            "exclude_st": CONFIG.exclude_st,
            "mainboard_prefixes": list(CONFIG.mainboard_prefixes),
            "max_workers": CONFIG.max_workers,
            "cache_expire_hours": CONFIG.cache_expire_hours,
            "list_cache_expire_hours": CONFIG.list_cache_expire_hours,
        },
        "source_health": source_flags,
        "source_policy": {
            "stock_list_source": list_source,
            "hist_primary": "eastmoney_forward_adjusted_only",
            "hist_fallback": "disabled_for_scoring",
        },
        "universe_summary": {
            "raw_ashare_count": int(len(raw_all)),
            "mainboard_count_after_filter": int(len(universe)),
            "excluded_count": int(len(excluded_df)),
            "excluded_not_mainboard_count": int((raw_all["exclude_reason"] == "not_mainboard").sum()),
            "excluded_st_count": int((raw_all["exclude_reason"] == "ST").sum()),
        },
        "fetch_summary": {
            "ok_count": int((fetch_status_df["status"] == "ok").sum()) if not fetch_status_df.empty else 0,
            "failed_count": int((fetch_status_df["status"] == "failed").sum()) if not fetch_status_df.empty else 0,
            "empty_hist_count": int((fetch_status_df["status"] == "empty_hist").sum()) if not fetch_status_df.empty else 0,
            "insufficient_bars_count": int((fetch_status_df["status"] == "insufficient_bars").sum()) if not fetch_status_df.empty else 0,
            "stale_hist_count": int((fetch_status_df["status"] == "stale_hist").sum()) if not fetch_status_df.empty else 0,
            "cache_ok_count": int(((fetch_status_df["status"] == "ok") & (fetch_status_df["source"] == "cache")).sum()) if not fetch_status_df.empty else 0,
            "eastmoney_ok_count": int(((fetch_status_df["status"] == "ok") & (fetch_status_df["source"] == "eastmoney")).sum()) if not fetch_status_df.empty else 0,
        },
        "calculation_summary": {
            "indicator_rows_count": int(len(indicator_table)),
            "feature_rows_count": int(len(filtered_table)),
            "coverage_vs_mainboard_pct": round(len(filtered_table) / len(universe) * 100, 2) if len(universe) > 0 else 0.0,
        },
        "result_summary": {
            "setup_all_count": int(len(setup_all)),
            "setup_top_count": int(len(setup_top)),
            "breakout_all_count": int(len(breakout_all)),
            "breakout_top_count": int(len(breakout_top)),
            "retest_all_count": int(len(retest_all)),
            "retest_top_count": int(len(retest_top)),
        },
        "failed_codes": fetch_status_df.loc[fetch_status_df["status"] == "failed", ["code", "name", "source", "error"]].to_dict(orient="records") if not fetch_status_df.empty else [],
        "stale_codes": fetch_status_df.loc[fetch_status_df["status"] == "stale_hist", ["code", "name", "source", "last_date"]].to_dict(orient="records") if not fetch_status_df.empty else [],
    }
    save_report(report)

    save_excel_bundle({
        "all_ashare_raw": raw_all,
        "stock_universe": universe,
        "fetch_status": fetch_status_df,
        "indicator_table": indicator_table,
        "feature_table": filtered_table,
        "scored_table": scored_table,
        "setup_all": setup_all,
        "breakout_all": breakout_all,
        "retest_all": retest_all,
        "state_snapshot": state_snapshot,
    })

    print("\n[6/6] 输出完成")
    print("输出目录：", CONFIG.output_dir)

    if not setup_top.empty:
        print("\n[SETUP TOP10]")
        print(setup_top[["code", "name", "close", "rsi14", "vol_ratio_20", "score_setup"]].head(10).to_string(index=False))
    if not breakout_top.empty:
        print("\n[BREAKOUT TOP10]")
        print(breakout_top[["code", "name", "close", "rsi14", "vol_ratio_20", "score_breakout"]].head(10).to_string(index=False))
    if not retest_top.empty:
        print("\n[RETEST TOP10]")
        print(retest_top[["code", "name", "close", "rsi14", "vol_ratio_20", "score_retest"]].head(10).to_string(index=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("程序运行失败：", e)
        print(traceback.format_exc())
