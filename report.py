from __future__ import annotations

import json
import os

import pandas as pd

from config import CONFIG


OUTPUT_FILES = {
    "all_ashare_raw": "all_ashare_raw.csv",
    "stock_universe": "stock_universe.csv",
    "fetch_status": "fetch_status.csv",
    "indicator_table": "indicator_table.csv",
    "feature_table": "feature_table.csv",
    "scored_table": "scored_table.csv",
    "setup_candidates_all": "setup_candidates_all.csv",
    "setup_candidates_top": "setup_candidates_top.csv",
    "breakout_candidates_all": "breakout_candidates_all.csv",
    "breakout_candidates_top": "breakout_candidates_top.csv",
    "retest_candidates_all": "retest_candidates_all.csv",
    "retest_candidates_top": "retest_candidates_top.csv",
    "state_snapshot": "state_snapshot.csv",
    "coverage_report": "coverage_report.json",
}


def save_df(df: pd.DataFrame, key: str):
    if df is None:
        return
    path = os.path.join(CONFIG.output_dir, OUTPUT_FILES[key])
    df.to_csv(path, index=False, encoding="utf-8-sig")


def save_report(report: dict):
    path = os.path.join(CONFIG.output_dir, OUTPUT_FILES["coverage_report"])
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


def save_excel_bundle(bundle: dict[str, pd.DataFrame]):
    if not CONFIG.save_excel:
        return
    excel_path = os.path.join(CONFIG.output_dir, "result.xlsx")
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        for sheet_name, df in bundle.items():
            if df is not None and not df.empty:
                df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
