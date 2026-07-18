from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd


OPTIONAL_FIELDS = (
    "main_net_inflow_ratio_pct",
    "main_net_inflow_amount",
    "main_net_inflow_positive_days_5",
    "institution_net_buy_amount",
    "margin_balance_growth_3d_pct",
    "shareholder_count_change_pct",
    "sector_profit_growth_median_pct",
    "has_reduction_plan",
    "industry_risk_ok",
)


def _empty_metadata(path: Path) -> dict:
    return {
        "path": str(path),
        "loaded": False,
        "matched_codes": 0,
        "effective_date": None,
        "note": "Optional evidence absent; no stock is rejected because of it.",
    }


def merge_optional_evidence(
    frame: pd.DataFrame,
    data_dir: str | Path,
    expected: date,
) -> tuple[pd.DataFrame, dict]:
    result = frame.copy()
    path = Path(data_dir) / "external" / "funding_signals.csv"
    metadata = _empty_metadata(path)

    for field in OPTIONAL_FIELDS:
        if field not in result:
            result[field] = np.nan

    if not path.exists():
        return result, metadata

    try:
        external = pd.read_csv(path, dtype={"code": str})
    except (pd.errors.EmptyDataError, UnicodeDecodeError, OSError) as exc:
        metadata["note"] = f"Optional evidence ignored: {type(exc).__name__}: {exc}"
        return result, metadata

    if external.empty or "code" not in external or "date" not in external:
        metadata["note"] = "Optional evidence ignored: CSV needs code and date columns."
        return result, metadata

    external = external.copy()
    external["code"] = external["code"].astype(str).str.zfill(6)
    external["date"] = pd.to_datetime(external["date"], errors="coerce")
    external = external[external["date"].dt.date <= expected]
    if external.empty:
        metadata["note"] = "Optional evidence has no rows on or before the scan date."
        return result, metadata

    available = [field for field in OPTIONAL_FIELDS if field in external]
    for field in available:
        external[field] = pd.to_numeric(external[field], errors="coerce")
    latest = (
        external.sort_values(["code", "date"])
        .drop_duplicates("code", keep="last")[["code", "date", *available]]
    )
    merged = result.merge(latest, on="code", how="left", suffixes=("", "_external"))
    for field in OPTIONAL_FIELDS:
        external_field = f"{field}_external"
        if external_field in merged:
            merged[field] = merged[external_field].where(merged[external_field].notna(), merged[field])
            merged = merged.drop(columns=external_field)

    matched = merged["date_external"].notna() if "date_external" in merged else pd.Series(False, index=merged.index)
    effective_dates = merged.loc[matched, "date_external"] if "date_external" in merged else pd.Series(dtype="datetime64[ns]")
    if "date_external" in merged:
        merged = merged.drop(columns="date_external")
    metadata.update(
        {
            "loaded": True,
            "matched_codes": int(matched.sum()),
            "effective_date": (
                pd.Timestamp(effective_dates.max()).strftime("%Y-%m-%d")
                if not effective_dates.empty
                else None
            ),
            "note": "Optional evidence affects scores only; missing values are neutral.",
        }
    )
    return merged, metadata