from datetime import date

import pandas as pd

from ashare_scanner.enrichment import merge_optional_evidence


def test_external_evidence_uses_latest_non_future_row(tmp_path):
    external = tmp_path / "external"
    external.mkdir()
    pd.DataFrame(
        [
            {"code": "000001", "date": "2026-07-14", "main_net_inflow_ratio_pct": 21},
            {"code": "000001", "date": "2026-07-16", "main_net_inflow_ratio_pct": 99},
        ]
    ).to_csv(external / "funding_signals.csv", index=False)
    frame = pd.DataFrame(
        [{"code": "000001", "date": pd.Timestamp("2026-07-15")}]
    )
    merged, metadata = merge_optional_evidence(frame, tmp_path, date(2026, 7, 15))
    assert merged.loc[0, "main_net_inflow_ratio_pct"] == 21
    assert metadata["loaded"] is True
    assert metadata["matched_codes"] == 1


def test_missing_external_evidence_is_neutral(tmp_path):
    frame = pd.DataFrame([{"code": "000001"}])
    merged, metadata = merge_optional_evidence(frame, tmp_path, date(2026, 7, 15))
    assert metadata["loaded"] is False
    assert pd.isna(merged.loc[0, "institution_net_buy_amount"])