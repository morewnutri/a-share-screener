import json
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from ashare_scanner.cache import HistoryCache, UniverseCache
from ashare_scanner.config import AppConfig, DataConfig, StrategyConfig
from ashare_scanner.scanner import DailyScanner


def test_daily_scanner_runs_end_to_end_from_fresh_cache(tmp_path, trending_history):
    expected = pd.Timestamp("2026-07-15").date()
    history = trending_history.copy()
    history["date"] = pd.bdate_range(end=expected.isoformat(), periods=len(history))
    history2 = history.copy()
    history2["code"] = "000002"
    history2["close"] *= 1.03
    history2["open"] *= 1.03
    history2["high"] *= 1.03
    history2["low"] *= 1.03
    history2["amount"] = history2["volume"] * history2["close"]

    data_config = DataConfig(
        start_date="2025-01-01",
        min_history_bars=150,
        max_workers=2,
        min_universe_size=2,
    )
    config = AppConfig(
        data_dir=str(tmp_path),
        data=data_config,
        strategy=StrategyConfig(min_amount_ma20=1),
    )
    history_cache = HistoryCache(tmp_path / "cache", data_config.start_date, data_config.fqt)
    history_cache.write("000001", history, "eastmoney:test", "stock", expected)
    history_cache.write("000002", history2, "eastmoney:test", "stock", expected)
    benchmark = history.copy()
    benchmark["code"] = "1_000300"
    history_cache.write(
        "benchmark_1.000300",
        benchmark,
        "eastmoney:test",
        "benchmark",
        expected,
    )
    universe = pd.DataFrame(
        [{"code": "000001", "name": "sample-a"}, {"code": "000002", "name": "sample-b"}]
    )
    UniverseCache(tmp_path / "cache").write(universe, "test", expected)

    run_dir = DailyScanner(config).run(
        datetime(2026, 7, 15, 15, 20, tzinfo=ZoneInfo("Asia/Shanghai"))
    )
    report = json.loads((run_dir / "coverage_report.json").read_text(encoding="utf-8"))
    assert report["fetch"]["status_counts"] == {"ok": 2}
    assert report["fetch"]["coverage_pct"] == 100.0
    assert (run_dir / "indicators_scored.csv").exists()
    assert (run_dir / "setup_contraction_all.csv").exists()

