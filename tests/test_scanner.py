import json
from datetime import date, datetime
from zoneinfo import ZoneInfo

import pandas as pd

from ashare_scanner.cache import HistoryCache, UniverseCache
from ashare_scanner.config import AppConfig, DataConfig, StrategyConfig
from ashare_scanner.reporting import print_run_summary
from ashare_scanner.scanner import DailyScanner


def test_daily_scanner_runs_end_to_end_from_fresh_cache(tmp_path, trending_history, capsys):
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
        [
            {"code": "000001", "name": "sample-a", "float_market_cap": 10_000_000_000},
            {"code": "000002", "name": "sample-b", "float_market_cap": 12_000_000_000},
        ]
    )
    UniverseCache(tmp_path / "cache").write(universe, "test", expected)

    run_dir = DailyScanner(config).run(
        datetime(2026, 7, 15, 15, 20, tzinfo=ZoneInfo("Asia/Shanghai"))
    )
    report = json.loads((run_dir / "coverage_report.json").read_text(encoding="utf-8"))
    assert report["fetch"]["status_counts"] == {"ok": 2}
    assert report["fetch"]["coverage_pct"] == 100.0
    assert report["external_evidence"]["loaded"] is False
    assert (run_dir / "indicators_scored.csv").exists()
    assert (run_dir / "chip_base_ready_all.csv").exists()
    assert (run_dir / "chip_base_launch_all.csv").exists()
    assert (run_dir / "chip_base_rebound_all.csv").exists()
    assert (run_dir / "shape_only_chip_base_ready_all.csv").exists()
    assert (run_dir / "shape_only_chip_base_launch_all.csv").exists()
    assert (run_dir / "shape_only_chip_base_rebound_all.csv").exists()
    assert (run_dir / "fund_flow_status.csv").exists()
    assert (run_dir / "screening_funnel.csv").exists()
    assert (run_dir / "near_miss_top100.csv").exists()
    print_run_summary(run_dir, top_n=5)
    output = capsys.readouterr().out
    assert "[低位横盘+筹码峰（待启动）]" in output
    assert "[筛选漏斗]" in output
    assert "筹码口径: modeled_cyq" in output


def test_low_coverage_invalidates_published_candidates_and_preserves_state(
    tmp_path,
    trending_history,
    capsys,
):
    expected = pd.Timestamp("2026-07-15").date()
    history = trending_history.copy()
    history["date"] = pd.bdate_range(end=expected.isoformat(), periods=len(history))
    data_config = DataConfig(
        start_date="2025-01-01",
        min_history_bars=150,
        max_workers=2,
        min_universe_size=3,
        min_coverage_pct=90,
    )
    config = AppConfig(
        data_dir=str(tmp_path),
        data=data_config,
        strategy=StrategyConfig(min_amount_ma20=1),
    )
    history_cache = HistoryCache(tmp_path / "cache", data_config.start_date, data_config.fqt)
    history_cache.write("000001", history, "eastmoney:test", "stock", expected)
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
        [
            {"code": "000001", "name": "sample-a"},
            {"code": "000002", "name": "sample-b"},
            {"code": "000003", "name": "sample-c"},
        ]
    )
    UniverseCache(tmp_path / "cache").write(universe, "test", expected)

    class FailedHistorySource:
        def fetch_stock_history(self, *args, **kwargs):
            raise RuntimeError("provider unavailable")

    scanner = DailyScanner(config)
    scanner.source = FailedHistorySource()
    run_dir = scanner.run(
        datetime(2026, 7, 15, 15, 20, tzinfo=ZoneInfo("Asia/Shanghai"))
    )

    report = json.loads((run_dir / "coverage_report.json").read_text(encoding="utf-8"))
    assert report["screening"]["valid"] is False
    assert report["fetch"]["coverage_pct"] < 90
    assert (run_dir / "provisional_chip_base_ready.csv").exists()
    assert pd.read_csv(run_dir / "chip_base_ready_all.csv").empty
    assert len(pd.read_csv(run_dir / "reference_examples_audit.csv")) > 10
    assert not (tmp_path / "state" / "watchlist.csv").exists()

    print_run_summary(run_dir, top_n=5)
    output = capsys.readouterr().out
    assert "[本次扫描无效]" in output


def test_fund_flow_failures_stop_early_without_dropping_candidates(tmp_path):
    data_config = DataConfig(
        fund_flow_max_workers=2,
        fund_flow_max_candidates=10,
        fund_flow_request_pause_seconds=0,
        fund_flow_progress_every=2,
        fund_flow_failure_streak_limit=2,
        fund_flow_stage_timeout_minutes=1,
    )
    scanner = DailyScanner(AppConfig(data_dir=str(tmp_path), data=data_config))

    class FailedFundFlowSource:
        def fetch_stock_fund_flow(self, *args, **kwargs):
            raise TimeoutError("simulated provider timeout")

    scanner.fund_flow_source = FailedFundFlowSource()
    signal_specs = {
        "chip_base_ready": ("000001", "chip_base_ready_score"),
        "chip_base_launch": ("000002", "chip_base_launch_score"),
        "chip_base_rebound": ("000003", "chip_base_rebound_score"),
    }
    signals = {
        signal: pd.DataFrame([{"code": code, "name": signal, score: 80.0}])
        for signal, (code, score) in signal_specs.items()
    }
    scored = pd.DataFrame([{"code": code} for code, _ in signal_specs.values()])

    _, ranked, status, metadata = scanner._enrich_fund_flow(
        scored,
        signals,
        date(2026, 8, 14),
    )

    assert metadata["stopped_reason"] == "consecutive_failures"
    assert metadata["attempted_candidate_count"] == 2
    assert len(status) == 2
    assert all(len(ranked[signal]) == 1 for signal in signals)
