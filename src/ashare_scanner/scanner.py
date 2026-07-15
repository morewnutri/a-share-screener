from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .cache import HistoryCache, UniverseCache, atomic_write_csv, atomic_write_json
from .calendar import expected_complete_session
from .config import AppConfig
from .datasource import EastmoneyDataSource
from .http import HttpClient
from .indicators import compute_indicators, latest_snapshot
from .state import update_watchlist
from .strategies import SIGNAL_ORDER, add_relative_strength, apply_strategies


LOGGER = logging.getLogger(__name__)


class DailyScanner:
    def __init__(self, config: AppConfig, data_dir: str | Path | None = None) -> None:
        self.config = config
        self.data_dir = Path(data_dir or config.data_dir).expanduser().resolve()
        self.cache_dir = self.data_dir / "cache"
        self.http = HttpClient(config.data.request_timeout, config.data.max_retries)
        self.source = EastmoneyDataSource(self.http, config.data.fqt)
        self.history_cache = HistoryCache(
            self.cache_dir,
            config.data.start_date,
            config.data.fqt,
        )
        self.universe_cache = UniverseCache(self.cache_dir)

    def run(self, as_of: datetime | None = None) -> Path:
        started = time.monotonic()
        expected = expected_complete_session(
            as_of,
            self.config.data.close_buffer_minutes,
        )
        run_dir = self.data_dir / "runs" / expected.isoformat()
        run_dir.mkdir(parents=True, exist_ok=True)
        LOGGER.info("Expected latest complete session: %s", expected)

        universe, universe_source, universe_warning = self._get_universe(expected)
        atomic_write_csv(universe, run_dir / "universe.csv")
        LOGGER.info("Universe: %d stocks (%s)", len(universe), universe_source)

        benchmark = self._get_benchmark(expected)
        benchmark_indicators = compute_indicators(benchmark)
        benchmark_map20 = benchmark_indicators.set_index("date")["return_20d_pct"]
        benchmark_map60 = benchmark_indicators.set_index("date")["return_60d_pct"]

        statuses: list[dict[str, Any]] = []
        snapshots: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=self.config.data.max_workers) as executor:
            futures = {
                executor.submit(self._process_stock, row.code, row.name, expected): (row.code, row.name)
                for row in universe[["code", "name"]].itertuples(index=False)
            }
            for completed, future in enumerate(as_completed(futures), start=1):
                code, name = futures[future]
                try:
                    status, snapshot = future.result()
                except Exception as exc:
                    status = {
                        "code": code,
                        "name": name,
                        "status": "failed",
                        "source": "",
                        "from_cache": 0,
                        "bars": 0,
                        "last_date": "",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    snapshot = None
                statuses.append(status)
                if snapshot is not None:
                    snapshots.append(snapshot)
                if completed % 50 == 0 or completed == len(futures):
                    ok_count = sum(row["status"] == "ok" for row in statuses)
                    LOGGER.info("Progress %d/%d, ok=%d", completed, len(futures), ok_count)

        status_frame = pd.DataFrame(statuses).sort_values("code").reset_index(drop=True)
        atomic_write_csv(status_frame, run_dir / "fetch_status.csv")
        if not snapshots:
            raise RuntimeError("No stock produced a valid latest-session indicator snapshot.")

        indicators = pd.DataFrame(snapshots)
        indicators["date"] = pd.to_datetime(indicators["date"])
        indicators["benchmark_return_20d_pct"] = indicators["date"].map(benchmark_map20)
        indicators["benchmark_return_60d_pct"] = indicators["date"].map(benchmark_map60)
        indicators = add_relative_strength(indicators)
        scored, signals = apply_strategies(
            indicators,
            self.config.data,
            self.config.strategy,
        )
        scored["date"] = scored["date"].dt.strftime("%Y-%m-%d")
        atomic_write_csv(scored.sort_values("code"), run_dir / "indicators_scored.csv")

        for signal_name in SIGNAL_ORDER:
            full = signals[signal_name].copy()
            full["date"] = pd.to_datetime(full["date"]).dt.strftime("%Y-%m-%d")
            atomic_write_csv(full, run_dir / f"{signal_name}_all.csv")
            atomic_write_csv(
                full.head(self.config.strategy.top_n),
                run_dir / f"{signal_name}_top{self.config.strategy.top_n}.csv",
            )

        _, active_watchlist, new_transitions = update_watchlist(
            self.data_dir,
            signals,
            scored,
            expected,
            self.config.strategy.watchlist_ttl_sessions,
        )
        atomic_write_csv(active_watchlist, run_dir / "watchlist_active.csv")
        atomic_write_csv(new_transitions, run_dir / "state_transitions.csv")

        report = self._build_report(
            expected,
            universe,
            universe_source,
            universe_warning,
            status_frame,
            scored,
            signals,
            active_watchlist,
            time.monotonic() - started,
        )
        atomic_write_json(report, run_dir / "coverage_report.json")
        atomic_write_json(
            {
                "latest_complete_session": expected.isoformat(),
                "run_dir": str(run_dir),
                "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            },
            self.data_dir / "latest_run.json",
        )
        LOGGER.info("Run complete: %s", run_dir)
        return run_dir

    def _get_universe(self, expected: date) -> tuple[pd.DataFrame, str, str]:
        cached = self.universe_cache.read(expected, self.config.data.universe_cache_hours)
        if cached.fresh and not self.config.data.force_refresh:
            return cached.frame, str(cached.metadata.get("source", "cache")), ""
        try:
            frame, source = self.source.fetch_universe(self.config.data.min_universe_size)
            self.universe_cache.write(frame, source, expected)
            return frame, source, ""
        except Exception as exc:
            if not cached.frame.empty:
                warning = f"Universe refresh failed; using stale cache: {type(exc).__name__}: {exc}"
                LOGGER.warning(warning)
                return cached.frame, str(cached.metadata.get("source", "stale_cache")), warning
            raise

    def _get_benchmark(self, expected: date) -> pd.DataFrame:
        key = f"benchmark_{self.config.data.benchmark_secid}"
        cached = self.history_cache.read(key, expected, source_kind="benchmark")
        if cached.fresh and not self.config.data.force_refresh:
            return cached.frame
        frame, source = self.source.fetch_benchmark_history(
            self.config.data.benchmark_secid,
            self.config.data.start_date,
            expected,
        )
        if frame.empty or frame["date"].max().date() != expected:
            last = "empty" if frame.empty else frame["date"].max().date().isoformat()
            raise RuntimeError(f"Benchmark is stale: expected={expected}, actual={last}")
        self.history_cache.write(key, frame, source, "benchmark", expected)
        return frame

    def _process_stock(self, code: str, name: str, expected: date) -> tuple[dict, dict | None]:
        status: dict[str, Any] = {
            "code": str(code).zfill(6),
            "name": name,
            "status": "",
            "source": "",
            "from_cache": 0,
            "bars": 0,
            "last_date": "",
            "error": "",
        }
        cached = self.history_cache.read(code, expected, source_kind="stock")
        if cached.fresh and not self.config.data.force_refresh:
            history = cached.frame
            source = str(cached.metadata.get("source", "eastmoney:cache"))
            status["from_cache"] = 1
        else:
            try:
                history, source = self.source.fetch_stock_history(
                    code,
                    self.config.data.start_date,
                    expected,
                )
                if not history.empty:
                    self.history_cache.write(code, history, source, "stock", expected)
            except Exception as exc:
                status["status"] = "failed"
                status["error"] = f"{type(exc).__name__}: {exc}"
                return status, None

        status["source"] = source
        if history.empty:
            status["status"] = "empty_hist"
            return status, None
        last_date = history["date"].max().date()
        status["bars"] = int(len(history))
        status["last_date"] = last_date.isoformat()
        if last_date != expected:
            status["status"] = "stale_hist"
            return status, None
        if len(history) < self.config.data.min_history_bars:
            status["status"] = "insufficient_bars"
            return status, None

        indicator_frame = compute_indicators(history)
        snapshot = latest_snapshot(indicator_frame, code, name, source)
        status["status"] = "ok"
        return status, snapshot

    def _build_report(
        self,
        expected: date,
        universe: pd.DataFrame,
        universe_source: str,
        universe_warning: str,
        statuses: pd.DataFrame,
        indicators: pd.DataFrame,
        signals: dict[str, pd.DataFrame],
        active_watchlist: pd.DataFrame,
        elapsed_seconds: float,
    ) -> dict[str, Any]:
        status_counts = statuses["status"].value_counts().to_dict()
        failures = statuses[statuses["status"].isin(["failed", "stale_hist", "empty_hist"])]
        return {
            "run_time": datetime.now().astimezone().isoformat(timespec="seconds"),
            "expected_complete_session": expected.isoformat(),
            "elapsed_seconds": round(elapsed_seconds, 2),
            "config": {
                "data": asdict(self.config.data),
                "strategy": asdict(self.config.strategy),
            },
            "data_policy": {
                "stock_adjustment": "forward_adjusted_fqt_1",
                "history_sources": "eastmoney_multi_host_only",
                "incomplete_daily_bars": "discarded",
                "stale_stock_history": "excluded_from_signals",
            },
            "universe": {
                "source": universe_source,
                "warning": universe_warning,
                "count": int(len(universe)),
                "prefixes": ["600", "601", "603", "605", "000", "001", "002", "003"],
            },
            "fetch": {
                "status_counts": {str(key): int(value) for key, value in status_counts.items()},
                "coverage_pct": round(len(indicators) / len(universe) * 100, 2) if len(universe) else 0,
                "problem_rows": failures[["code", "name", "status", "last_date", "error"]].to_dict(
                    orient="records"
                ),
            },
            "signals": {name: int(len(frame)) for name, frame in signals.items()},
            "top_n_is_separate_from_full_results": True,
            "active_watchlist_count": int(len(active_watchlist)),
            "disclaimer": "Research output only; strategy thresholds require walk-forward validation.",
        }

