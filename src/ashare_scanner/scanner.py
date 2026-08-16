from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import __version__
from .cache import HistoryCache, UniverseCache, atomic_write_csv, atomic_write_json
from .calendar import expected_complete_session
from .chips import CHIP_MODEL_NAME
from .config import AppConfig
from .datasource import BaoStockDataSource, EastmoneyDataSource, HybridDataSource
from .enrichment import merge_optional_evidence
from .fund_flow import (
    FUND_FLOW_FEATURE_COLUMNS,
    merge_fund_flow_features,
    rank_signal_by_fund_flow,
    summarize_fund_flow,
)
from .http import HttpClient
from .indicators import compute_indicators, latest_snapshot
from .reference import REFERENCE_CODES, REFERENCE_STOCKS
from .state import TRANSITION_COLUMNS, read_active_watchlist, update_watchlist
from .strategies import (
    SIGNAL_ORDER,
    add_relative_strength,
    apply_strategies,
    screening_diagnostics,
)


LOGGER = logging.getLogger(__name__)


class DailyScanner:
    def __init__(self, config: AppConfig, data_dir: str | Path | None = None) -> None:
        self.config = config
        self.data_dir = Path(data_dir or config.data_dir).expanduser().resolve()
        self.cache_dir = self.data_dir / "cache"
        self.http = HttpClient(
            config.data.request_timeout,
            config.data.max_retries,
            config.data.request_min_interval_seconds,
        )
        web_source = EastmoneyDataSource(self.http, config.data.fqt)
        baostock_source = BaoStockDataSource(max_retries=config.data.baostock_max_retries)
        self.source = HybridDataSource(web_source, baostock_source)
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
        benchmark_indicators = compute_indicators(benchmark, include_chips=False)
        benchmark_by_date = benchmark_indicators.set_index("date")
        benchmark_map20 = benchmark_by_date["return_20d_pct"]
        benchmark_map60 = benchmark_by_date["return_60d_pct"]
        benchmark_risk_map = (
            (benchmark_by_date["close"] >= benchmark_by_date["ma20"] * 0.97)
            & (benchmark_by_date["return_20d_pct"] > -8)
        ).astype(int)

        statuses, snapshots = self._scan_universe(universe, expected)

        status_frame = pd.DataFrame(statuses).sort_values("code").reset_index(drop=True)
        atomic_write_csv(status_frame, run_dir / "fetch_status.csv")
        if not snapshots:
            raise RuntimeError("No stock produced a valid latest-session indicator snapshot.")
        coverage_pct = len(snapshots) / len(universe) * 100 if len(universe) else 0.0
        coverage_valid = coverage_pct >= self.config.data.min_coverage_pct
        if not coverage_valid:
            LOGGER.error(
                "Scan coverage %.2f%% is below %.2f%%; candidate publication and state updates are disabled.",
                coverage_pct,
                self.config.data.min_coverage_pct,
            )

        indicators = pd.DataFrame(snapshots)
        indicators["date"] = pd.to_datetime(indicators["date"])
        indicators["benchmark_return_20d_pct"] = indicators["date"].map(benchmark_map20)
        indicators["benchmark_return_60d_pct"] = indicators["date"].map(benchmark_map60)
        indicators["benchmark_risk_ok"] = indicators["date"].map(benchmark_risk_map).fillna(1)
        context_columns = [
            column
            for column in (
                "code",
                "market_cap",
                "float_market_cap",
                "main_net_inflow_amount",
                "main_net_inflow_ratio_pct",
            )
            if column in universe.columns
        ]
        universe_context = universe[context_columns].copy()
        universe_context["code"] = universe_context["code"].astype(str).str.zfill(6)
        indicators = indicators.merge(universe_context, on="code", how="left")
        indicators, external_metadata = merge_optional_evidence(
            indicators,
            self.data_dir,
            expected,
        )
        indicators = add_relative_strength(indicators)
        scored, signals = apply_strategies(
            indicators,
            self.config.data,
            self.config.strategy,
        )
        if coverage_valid:
            scored, signals, fund_flow_status, fund_flow_metadata = self._enrich_fund_flow(
                scored,
                signals,
                expected,
            )
        else:
            fund_flow_status = pd.DataFrame(
                columns=("code", "name", "status", "rows", "last_date", "source", "error")
            )
            fund_flow_metadata = {
                "enabled": self.config.data.fund_flow_enabled,
                "requested_candidate_count": 0,
                "status_counts": {},
                "current_count": 0,
                "all_windows_positive_count": 0,
                "skipped_reason": "invalid_data_coverage",
            }
        atomic_write_csv(fund_flow_status, run_dir / "fund_flow_status.csv")
        funnel, near_misses = screening_diagnostics(
            scored,
            self.config.data,
            self.config.strategy,
        )
        atomic_write_csv(funnel, run_dir / "screening_funnel.csv")
        near_misses["date"] = pd.to_datetime(near_misses["date"]).dt.strftime("%Y-%m-%d")
        atomic_write_csv(near_misses.head(100), run_dir / "near_miss_top100.csv")
        reference_audit = self._build_reference_audit(
            universe,
            status_frame,
            scored,
            signals,
            near_misses,
            coverage_valid,
        )
        atomic_write_csv(reference_audit, run_dir / "reference_examples_audit.csv")
        scored["date"] = scored["date"].dt.strftime("%Y-%m-%d")
        atomic_write_csv(scored.sort_values("code"), run_dir / "indicators_scored.csv")

        for signal_name in SIGNAL_ORDER:
            full = signals[signal_name].copy()
            full["date"] = pd.to_datetime(full["date"]).dt.strftime("%Y-%m-%d")
            provisional_path = run_dir / f"provisional_{signal_name}.csv"
            if coverage_valid:
                published = full
                provisional_path.unlink(missing_ok=True)
            else:
                atomic_write_csv(full, provisional_path)
                published = full.iloc[0:0].copy()
            atomic_write_csv(published, run_dir / f"{signal_name}_all.csv")
            atomic_write_csv(
                published.head(self.config.strategy.top_n),
                run_dir / f"{signal_name}_top{self.config.strategy.top_n}.csv",
            )

        if coverage_valid:
            _, active_watchlist, new_transitions = update_watchlist(
                self.data_dir,
                signals,
                scored,
                expected,
                self.config.strategy.watchlist_ttl_sessions,
            )
        else:
            active_watchlist = read_active_watchlist(self.data_dir)
            new_transitions = pd.DataFrame(columns=TRANSITION_COLUMNS)
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
            funnel,
            external_metadata,
            fund_flow_metadata,
            coverage_valid,
            reference_audit,
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

    def _scan_universe(
        self,
        universe: pd.DataFrame,
        expected: date,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        records = universe.to_dict(orient="records")
        priority = [row for row in records if str(row.get("code", "")).zfill(6) in REFERENCE_CODES]
        regular = [row for row in records if str(row.get("code", "")).zfill(6) not in REFERENCE_CODES]
        statuses: list[dict[str, Any]] = []
        snapshots: list[dict[str, Any]] = []
        completed_total = 0

        for label, rows, worker_limit in (
            ("reference", priority, min(2, self.config.data.max_workers)),
            ("market", regular, self.config.data.max_workers),
        ):
            if not rows:
                continue
            LOGGER.info("Fetching %s group: %d stocks, workers=%d", label, len(rows), worker_limit)
            with ThreadPoolExecutor(max_workers=worker_limit) as executor:
                futures = {
                    executor.submit(
                        self._process_stock,
                        str(row.get("code", "")).zfill(6),
                        str(row.get("name", "")),
                        expected,
                        row,
                    ): (str(row.get("code", "")).zfill(6), str(row.get("name", "")))
                    for row in rows
                }
                for future in as_completed(futures):
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
                    completed_total += 1
                    if completed_total % 50 == 0 or completed_total == len(records):
                        ok_count = sum(row["status"] == "ok" for row in statuses)
                        LOGGER.info("Progress %d/%d, ok=%d", completed_total, len(records), ok_count)
        return statuses, snapshots

    def _get_universe(self, expected: date) -> tuple[pd.DataFrame, str, str]:
        cached = self.universe_cache.read(expected, self.config.data.universe_cache_hours)
        if cached.fresh and not self.config.data.force_refresh:
            return cached.frame, str(cached.metadata.get("source", "cache")), ""
        try:
            frame, source = self.source.fetch_universe(
                expected,
                self.config.data.min_universe_size,
            )
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

    def _process_stock(
        self,
        code: str,
        name: str,
        expected: date,
        market_context: dict[str, Any] | None = None,
    ) -> tuple[dict, dict | None]:
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
                context = market_context or {}
                cached_source = str(cached.metadata.get("source", ""))
                incremental = (
                    not cached.frame.empty
                    and cached_source.startswith("baostock:")
                    and not self.config.data.force_refresh
                )
                fetch_start = self.config.data.start_date
                if incremental:
                    fetch_start = (
                        cached.frame["date"].max().date()
                        - timedelta(days=self.config.data.incremental_refresh_days)
                    ).isoformat()
                history, source = self.source.fetch_stock_history(
                    code,
                    fetch_start,
                    expected,
                    latest_price=pd.to_numeric(context.get("latest"), errors="coerce"),
                    float_market_cap=pd.to_numeric(
                        context.get("float_market_cap"), errors="coerce"
                    ),
                    turnover_hint=pd.to_numeric(context.get("turnover"), errors="coerce"),
                )
                if incremental and source.startswith("baostock:"):
                    merged = self._merge_incremental_history(cached.frame, history)
                    if merged is None:
                        history, source = self.source.fetch_stock_history(
                            code,
                            self.config.data.start_date,
                            expected,
                            latest_price=pd.to_numeric(context.get("latest"), errors="coerce"),
                            float_market_cap=pd.to_numeric(
                                context.get("float_market_cap"), errors="coerce"
                            ),
                            turnover_hint=pd.to_numeric(
                                context.get("turnover"), errors="coerce"
                            ),
                        )
                    else:
                        history = merged
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

        indicator_frame = compute_indicators(history, chip_latest_only=True)
        snapshot = latest_snapshot(indicator_frame, code, name, source)
        status["status"] = "ok"
        return status, snapshot

    @staticmethod
    def _merge_incremental_history(
        cached: pd.DataFrame,
        refreshed: pd.DataFrame,
    ) -> pd.DataFrame | None:
        if cached.empty or refreshed.empty:
            return None
        old = cached.copy()
        new = refreshed.copy()
        old["date"] = pd.to_datetime(old["date"], errors="coerce")
        new["date"] = pd.to_datetime(new["date"], errors="coerce")
        overlap = old[["date", "close"]].merge(
            new[["date", "close"]], on="date", suffixes=("_old", "_new")
        )
        if len(overlap) < 3:
            return None
        old_close = pd.to_numeric(overlap["close_old"], errors="coerce")
        new_close = pd.to_numeric(overlap["close_new"], errors="coerce")
        comparable = old_close.notna() & new_close.notna()
        if comparable.sum() < 3 or not np.allclose(
            old_close[comparable],
            new_close[comparable],
            rtol=0.0015,
            atol=0.011,
        ):
            return None
        return (
            pd.concat([old, new], ignore_index=True)
            .drop_duplicates("date", keep="last")
            .sort_values("date")
            .reset_index(drop=True)
        )

    @staticmethod
    def _build_reference_audit(
        universe: pd.DataFrame,
        statuses: pd.DataFrame,
        scored: pd.DataFrame,
        signals: dict[str, pd.DataFrame],
        near_misses: pd.DataFrame,
        coverage_valid: bool,
    ) -> pd.DataFrame:
        audit = pd.DataFrame(REFERENCE_STOCKS, columns=("code", "reference_name"))
        market_names = universe[["code", "name"]].copy()
        market_names["code"] = market_names["code"].astype(str).str.zfill(6)
        market_names = market_names.rename(columns={"name": "market_name"})
        audit = audit.merge(market_names, on="code", how="left")
        status_columns = [
            "code",
            "status",
            "source",
            "from_cache",
            "bars",
            "last_date",
            "error",
        ]
        fetch = statuses[[column for column in status_columns if column in statuses.columns]].copy()
        fetch = fetch.rename(columns={"status": "fetch_status", "source": "history_source"})
        audit = audit.merge(fetch, on="code", how="left")
        scored_copy = scored.drop(columns=["name", "source", "bars"], errors="ignore").copy()
        audit = audit.merge(scored_copy, on="code", how="left")
        audit["name"] = audit["market_name"].fillna(audit["reference_name"])
        audit["in_universe"] = audit["market_name"].notna().astype(int)
        audit["indicators_ready"] = audit["close"].notna().astype(int)
        audit["scan_coverage_valid"] = int(coverage_valid)
        for signal_name in SIGNAL_ORDER:
            selected_codes = set(
                signals[signal_name]
                .get("code", pd.Series(dtype=str))
                .astype(str)
                .str.zfill(6)
            )
            audit[signal_name] = audit["code"].isin(selected_codes).astype(int)
        if not near_misses.empty:
            diagnostic_columns = [
                column for column in ("code", "closest_signal", "failed_at") if column in near_misses
            ]
            audit = audit.merge(
                near_misses[diagnostic_columns].drop_duplicates("code"),
                on="code",
                how="left",
            )
        leading = [
            "code",
            "name",
            "fetch_status",
            "history_source",
            "last_date",
            "bars",
            "error",
            "in_universe",
            "indicators_ready",
            "scan_coverage_valid",
            *SIGNAL_ORDER,
            "closest_signal",
            "failed_at",
        ]
        ordered = [column for column in leading if column in audit.columns]
        ordered.extend(column for column in audit.columns if column not in ordered)
        return audit[ordered]

    def _enrich_fund_flow(
        self,
        scored: pd.DataFrame,
        signals: dict[str, pd.DataFrame],
        expected: date,
    ) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame, dict[str, Any]]:
        status_columns = ("code", "name", "status", "rows", "last_date", "source", "error")
        candidate_rows = [
            frame[["code", "name"]]
            for frame in signals.values()
            if not frame.empty
        ]
        candidates = (
            pd.concat(candidate_rows, ignore_index=True).drop_duplicates("code")
            if candidate_rows
            else pd.DataFrame(columns=["code", "name"])
        )
        feature_rows: list[dict[str, Any]] = []
        statuses: list[dict[str, Any]] = []

        if self.config.data.fund_flow_enabled and not candidates.empty:
            workers = min(self.config.data.fund_flow_max_workers, len(candidates))
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(
                        self._get_stock_fund_flow,
                        row.code,
                        expected,
                    ): (str(row.code).zfill(6), row.name)
                    for row in candidates.itertuples(index=False)
                }
                for future in as_completed(futures):
                    code, name = futures[future]
                    try:
                        history, source = future.result()
                        summary = summarize_fund_flow(history, expected)
                        if not summary:
                            raise RuntimeError("fund-flow history has no usable rows")
                        feature_rows.append({"code": code, **summary})
                        statuses.append(
                            {
                                "code": code,
                                "name": name,
                                "status": "ok"
                                if summary["fund_flow_is_current"]
                                else "stale",
                                "rows": int(len(history)),
                                "last_date": summary["fund_flow_latest_date"],
                                "source": source,
                                "error": "",
                            }
                        )
                    except Exception as exc:
                        statuses.append(
                            {
                                "code": code,
                                "name": name,
                                "status": "failed",
                                "rows": 0,
                                "last_date": "",
                                "source": "",
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                        )

        features = pd.DataFrame(feature_rows, columns=("code", *FUND_FLOW_FEATURE_COLUMNS))
        status_frame = pd.DataFrame(statuses, columns=status_columns)
        if not status_frame.empty:
            status_frame = status_frame.sort_values("code").reset_index(drop=True)
        scored = merge_fund_flow_features(scored, features)
        model_scores = {
            "chip_base_ready": "chip_base_ready_score",
            "chip_base_launch": "chip_base_launch_score",
            "chip_base_rebound": "chip_base_rebound_score",
        }
        ranked: dict[str, pd.DataFrame] = {}
        for signal, frame in signals.items():
            enriched = merge_fund_flow_features(frame, features)
            ranked[signal] = rank_signal_by_fund_flow(enriched, model_scores[signal])

        status_counts = status_frame["status"].value_counts().to_dict() if not status_frame.empty else {}
        metadata = {
            "enabled": self.config.data.fund_flow_enabled,
            "requested_candidate_count": int(len(candidates)),
            "status_counts": {str(key): int(value) for key, value in status_counts.items()},
            "current_count": int(
                pd.to_numeric(features.get("fund_flow_is_current"), errors="coerce").fillna(0).sum()
            )
            if not features.empty
            else 0,
            "all_windows_positive_count": int(
                pd.to_numeric(
                    features.get("fund_flow_all_windows_positive"), errors="coerce"
                )
                .fillna(0)
                .sum()
            )
            if not features.empty
            else 0,
            "ranking_policy": "current_data,all_3_5_10_20_positive,positive_window_count,3d,5d,10d,20d,model_score",
            "failure_policy": "keep_candidate_and_fall_back_to_model_score",
        }
        return scored, ranked, status_frame, metadata

    def _get_stock_fund_flow(
        self,
        code: str,
        expected: date,
    ) -> tuple[pd.DataFrame, str]:
        cache_path = self.cache_dir / "fund_flow" / f"{str(code).zfill(6)}.csv"
        cached = pd.DataFrame()
        if cache_path.exists():
            try:
                cached = pd.read_csv(cache_path, dtype={"code": str}, parse_dates=["date"])
            except (OSError, ValueError, pd.errors.EmptyDataError):
                cached = pd.DataFrame()
        if (
            not cached.empty
            and cached["date"].max().date() == expected
            and not self.config.data.force_refresh
        ):
            return cached, "fund_flow_cache"

        try:
            history, source = self.source.fetch_stock_fund_flow(
                code,
                self.config.data.fund_flow_limit,
            )
            atomic_write_csv(history, cache_path)
            return history, source
        except Exception:
            if not cached.empty:
                return cached, "stale_fund_flow_cache"
            raise
        finally:
            time.sleep(self.config.data.fund_flow_request_pause_seconds)

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
        funnel: pd.DataFrame,
        external_metadata: dict[str, Any],
        fund_flow_metadata: dict[str, Any],
        coverage_valid: bool,
        reference_audit: pd.DataFrame,
        elapsed_seconds: float,
    ) -> dict[str, Any]:
        status_counts = statuses["status"].value_counts().to_dict()
        source_counts = (
            statuses.loc[statuses["status"] == "ok", "source"]
            .replace("", "unknown")
            .value_counts()
            .to_dict()
        )
        failures = statuses[statuses["status"].isin(["failed", "stale_hist", "empty_hist"])]
        candidate_codes = {
            str(code).zfill(6)
            for frame in signals.values()
            for code in frame.get("code", pd.Series(dtype=str)).tolist()
        }
        published_candidate_count = len(candidate_codes) if coverage_valid else 0
        candidate_rate = (
            published_candidate_count / len(indicators) * 100 if len(indicators) else 0.0
        )
        coverage = len(indicators) / len(universe) * 100 if len(universe) else 0.0
        if not coverage_valid:
            assessment = "数据覆盖率低于90%，应先排查抓取失败，不能据此判断策略或市场。"
        elif candidate_rate < 0.2:
            assessment = "候选率低于0.2%。先看筛选漏斗：主要卡在阶段评分通常表示阈值偏严；大量股票卡在低位、平台或筹码峰结构，才更可能是当日匹配度低。"
        elif candidate_rate < 1.0:
            assessment = "候选率低于1%，属于选择性较强的结果；结合三个阶段的近似入选股和分项得分判断，不要只看数量。"
        else:
            assessment = "候选数量不低，但数量不代表有效性，仍需用滚动回测检查命中率、回撤和不同市场阶段的稳定性。"
        funnel_records = [
            {
                "signal": row.signal,
                "step_number": int(row.step_number),
                "step": row.step,
                "remaining_count": int(row.remaining_count),
                "retention_from_previous_pct": float(row.retention_from_previous_pct),
                "retention_from_all_pct": float(row.retention_from_all_pct),
            }
            for row in funnel.itertuples(index=False)
        ]
        return {
            "program_version": __version__,
            "run_time": datetime.now().astimezone().isoformat(timespec="seconds"),
            "expected_complete_session": expected.isoformat(),
            "elapsed_seconds": round(elapsed_seconds, 2),
            "config": {
                "data": asdict(self.config.data),
                "strategy": asdict(self.config.strategy),
            },
            "data_policy": {
                "stock_adjustment": "forward_adjusted_fqt_1",
                "history_sources": "baostock_qfq_primary,tencent_qfq_secondary,eastmoney_qfq_fallback",
                "baostock_turnover": "provider_reported_historical_turnover",
                "tencent_turnover": "inferred_from_same_quote_float_volume_basis",
                "tencent_amount": "estimated_from_ohlc_typical_price_and_volume",
                "chip_distribution": CHIP_MODEL_NAME,
                "chip_distribution_is_account_level_data": False,
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
                "source_counts": {str(key): int(value) for key, value in source_counts.items()},
                "coverage_pct": round(coverage, 2),
                "problem_rows": failures[["code", "name", "status", "last_date", "error"]].to_dict(
                    orient="records"
                ),
            },
            "signals": {
                name: int(len(frame)) if coverage_valid else 0 for name, frame in signals.items()
            },
            "provisional_signals": {name: int(len(frame)) for name, frame in signals.items()},
            "external_evidence": external_metadata,
            "fund_flow": fund_flow_metadata,
            "screening": {
                "valid": coverage_valid,
                "minimum_coverage_pct": self.config.data.min_coverage_pct,
                "unique_candidate_count": int(published_candidate_count),
                "provisional_unique_candidate_count": int(len(candidate_codes)),
                "candidate_rate_pct": round(candidate_rate, 3),
                "assessment": assessment,
                "funnel": funnel_records,
            },
            "reference_audit": {
                "count": int(len(reference_audit)),
                "indicators_ready": int(reference_audit["indicators_ready"].sum()),
                "fetch_status_counts": {
                    str(key): int(value)
                    for key, value in reference_audit["fetch_status"].fillna("not_in_universe").value_counts().items()
                },
            },
            "top_n_is_separate_from_full_results": True,
            "active_watchlist_count": int(len(active_watchlist)),
            "disclaimer": "Research output only; strategy thresholds require walk-forward validation.",
        }
