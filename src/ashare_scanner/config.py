from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DataConfig:
    start_date: str = "2022-01-01"
    fqt: int = 1
    min_history_bars: int = 150
    max_workers: int = 4
    request_timeout: float = 12.0
    max_retries: int = 2
    request_min_interval_seconds: float = 0.12
    baostock_max_retries: int = 2
    incremental_refresh_days: int = 30
    min_coverage_pct: float = 90.0
    close_buffer_minutes: int = 10
    universe_cache_hours: int = 24
    min_universe_size: int = 2000
    benchmark_secid: str = "1.000300"
    benchmark_name: str = "CSI300"
    force_refresh: bool = False
    fund_flow_enabled: bool = True
    fund_flow_max_workers: int = 2
    fund_flow_max_candidates: int = 120
    fund_flow_limit: int = 100
    fund_flow_request_timeout: float = 5.0
    fund_flow_max_hosts: int = 2
    fund_flow_request_pause_seconds: float = 0.40
    fund_flow_progress_every: int = 10
    fund_flow_failure_streak_limit: int = 10
    fund_flow_stage_timeout_minutes: float = 20.0


@dataclass(frozen=True)
class StrategyConfig:
    top_n: int = 100
    min_amount_ma20: float = 30_000_000
    chip_base_ready_score_min: float = 60.0
    chip_base_launch_score_min: float = 65.0
    chip_base_rebound_score_min: float = 62.0
    chip_max_position_250: float = 0.92
    chip_max_base_width_pct: float = 42.0
    chip_max_base_abs_return_pct: float = 22.0
    chip_max_70_width_pct: float = 45.0
    chip_max_peak_position: float = 0.92
    chip_min_peak_band_share_pct: float = 10.0
    chip_strong_peak_band_share_pct: float = 25.0
    chip_strong_peak_max_70_width_pct: float = 60.0
    chip_min_low_zone_share_pct: float = 42.0
    chip_ready_max_peak_distance_pct: float = 30.0
    chip_launch_max_peak_distance_pct: float = 60.0
    chip_launch_max_return_10d_pct: float = 55.0
    chip_max_distribution_days_5: int = 3
    chip_rebound_max_position_250: float = 0.98
    chip_rebound_max_base_width_pct: float = 48.0
    chip_rebound_max_70_width_pct: float = 50.0
    chip_rebound_max_peak_position: float = 0.96
    chip_rebound_min_peak_band_share_pct: float = 8.0
    chip_rebound_min_low_zone_share_pct: float = 30.0
    chip_rebound_max_peak_distance_pct: float = 90.0
    chip_rebound_max_return_20d_pct: float = 75.0
    chip_rebound_min_return_5d_pct: float = 3.0
    watchlist_ttl_sessions: int = 12


@dataclass(frozen=True)
class BacktestConfig:
    horizon_sessions: int = 10
    target_return_pct: float = 12.0
    max_drawdown_pct: float = 6.0
    top_k: tuple[int, ...] = (20, 50, 100)


@dataclass(frozen=True)
class AppConfig:
    data_dir: str = "data"
    data: DataConfig = field(default_factory=DataConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)

    def validate(self) -> None:
        if self.data.fqt != 1:
            raise ValueError("This project only accepts fqt=1 (forward-adjusted) stock data.")
        if self.data.min_history_bars < 120:
            raise ValueError("min_history_bars must be at least 120 for the configured indicators.")
        if self.data.max_workers < 1:
            raise ValueError("max_workers must be positive.")
        if self.data.request_min_interval_seconds < 0:
            raise ValueError("request_min_interval_seconds must not be negative.")
        if self.data.baostock_max_retries < 1:
            raise ValueError("baostock_max_retries must be positive.")
        if self.data.incremental_refresh_days < 10:
            raise ValueError("incremental_refresh_days must be at least 10.")
        if not 0 < self.data.min_coverage_pct <= 100:
            raise ValueError("min_coverage_pct must be in (0, 100].")
        if (
            self.data.fund_flow_max_workers < 1
            or self.data.fund_flow_max_candidates < 1
            or self.data.fund_flow_limit < 20
        ):
            raise ValueError(
                "fund-flow workers/candidate cap must be positive and history limit at least 20."
            )
        if self.data.fund_flow_request_timeout <= 0 or self.data.fund_flow_max_hosts < 1:
            raise ValueError("fund-flow timeout and host limit must be positive.")
        if self.data.fund_flow_request_pause_seconds < 0:
            raise ValueError("fund_flow_request_pause_seconds must not be negative.")
        if (
            self.data.fund_flow_progress_every < 1
            or self.data.fund_flow_failure_streak_limit < 1
            or self.data.fund_flow_stage_timeout_minutes <= 0
        ):
            raise ValueError("fund-flow progress, failure streak, and stage timeout must be positive.")
        for value, name in (
            (self.strategy.chip_base_ready_score_min, "chip_base_ready_score_min"),
            (self.strategy.chip_base_launch_score_min, "chip_base_launch_score_min"),
            (self.strategy.chip_base_rebound_score_min, "chip_base_rebound_score_min"),
        ):
            if not 0 <= value <= 100:
                raise ValueError(f"{name} must be between 0 and 100.")
        if (
            self.strategy.chip_strong_peak_band_share_pct
            < self.strategy.chip_min_peak_band_share_pct
            or self.strategy.chip_strong_peak_max_70_width_pct
            < self.strategy.chip_max_70_width_pct
        ):
            raise ValueError(
                "strong-peak thresholds must be no stricter than the standard chip thresholds."
            )
        for value, name in (
            (self.strategy.chip_max_position_250, "chip_max_position_250"),
            (self.strategy.chip_max_peak_position, "chip_max_peak_position"),
            (self.strategy.chip_rebound_max_position_250, "chip_rebound_max_position_250"),
            (self.strategy.chip_rebound_max_peak_position, "chip_rebound_max_peak_position"),
        ):
            if not 0 < value <= 1:
                raise ValueError(f"{name} must be in (0, 1].")
        if self.backtest.horizon_sessions < 1:
            raise ValueError("horizon_sessions must be positive.")


def _section(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key, {})
    if not isinstance(value, dict):
        raise ValueError(f"Config section '{key}' must be a mapping.")
    return value


def load_config(path: str | Path) -> AppConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError("The config root must be a mapping.")

    backtest_raw = _section(raw, "backtest")
    if "top_k" in backtest_raw:
        backtest_raw = {**backtest_raw, "top_k": tuple(int(x) for x in backtest_raw["top_k"])}

    config = AppConfig(
        data_dir=str(raw.get("data_dir", "data")),
        data=DataConfig(**_section(raw, "data")),
        strategy=StrategyConfig(**_section(raw, "strategy")),
        backtest=BacktestConfig(**backtest_raw),
    )
    config.validate()
    return config
