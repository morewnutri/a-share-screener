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
    max_workers: int = 12
    request_timeout: float = 12.0
    max_retries: int = 3
    close_buffer_minutes: int = 10
    universe_cache_hours: int = 24
    min_universe_size: int = 2000
    benchmark_secid: str = "1.000300"
    benchmark_name: str = "CSI300"
    force_refresh: bool = False


@dataclass(frozen=True)
class StrategyConfig:
    top_n: int = 100
    min_amount_ma20: float = 50_000_000
    max_setup_extension_ma20_pct: float = 8.0
    max_breakout_extension_ma20_pct: float = 12.0
    max_setup_return_5d_pct: float = 10.0
    max_breakout_return_5d_pct: float = 15.0
    setup_min_rs_percentile: float = 0.65
    breakout_min_rs_percentile: float = 0.70
    setup_contraction_distance_min_pct: float = -6.0
    setup_distance_max_pct: float = -0.2
    accumulation_distance_min_pct: float = -12.0
    contraction_ratio_max: float = 0.90
    contraction_min_count: int = 2
    accumulation_up_down_volume_min: float = 1.30
    accumulation_range_position_min: float = 0.60
    breakout_distance_max_pct: float = 3.0
    breakout_volume_ratio_min: float = 1.20
    breakout_volume_ratio_max: float = 4.0
    retest_volume_ratio_max: float = 1.10
    watchlist_ttl_sessions: int = 10


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
        if not 0 <= self.strategy.setup_min_rs_percentile <= 1:
            raise ValueError("setup_min_rs_percentile must be between 0 and 1.")
        if not 0 <= self.strategy.breakout_min_rs_percentile <= 1:
            raise ValueError("breakout_min_rs_percentile must be between 0 and 1.")
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

