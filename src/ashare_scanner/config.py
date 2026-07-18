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
    min_amount_ma20: float = 30_000_000
    accumulation_score_min: float = 55.0
    accumulation_min_evidence_groups: int = 3
    accumulation_max_position_250: float = 0.80
    accumulation_max_return_20d_pct: float = 30.0
    accumulation_max_extension_ma60_pct: float = 18.0
    accumulation_max_distribution_days_5: int = 2
    main_wave_score_min: float = 58.0
    main_wave_max_extension_ma20_pct: float = 16.0
    main_wave_max_return_10d_pct: float = 25.0
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
        for value, name in (
            (self.strategy.accumulation_score_min, "accumulation_score_min"),
            (self.strategy.main_wave_score_min, "main_wave_score_min"),
        ):
            if not 0 <= value <= 100:
                raise ValueError(f"{name} must be between 0 and 100.")
        if not 1 <= self.strategy.accumulation_min_evidence_groups <= 5:
            raise ValueError("accumulation_min_evidence_groups must be between 1 and 5.")
        if not 0 < self.strategy.accumulation_max_position_250 <= 1:
            raise ValueError("accumulation_max_position_250 must be in (0, 1].")
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