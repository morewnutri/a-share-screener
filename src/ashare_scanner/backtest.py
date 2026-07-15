from __future__ import annotations

import json
import logging
import re
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .cache import HistoryCache, UniverseCache, atomic_write_csv, atomic_write_json
from .config import AppConfig
from .indicators import compute_indicators
from .strategies import SIGNAL_ORDER, add_factor_scores, add_relative_strength, signal_masks


LOGGER = logging.getLogger(__name__)
CODE_FILE = re.compile(r"^(\d{6})\.csv$")


def _outcomes(
    history: pd.DataFrame,
    horizon: int,
    target_pct: float,
    max_drawdown_pct: float,
) -> pd.DataFrame:
    count = len(history)
    success = np.full(count, np.nan)
    max_return = np.full(count, np.nan)
    drawdown_to_hit = np.full(count, np.nan)
    opens = history["open"].to_numpy(dtype=float)
    highs = history["high"].to_numpy(dtype=float)
    lows = history["low"].to_numpy(dtype=float)
    for index in range(count - horizon):
        entry = opens[index + 1]
        if not np.isfinite(entry) or entry <= 0:
            continue
        future_highs = highs[index + 1 : index + 1 + horizon]
        future_lows = lows[index + 1 : index + 1 + horizon]
        returns = (future_highs / entry - 1) * 100
        max_return[index] = np.nanmax(returns)
        hit = np.flatnonzero(returns >= target_pct)
        end = int(hit[0]) + 1 if len(hit) else horizon
        drawdown = np.nanmin((future_lows[:end] / entry - 1) * 100)
        drawdown_to_hit[index] = drawdown
        success[index] = float(bool(len(hit)) and drawdown >= -max_drawdown_pct)
    return pd.DataFrame(
        {
            "outcome_success": success,
            "forward_max_return_pct": max_return,
            "drawdown_before_target_pct": drawdown_to_hit,
        }
    )


class Backtester:
    def __init__(self, config: AppConfig, data_dir: str | Path | None = None) -> None:
        self.config = config
        self.data_dir = Path(data_dir or config.data_dir).expanduser().resolve()
        self.cache = HistoryCache(
            self.data_dir / "cache",
            config.data.start_date,
            config.data.fqt,
        )

    def run(self, start: date, end: date) -> Path:
        benchmark_key = f"benchmark_{self.config.data.benchmark_secid}"
        benchmark_read = self.cache.read(benchmark_key, end, source_kind="benchmark")
        if benchmark_read.frame.empty:
            raise RuntimeError("Benchmark cache is missing. Run the daily scanner first.")
        benchmark = compute_indicators(benchmark_read.frame).set_index("date")

        universe_cache = UniverseCache(self.data_dir / "cache")
        universe_read = universe_cache.read(end, max_age_hours=10**6)
        names = {}
        if not universe_read.frame.empty:
            names = dict(zip(universe_read.frame["code"], universe_read.frame["name"]))

        panels: list[pd.DataFrame] = []
        history_dir = self.data_dir / "cache" / "history"
        files = [path for path in history_dir.glob("*.csv") if CODE_FILE.match(path.name)]
        if not files:
            raise RuntimeError("No stock history cache found. Run the daily scanner first.")
        for index, path in enumerate(files, start=1):
            code = CODE_FILE.match(path.name).group(1)  # type: ignore[union-attr]
            try:
                raw = pd.read_csv(path, dtype={"code": str}, parse_dates=["date"])
                features = compute_indicators(raw)
                outcomes = _outcomes(
                    features,
                    self.config.backtest.horizon_sessions,
                    self.config.backtest.target_return_pct,
                    self.config.backtest.max_drawdown_pct,
                )
                features = pd.concat([features.reset_index(drop=True), outcomes], axis=1)
                features["bars"] = np.arange(1, len(features) + 1)
                features["code"] = code
                features["name"] = names.get(code, "")
                features = features[
                    (features["date"].dt.date >= start) & (features["date"].dt.date <= end)
                ]
                if not features.empty:
                    panels.append(features)
            except Exception as exc:
                LOGGER.warning("Skipping backtest cache %s: %s", path.name, exc)
            if index % 200 == 0:
                LOGGER.info("Prepared %d/%d cached histories", index, len(files))

        if not panels:
            raise RuntimeError("No eligible cached rows in the requested backtest range.")
        panel = pd.concat(panels, ignore_index=True)
        panel["benchmark_return_20d_pct"] = panel["date"].map(benchmark["return_20d_pct"])
        panel["benchmark_return_60d_pct"] = panel["date"].map(benchmark["return_60d_pct"])
        panel = add_relative_strength(panel)
        panel = add_factor_scores(panel)
        masks = signal_masks(panel, self.config.data, self.config.strategy)
        for signal_name in SIGNAL_ORDER:
            panel[signal_name] = masks[signal_name]
        panel["any_signal"] = panel[list(SIGNAL_ORDER)].any(axis=1)
        panel["signal_names"] = panel.apply(
            lambda row: ",".join(name for name in SIGNAL_ORDER if row[name]),
            axis=1,
        )

        eligible = (
            (panel["bars"] >= self.config.data.min_history_bars)
            & (panel["amount_ma20_prev"] >= self.config.strategy.min_amount_ma20)
            & (panel["one_price_limit"] == 0)
            & panel["outcome_success"].notna()
            & panel["rs20_percentile"].notna()
        )
        positives = eligible & (panel["outcome_success"] == 1)
        signals = eligible & panel["any_signal"]
        successful_signals = signals & (panel["outcome_success"] == 1)

        metrics: dict = {
            "period": {"start": start.isoformat(), "end": end.isoformat()},
            "definition": {
                "entry": "next_session_open",
                "horizon_sessions": self.config.backtest.horizon_sessions,
                "target_return_pct": self.config.backtest.target_return_pct,
                "max_drawdown_before_target_pct": self.config.backtest.max_drawdown_pct,
            },
            "observation_level": {
                "eligible_rows": int(eligible.sum()),
                "positive_rows": int(positives.sum()),
                "signal_rows": int(signals.sum()),
                "successful_signal_rows": int(successful_signals.sum()),
                "precision": round(float(successful_signals.sum() / signals.sum()), 4)
                if signals.sum()
                else None,
                "recall": round(float(successful_signals.sum() / positives.sum()), 4)
                if positives.sum()
                else None,
            },
            "by_signal": {},
            "top_k_recall": {},
            "limitations": [
                "Uses the current cached universe and therefore has survivorship bias.",
                "Overlapping stock-date opportunities are counted as separate observations.",
                "Does not model limit-up queue depth, commissions, slippage, or corporate-action availability dates.",
            ],
        }
        for signal_name in SIGNAL_ORDER:
            mask = eligible & panel[signal_name]
            wins = mask & (panel["outcome_success"] == 1)
            metrics["by_signal"][signal_name] = {
                "count": int(mask.sum()),
                "successful": int(wins.sum()),
                "precision": round(float(wins.sum() / mask.sum()), 4) if mask.sum() else None,
            }

        ranked_signals = panel.loc[signals].copy()
        ranked_signals["daily_rank"] = ranked_signals.groupby("date")["score_total"].rank(
            method="first", ascending=False
        )
        for top_k in self.config.backtest.top_k:
            covered = ranked_signals[
                (ranked_signals["daily_rank"] <= top_k)
                & (ranked_signals["outcome_success"] == 1)
            ]
            metrics["top_k_recall"][str(top_k)] = (
                round(float(len(covered) / positives.sum()), 4) if positives.sum() else None
            )

        daily = panel.loc[eligible].groupby("date").agg(
            eligible=("code", "size"),
            opportunities=("outcome_success", "sum"),
            signals=("any_signal", "sum"),
        )
        daily_wins = panel.loc[successful_signals].groupby("date").size().rename("successful_signals")
        daily = daily.join(daily_wins, how="left").fillna({"successful_signals": 0}).reset_index()

        output_dir = self.data_dir / "backtests" / datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir.mkdir(parents=True, exist_ok=True)
        signal_columns = [
            "date",
            "code",
            "name",
            "signal_names",
            "score_total",
            "daily_rank",
            "close",
            "rs20_percentile",
            "outcome_success",
            "forward_max_return_pct",
            "drawdown_before_target_pct",
        ]
        atomic_write_csv(ranked_signals[signal_columns], output_dir / "signals_with_outcomes.csv")
        atomic_write_csv(daily, output_dir / "daily_summary.csv")
        atomic_write_json(metrics, output_dir / "metrics.json")
        return output_dir
