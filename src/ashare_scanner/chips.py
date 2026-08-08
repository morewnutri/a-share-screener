from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd


CHIP_MODEL_NAME = "modeled_turnover_triangle_250d_v2"
DEFAULT_CHIP_LOOKBACK = 250
DEFAULT_CHIP_BINS = 2048
PRICE_GRID_MIN = 0.001
PRICE_GRID_MAX = 100_000.0
CHIP_FEATURE_COLUMNS = (
    "chip_peak_price",
    "chip_secondary_peak_price",
    "chip_peak_band_share_pct",
    "chip_peak_dominance",
    "chip_significant_peak_count",
    "chip_70_low",
    "chip_70_high",
    "chip_70_width_pct",
    "chip_90_low",
    "chip_90_high",
    "chip_90_width_pct",
    "chip_profit_ratio_pct",
    "chip_overhead_ratio_pct",
    "chip_low_zone_share_pct",
    "chip_peak_position",
    "chip_peak_distance_pct",
    "chip_model_coverage_pct",
)


def _price_grid(bins: int) -> tuple[np.ndarray, np.ndarray]:
    centers = np.geomspace(PRICE_GRID_MIN, PRICE_GRID_MAX, bins)
    edges = np.empty(bins + 1)
    edges[1:-1] = np.sqrt(centers[:-1] * centers[1:])
    half_ratio = np.sqrt(centers[1] / centers[0])
    edges[0] = centers[0] / half_ratio
    edges[-1] = centers[-1] * half_ratio
    return centers, edges


def _triangular_cdf(
    values: np.ndarray,
    low: float,
    mode: float,
    high: float,
) -> np.ndarray:
    result = np.zeros_like(values, dtype=float)
    result[values >= high] = 1.0
    if high - low <= 1e-12:
        return result
    mode = float(np.clip(mode, low, high))
    left = (values > low) & (values <= mode)
    right = (values > mode) & (values < high)
    if mode - low > 1e-12:
        result[left] = (values[left] - low) ** 2 / ((high - low) * (mode - low))
    if high - mode > 1e-12:
        result[right] = 1 - (high - values[right]) ** 2 / (
            (high - low) * (high - mode)
        )
    else:
        result[right] = 1.0
    return np.clip(result, 0.0, 1.0)


def _new_chip_mass_sparse(
    centers: np.ndarray,
    edges: np.ndarray,
    low: float,
    mode: float,
    high: float,
) -> tuple[np.ndarray, np.ndarray]:
    if high - low <= 1e-8:
        index = int(np.argmin(np.abs(centers - mode)))
        return np.array([index]), np.array([1.0])

    start = int(np.searchsorted(edges, low, side="right") - 1)
    stop = int(np.searchsorted(edges, high, side="left"))
    start = int(np.clip(start, 0, len(centers) - 1))
    stop = int(np.clip(stop, start + 1, len(centers)))
    selected_edges = edges[start : stop + 1]
    values = np.diff(_triangular_cdf(selected_edges, low, mode, high))
    values = np.clip(values, 0.0, None)
    total = float(values.sum())
    if total <= 0:
        index = int(np.argmin(np.abs(centers - mode)))
        return np.array([index]), np.array([1.0])
    return np.arange(start, stop), values / total


def _quantile_price(centers: np.ndarray, mass: np.ndarray, quantile: float) -> float:
    cumulative = np.cumsum(mass)
    index = int(np.searchsorted(cumulative, quantile, side="left"))
    return float(centers[min(index, len(centers) - 1)])


def _peak_summary(
    centers: np.ndarray,
    mass: np.ndarray,
) -> tuple[float, float, float, float, int]:
    kernel = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
    kernel /= kernel.sum()
    smooth = np.convolve(mass, kernel, mode="same")
    primary_index = int(np.argmax(smooth))
    primary_value = float(smooth[primary_index])
    local = np.flatnonzero(
        (smooth >= np.r_[smooth[0], smooth[:-1]])
        & (smooth >= np.r_[smooth[1:], smooth[-1]])
        & (smooth >= primary_value * 0.20)
    )
    neighbor_low = max(0, primary_index - 1)
    neighbor_high = min(len(centers) - 1, primary_index + 1)
    local_step = (centers[neighbor_high] - centers[neighbor_low]) / max(
        1,
        neighbor_high - neighbor_low,
    )
    min_separation = max(3, int(round(centers[primary_index] * 0.04 / local_step)))
    selected: list[int] = []
    for index in local[np.argsort(smooth[local])[::-1]]:
        if all(abs(int(index) - other) >= min_separation for other in selected):
            selected.append(int(index))
    secondary_index = selected[1] if len(selected) > 1 else primary_index
    secondary_value = float(smooth[secondary_index]) if len(selected) > 1 else 0.0
    dominance = primary_value / secondary_value if secondary_value > 0 else np.inf
    peak_price = float(centers[primary_index])
    peak_band = (centers >= peak_price * 0.96) & (centers <= peak_price * 1.04)
    return (
        peak_price,
        float(centers[secondary_index]) if len(selected) > 1 else np.nan,
        float(mass[peak_band].sum() * 100),
        float(dominance),
        len(selected),
    )


def _rebase_entries(
    entries: deque[tuple[int, np.ndarray, np.ndarray, float]],
    survival_scale: float,
) -> deque[tuple[int, np.ndarray, np.ndarray, float]]:
    rebased: deque[tuple[int, np.ndarray, np.ndarray, float]] = deque()
    for day, indices, insertion, insertion_scale in entries:
        rebased.append(
            (day, indices, insertion * (survival_scale / insertion_scale), 1.0)
        )
    return rebased


def compute_modeled_cyq(
    history: pd.DataFrame,
    bins: int = DEFAULT_CHIP_BINS,
    lookback: int = DEFAULT_CHIP_LOOKBACK,
    latest_only: bool = False,
) -> pd.DataFrame:
    """Estimate a rolling cost distribution from OHLC and turnover.

    This is a modeled CYQ profile, not account-level holder cost data. Within
    the rolling window, old mass decays proportionally to turnover and new mass
    follows an intraday triangular distribution centered on the OHLC average.
    """
    if bins < 300:
        raise ValueError("bins must be at least 300")
    if lookback < 60:
        raise ValueError("lookback must be at least 60")

    frame = history.reset_index(drop=True)
    output = pd.DataFrame(np.nan, index=frame.index, columns=CHIP_FEATURE_COLUMNS)
    output["chip_model"] = ""
    if frame.empty:
        return output

    highs = pd.to_numeric(frame["high"], errors="coerce").to_numpy(dtype=float)
    lows = pd.to_numeric(frame["low"], errors="coerce").to_numpy(dtype=float)
    opens = pd.to_numeric(frame["open"], errors="coerce").to_numpy(dtype=float)
    closes = pd.to_numeric(frame["close"], errors="coerce").to_numpy(dtype=float)
    turnover = pd.to_numeric(frame["turnover"], errors="coerce").to_numpy(dtype=float)

    centers, edges = _price_grid(bins)
    mass = np.zeros(bins, dtype=float)
    entries: deque[tuple[int, np.ndarray, np.ndarray, float]] = deque()
    survival_scale = 1.0

    for index, (open_, high, low, close, turnover_pct) in enumerate(
        zip(opens, highs, lows, closes, turnover, strict=True)
    ):
        if not np.all(np.isfinite([open_, high, low, close, turnover_pct])):
            continue
        if low <= 0 or high < low:
            continue

        rate = float(np.clip(turnover_pct / 100, 0.0, 1.0))
        retained = 1 - rate
        if retained <= 1e-12:
            mass.fill(0.0)
            entries.clear()
            survival_scale = 1.0
        else:
            mass *= retained
            survival_scale *= retained
            if survival_scale < 1e-80:
                entries = _rebase_entries(entries, survival_scale)
                survival_scale = 1.0

        mode = (open_ + high + low + close) / 4
        indices, distribution = _new_chip_mass_sparse(
            centers,
            edges,
            low,
            mode,
            high,
        )
        insertion = distribution * rate
        if rate > 0:
            mass[indices] += insertion
            entries.append((index, indices, insertion, survival_scale))

        first_kept_day = index - lookback + 1
        while entries and entries[0][0] < first_kept_day:
            _, old_indices, old_insertion, insertion_scale = entries.popleft()
            mass[old_indices] -= old_insertion * (survival_scale / insertion_scale)
        np.maximum(mass, 0.0, out=mass)

        coverage = float(mass.sum())
        if coverage <= 1e-8:
            continue
        if latest_only and index != len(frame) - 1:
            continue
        normalized = mass / coverage

        low70 = _quantile_price(centers, normalized, 0.15)
        high70 = _quantile_price(centers, normalized, 0.85)
        low90 = _quantile_price(centers, normalized, 0.05)
        high90 = _quantile_price(centers, normalized, 0.95)
        peak, secondary, band_share, dominance, peak_count = _peak_summary(
            centers,
            normalized,
        )
        window_start = max(0, index - lookback + 1)
        observed_low = float(np.nanmin(lows[window_start : index + 1]))
        observed_high = float(np.nanmax(highs[window_start : index + 1]))
        price_span = observed_high - observed_low
        low_zone_limit = observed_low + price_span * 0.35

        output.loc[index, CHIP_FEATURE_COLUMNS] = (
            peak,
            secondary,
            band_share,
            dominance,
            peak_count,
            low70,
            high70,
            (high70 / low70 - 1) * 100 if low70 > 0 else np.nan,
            low90,
            high90,
            (high90 / low90 - 1) * 100 if low90 > 0 else np.nan,
            float(normalized[centers <= close].sum() * 100),
            float(normalized[centers > close].sum() * 100),
            float(normalized[centers <= low_zone_limit].sum() * 100),
            (peak - observed_low) / price_span if price_span > 0 else 0.5,
            (close / peak - 1) * 100 if peak > 0 else np.nan,
            coverage * 100,
        )
        output.loc[index, "chip_model"] = CHIP_MODEL_NAME

    return output
