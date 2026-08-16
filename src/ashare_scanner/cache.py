from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd


HISTORY_SCHEMA_VERSION = 4
UNIVERSE_SCHEMA_VERSION = 2


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    os.replace(temporary, path)


def _atomic_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


@dataclass
class CacheRead:
    frame: pd.DataFrame
    metadata: dict[str, Any]
    fresh: bool
    reason: str


class HistoryCache:
    def __init__(self, root: Path, start_date: str, fqt: int) -> None:
        self.root = root / "history"
        self.start_date = start_date
        self.fqt = fqt

    def _paths(self, key: str) -> tuple[Path, Path]:
        safe_key = key.replace(".", "_").replace("/", "_")
        return self.root / f"{safe_key}.csv", self.root / f"{safe_key}.meta.json"

    def read(self, key: str, expected: date, source_kind: str = "stock") -> CacheRead:
        csv_path, meta_path = self._paths(key)
        if not csv_path.exists() or not meta_path.exists():
            return CacheRead(pd.DataFrame(), {}, False, "missing")
        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            expected_fqt = self.fqt if source_kind == "stock" else 0
            compatible = (
                metadata.get("schema_version") == HISTORY_SCHEMA_VERSION
                and metadata.get("source_kind") == source_kind
                and metadata.get("source", "").startswith(
                    ("eastmoney:", "tencent:", "baostock:")
                )
                and int(metadata.get("fqt", -1)) == expected_fqt
                and metadata.get("start_date") == self.start_date
            )
            if not compatible:
                return CacheRead(pd.DataFrame(), metadata, False, "incompatible_metadata")
            frame = pd.read_csv(csv_path, dtype={"code": str}, parse_dates=["date"])
            frame = frame[frame["date"].dt.date <= expected].copy()
            if frame.empty:
                return CacheRead(frame, metadata, False, "empty")
            last_date = frame["date"].max().date()
            fresh = last_date == expected
            return CacheRead(frame, metadata, fresh, "fresh" if fresh else "stale_date")
        except Exception as exc:
            return CacheRead(pd.DataFrame(), {}, False, f"corrupt:{type(exc).__name__}")

    def write(
        self,
        key: str,
        frame: pd.DataFrame,
        source: str,
        source_kind: str,
        expected: date,
    ) -> None:
        csv_path, meta_path = self._paths(key)
        clean = frame.copy()
        clean["date"] = pd.to_datetime(clean["date"])
        clean = clean[clean["date"].dt.date <= expected]
        clean = clean.drop_duplicates("date", keep="last").sort_values("date")
        if clean.empty:
            raise ValueError("Refusing to cache an empty history frame.")
        fqt = self.fqt if source_kind == "stock" else 0
        metadata = {
            "key": key,
            "source": source,
            "source_kind": source_kind,
            "fqt": fqt,
            "schema_version": HISTORY_SCHEMA_VERSION,
            "start_date": self.start_date,
            "last_complete_date": clean["date"].max().strftime("%Y-%m-%d"),
            "saved_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        }
        _atomic_csv(clean, csv_path)
        _atomic_json(metadata, meta_path)


class UniverseCache:
    def __init__(self, root: Path) -> None:
        self.csv_path = root / "universe" / "mainboard.csv"
        self.meta_path = root / "universe" / "mainboard.meta.json"

    def read(self, expected: date, max_age_hours: int) -> CacheRead:
        if not self.csv_path.exists() or not self.meta_path.exists():
            return CacheRead(pd.DataFrame(), {}, False, "missing")
        try:
            metadata = json.loads(self.meta_path.read_text(encoding="utf-8"))
            frame = pd.read_csv(self.csv_path, dtype={"code": str})
            saved_at = datetime.fromisoformat(metadata["saved_at"])
            age_ok = datetime.now().astimezone() - saved_at < timedelta(hours=max_age_hours)
            fresh = (
                metadata.get("schema_version") == UNIVERSE_SCHEMA_VERSION
                and metadata.get("as_of") == expected.isoformat()
                and age_ok
            )
            return CacheRead(frame, metadata, fresh, "fresh" if fresh else "stale")
        except Exception as exc:
            return CacheRead(pd.DataFrame(), {}, False, f"corrupt:{type(exc).__name__}")

    def write(self, frame: pd.DataFrame, source: str, expected: date) -> None:
        metadata = {
            "source": source,
            "schema_version": UNIVERSE_SCHEMA_VERSION,
            "as_of": expected.isoformat(),
            "count": int(len(frame)),
            "saved_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        }
        _atomic_csv(frame, self.csv_path)
        _atomic_json(metadata, self.meta_path)


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    _atomic_csv(frame, path)


def atomic_write_json(data: dict[str, Any], path: Path) -> None:
    _atomic_json(data, path)
