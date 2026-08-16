from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from .cache import atomic_write_csv
from .calendar import session_offset, sessions_between


ACTIVE_STATES = {"SETUP", "TRIGGER", "RETEST"}
SIGNAL_TO_STATE = {
    "chip_base_ready": "SETUP",
    "chip_base_launch": "TRIGGER",
    "chip_base_rebound": "RETEST",
}
STATE_PRIORITY = {"SETUP": 1, "TRIGGER": 2, "RETEST": 3}
WATCHLIST_COLUMNS = [
    "code",
    "name",
    "state",
    "signals",
    "first_seen_date",
    "last_signal_date",
    "last_evaluated_date",
    "data_status",
    "age_sessions",
    "score_total",
    "close",
    "resistance_price",
    "support_price",
    "expires_date",
    "trigger_date",
    "trigger_price",
]
TRANSITION_COLUMNS = [
    "date",
    "code",
    "name",
    "from_state",
    "to_state",
    "signals",
    "close",
]


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, dtype={"code": str})
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def update_watchlist(
    data_dir: Path,
    signals: dict[str, pd.DataFrame],
    indicators: pd.DataFrame,
    expected: date,
    ttl_sessions: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    state_dir = data_dir / "state"
    state_path = state_dir / "watchlist.csv"
    transitions_path = state_dir / "transitions.csv"
    previous = _read_csv(state_path)
    previous_by_code = {
        str(row["code"]).zfill(6): row.to_dict() for _, row in previous.iterrows()
    }
    indicator_by_code = {
        str(row["code"]).zfill(6): row.to_dict() for _, row in indicators.iterrows()
    }

    current_signals: dict[str, dict] = {}
    for signal_name, frame in signals.items():
        for _, row in frame.iterrows():
            code = str(row["code"]).zfill(6)
            item = current_signals.setdefault(code, {"row": row.to_dict(), "signals": []})
            item["signals"].append(signal_name)

    records: list[dict] = []
    transitions: list[dict] = []
    all_codes = sorted(set(previous_by_code) | set(current_signals))
    today = expected.isoformat()

    for code in all_codes:
        old = previous_by_code.get(code, {})
        signal_item = current_signals.get(code)
        market_row = indicator_by_code.get(code, {})
        old_state = str(old.get("state", ""))
        record = dict(old)
        record["code"] = code

        if signal_item:
            row = signal_item["row"]
            signal_names = sorted(signal_item["signals"])
            desired_state = max(
                (SIGNAL_TO_STATE[name] for name in signal_names),
                key=STATE_PRIORITY.get,
            )
            if old_state not in ACTIVE_STATES:
                record["first_seen_date"] = today
                record["age_sessions"] = 0
            record.update(
                {
                    "name": row.get("name", old.get("name", "")),
                    "state": desired_state,
                    "signals": ",".join(signal_names),
                    "last_signal_date": today,
                    "last_evaluated_date": today,
                    "data_status": "ok",
                    "score_total": row.get("score_total", np.nan),
                    "close": row.get("close", np.nan),
                    "resistance_price": row.get(
                        "rebound_base_high",
                        row.get("base_high_20_pre3", np.nan),
                    ),
                    "support_price": row.get("chip_peak_price", np.nan),
                    "expires_date": session_offset(expected, ttl_sessions).isoformat(),
                }
            )
            if desired_state == "TRIGGER":
                record["trigger_date"] = today
                record["trigger_price"] = row.get("close", np.nan)
        elif old:
            record["signals"] = ""
            if market_row:
                record["data_status"] = "ok"
                record["close"] = market_row.get("close", record.get("close", np.nan))
                support = float(record.get("support_price", np.nan))
                close = float(market_row.get("close", np.nan))
                if np.isfinite(support) and np.isfinite(close) and close < support * 0.98:
                    record["state"] = "INVALID"
                elif str(record.get("expires_date", "")) < today:
                    record["state"] = "EXPIRED"
            else:
                record["data_status"] = "missing_today"

            last_evaluated = str(record.get("last_evaluated_date", ""))
            if last_evaluated and last_evaluated != today:
                try:
                    elapsed = sessions_between(date.fromisoformat(last_evaluated), expected)
                    record["age_sessions"] = int(record.get("age_sessions", 0)) + elapsed
                except ValueError:
                    pass
            record["last_evaluated_date"] = today

        new_state = str(record.get("state", ""))
        if new_state and new_state != old_state:
            transitions.append(
                {
                    "date": today,
                    "code": code,
                    "name": record.get("name", ""),
                    "from_state": old_state or "NEW",
                    "to_state": new_state,
                    "signals": record.get("signals", ""),
                    "close": record.get("close", np.nan),
                }
            )
        records.append(record)

    watchlist = pd.DataFrame(records)
    if watchlist.empty:
        watchlist = pd.DataFrame(columns=WATCHLIST_COLUMNS)
    else:
        watchlist = watchlist.sort_values(["state", "score_total"], ascending=[True, False])
    active = watchlist[watchlist["state"].isin(ACTIVE_STATES)].copy()
    new_transitions = pd.DataFrame(transitions)
    if new_transitions.empty:
        new_transitions = pd.DataFrame(columns=TRANSITION_COLUMNS)
    transition_history = _read_csv(transitions_path)
    if transition_history.empty:
        transition_history = pd.DataFrame(columns=TRANSITION_COLUMNS)
    if not new_transitions.empty:
        if transition_history.empty:
            transition_history = new_transitions.copy()
        else:
            transition_history = pd.concat(
                [transition_history, new_transitions],
                ignore_index=True,
            )
        transition_history = transition_history.drop_duplicates(
            ["date", "code", "from_state", "to_state"], keep="last"
        )

    atomic_write_csv(watchlist, state_path)
    atomic_write_csv(transition_history, transitions_path)
    return watchlist.reset_index(drop=True), active.reset_index(drop=True), new_transitions
