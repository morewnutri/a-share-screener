from __future__ import annotations

from datetime import date, datetime, time, timedelta
from functools import lru_cache
from zoneinfo import ZoneInfo

import exchange_calendars as xcals
import pandas as pd


SHANGHAI_TZ = ZoneInfo("Asia/Shanghai")


@lru_cache(maxsize=1)
def get_calendar():
    return xcals.get_calendar("XSHG")


def _session_date(value: pd.Timestamp) -> date:
    return value.tz_localize(None).date() if value.tzinfo else value.date()


def expected_complete_session(
    now: datetime | None = None,
    close_buffer_minutes: int = 10,
) -> date:
    """Return the latest session whose daily bar should be complete now."""
    now = now or datetime.now(SHANGHAI_TZ)
    if now.tzinfo is None:
        now = now.replace(tzinfo=SHANGHAI_TZ)
    else:
        now = now.astimezone(SHANGHAI_TZ)

    calendar = get_calendar()
    today = pd.Timestamp(now.date())
    cutoff = datetime.combine(
        now.date(),
        time(15, 0),
        tzinfo=SHANGHAI_TZ,
    ) + timedelta(minutes=close_buffer_minutes)

    if calendar.is_session(today) and now >= cutoff:
        return now.date()

    if calendar.is_session(today):
        return _session_date(calendar.previous_session(today))

    return _session_date(calendar.date_to_session(today, direction="previous"))


def session_offset(session: date, count: int) -> date:
    calendar = get_calendar()
    current = calendar.date_to_session(pd.Timestamp(session), direction="none")
    step = calendar.next_session if count >= 0 else calendar.previous_session
    for _ in range(abs(count)):
        current = step(current)
    return _session_date(current)


def sessions_between(start: date, end: date) -> int:
    if end <= start:
        return 0
    sessions = get_calendar().sessions_in_range(pd.Timestamp(start), pd.Timestamp(end))
    return max(0, len(sessions) - 1)

