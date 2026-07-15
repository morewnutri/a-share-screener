from __future__ import annotations

from datetime import date, datetime, timedelta, time as dtime


def _is_weekend(d: date) -> bool:
    return d.weekday() >= 5


def previous_weekday(d: date) -> date:
    while _is_weekend(d):
        d -= timedelta(days=1)
    return d


def latest_complete_trade_date(now_dt: datetime, market_close_time: str = "15:10") -> date:
    hh, mm = market_close_time.split(":")
    close_t = dtime(hour=int(hh), minute=int(mm))
    d = now_dt.date()

    if _is_weekend(d):
        return previous_weekday(d - timedelta(days=1))

    if now_dt.time() >= close_t:
        return d

    return previous_weekday(d - timedelta(days=1))


def is_complete_daily_bar(last_bar_date: date, now_dt: datetime, market_close_time: str = "15:10") -> bool:
    return last_bar_date == latest_complete_trade_date(now_dt, market_close_time)
