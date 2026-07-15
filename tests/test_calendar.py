from datetime import datetime
from zoneinfo import ZoneInfo

from ashare_scanner.calendar import expected_complete_session


TZ = ZoneInfo("Asia/Shanghai")


def test_before_close_uses_previous_session():
    now = datetime(2026, 7, 15, 8, 0, tzinfo=TZ)
    assert expected_complete_session(now).isoformat() == "2026-07-14"


def test_after_close_uses_current_session():
    now = datetime(2026, 7, 15, 15, 20, tzinfo=TZ)
    assert expected_complete_session(now).isoformat() == "2026-07-15"


def test_weekend_uses_previous_session():
    now = datetime(2026, 7, 18, 12, 0, tzinfo=TZ)
    assert expected_complete_session(now).isoformat() == "2026-07-17"

