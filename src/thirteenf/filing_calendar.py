from __future__ import annotations

import calendar
from datetime import date, datetime, timezone

import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

_BUSINESS_DAY = CustomBusinessDay(calendar=USFederalHolidayCalendar())
DEADLINE_DAYS = 45
WINDOW_DAYS_BEFORE = 3
WINDOW_DAYS_AFTER = 1
DAILY_CHECK_HOUR_UTC = 6


def filing_deadline(quarter_end: date) -> date:
    """13F deadline: 45 days after quarter end, rolled to the next business day."""
    due = pd.Timestamp(quarter_end) + pd.Timedelta(days=DEADLINE_DAYS)
    return _BUSINESS_DAY.rollforward(due).date()


def _quarter_ends(year: int) -> list[date]:
    return [date(year, month, calendar.monthrange(year, month)[1]) for month in (3, 6, 9, 12)]


def in_polling_window(today: date) -> bool:
    """True from three days before a filing deadline through the day after it."""
    for year in (today.year - 1, today.year):
        for quarter_end in _quarter_ends(year):
            deadline = filing_deadline(quarter_end)
            start = deadline - pd.Timedelta(days=WINDOW_DAYS_BEFORE)
            end = deadline + pd.Timedelta(days=WINDOW_DAYS_AFTER)
            if pd.Timestamp(start) <= pd.Timestamp(today) <= pd.Timestamp(end):
                return True
    return False


def should_poll(now: datetime | None = None) -> bool:
    """Poll every run inside a deadline window, and once a day otherwise for late filers."""
    current = now or datetime.now(timezone.utc)
    if in_polling_window(current.date()):
        return True
    return current.hour == DAILY_CHECK_HOUR_UTC and current.minute < 5
