"""Tests for prism_calendar.py — Gap Prompt 10b."""
from datetime import datetime, timedelta

from prism_calendar import CalendarEvent, PrismCalendar


def test_not_configured_when_empty():
    """PrismCalendar() with no args should not be configured."""
    assert PrismCalendar().configured is False


def test_parse_ics_basic():
    """_parse_ics with a valid ICS string returns a list with one event."""
    ics = (
        "BEGIN:VCALENDAR\r\n"
        "BEGIN:VEVENT\r\n"
        "UID:test-uid-001\r\n"
        "SUMMARY:Team Standup\r\n"
        "DTSTART:20240115T140000Z\r\n"
        "DTEND:20240115T143000Z\r\n"
        "END:VEVENT\r\n"
        "END:VCALENDAR\r\n"
    )
    events = PrismCalendar()._parse_ics(ics)
    assert len(events) == 1
    assert events[0].title == "Team Standup"
    assert events[0].event_id == "test-uid-001"


def test_parse_dt_formats():
    """_parse_dt should handle the UTC Z-suffixed format."""
    dt = PrismCalendar._parse_dt("20240115T140000Z")
    assert dt is not None
    assert isinstance(dt, datetime)
    assert dt.year == 2024
    assert dt.month == 1
    assert dt.day == 15
    assert dt.hour == 14


def test_find_free_slot_returns_datetime_or_none():
    """find_free_slot returns a datetime or None (never raises)."""
    cal = PrismCalendar()
    result = cal.find_free_slot()
    assert result is None or isinstance(result, datetime)


def test_today_filters_correctly():
    """today() includes events starting today and excludes events starting tomorrow."""
    now = datetime.now()

    today_event = CalendarEvent(
        event_id="ev-today",
        title="Today Meeting",
        start=now.replace(hour=10, minute=0, second=0, microsecond=0),
        end=now.replace(hour=11, minute=0, second=0, microsecond=0),
    )
    tomorrow_event = CalendarEvent(
        event_id="ev-tomorrow",
        title="Tomorrow Meeting",
        start=(now + timedelta(days=1)).replace(hour=10, minute=0, second=0, microsecond=0),
        end=(now + timedelta(days=1)).replace(hour=11, minute=0, second=0, microsecond=0),
    )

    cal = PrismCalendar()
    # Patch _fetch_events to return both events
    cal._fetch_events = lambda days_ahead=1: [today_event, tomorrow_event]

    result = cal.today()
    titles = [e.title for e in result]
    assert "Today Meeting" in titles
    assert "Tomorrow Meeting" not in titles


def test_status_unconfigured():
    """status_summary() on unconfigured instance returns configured=False."""
    summary = PrismCalendar().status_summary()
    assert summary.get("configured") is False
