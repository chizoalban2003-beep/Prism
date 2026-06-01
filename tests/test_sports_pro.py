"""Tests for sports_pro.py"""
from sports_pro import Role, SportsProProfile, DailyPlanner, SportsProAssistant


def test_import():
    """Module imports without error."""
    pass  # import above is the test


def test_role_enum():
    """Role enum has expected values."""
    assert Role.ATHLETE is not None
    assert Role.COACH is not None


def test_sports_pro_profile_instantiation():
    """SportsProProfile instantiates without error."""
    profile = SportsProProfile(name="Test", role=Role.ATHLETE, sport="Football", team="City FC")
    assert profile is not None
    assert profile.name == "Test"


def test_daily_planner_instantiation():
    """DailyPlanner instantiates without error."""
    obj = DailyPlanner()
    assert obj is not None


def test_sports_pro_assistant_instantiation():
    """SportsProAssistant instantiates without error."""
    obj = SportsProAssistant()
    assert obj is not None
