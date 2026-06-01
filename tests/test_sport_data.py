"""Tests for sport_data.py"""
from sport_data import StatsBombConnector


def test_import():
    """Module imports without error."""
    pass  # import above is the test


def test_instantiation():
    """StatsBombConnector instantiates without error."""
    obj = StatsBombConnector()
    assert obj is not None
