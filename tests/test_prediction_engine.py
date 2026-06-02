"""Tests for prediction_engine.py"""
from prediction_engine import (
    InjuryRiskPredictor,
    MatchPredictor,
    PerformancePredictor,
    PredictionPlatform,
)


def test_import():
    """Module imports without error."""
    pass  # import above is the test


def test_match_predictor_instantiation():
    """MatchPredictor instantiates without error."""
    obj = MatchPredictor()
    assert obj is not None


def test_injury_risk_predictor_instantiation():
    """InjuryRiskPredictor instantiates without error."""
    obj = InjuryRiskPredictor()
    assert obj is not None


def test_performance_predictor_instantiation():
    """PerformancePredictor instantiates without error."""
    obj = PerformancePredictor()
    assert obj is not None


def test_prediction_platform_instantiation():
    """PredictionPlatform instantiates without error."""
    obj = PredictionPlatform()
    assert obj is not None
