"""Tests for SignalAnomalyDetector and OrganBus anomaly wiring."""
from __future__ import annotations

from prism_organ_bus import NORMAL, OrganBus, OrganSignal, SignalAnomalyDetector


def test_no_anomaly_below_threshold():
    det = SignalAnomalyDetector(baseline_per_window=2.0, spike_multiplier=3.0)
    fired = []
    det.on_anomaly(lambda t, c, w: fired.append(t))
    for _ in range(5):  # 5 < 2*3=6
        det.record("health_alert")
    assert not fired


def test_anomaly_fires_at_threshold():
    det = SignalAnomalyDetector(baseline_per_window=1.0, spike_multiplier=3.0)
    fired = []
    det.on_anomaly(lambda t, c, w: fired.append((t, c)))
    for _ in range(3):
        det.record("finance_alert")
    assert any(t == "finance_alert" for t, _ in fired)


def test_counts_reflects_current_window():
    det = SignalAnomalyDetector(window_seconds=60.0)
    det.record("test_signal")
    det.record("test_signal")
    assert det.counts().get("test_signal", 0) == 2


def test_multiple_types_tracked_independently():
    det = SignalAnomalyDetector()
    det.record("type_a")
    det.record("type_b")
    det.record("type_b")
    counts = det.counts()
    assert counts.get("type_a", 0) == 1
    assert counts.get("type_b", 0) == 2


def test_callback_receives_correct_args():
    det = SignalAnomalyDetector(baseline_per_window=1.0, spike_multiplier=2.0)
    calls = []
    det.on_anomaly(lambda t, c, w: calls.append((t, c, w)))
    det.record("goal_triggered")
    det.record("goal_triggered")
    assert calls
    assert calls[0][0] == "goal_triggered"
    assert calls[0][1] >= 2


def test_organ_bus_records_signals_in_detector():
    bus = OrganBus(llm_router=None)
    bus.emit(OrganSignal(source="physics", signal_type="health_alert", payload={}, priority=NORMAL))
    bus.emit(OrganSignal(source="physics", signal_type="health_alert", payload={}, priority=NORMAL))
    assert bus.anomaly_detector.counts().get("health_alert", 0) == 2


def test_anomaly_callback_wired_to_bus():
    bus = OrganBus(llm_router=None)
    fired = []
    bus.anomaly_detector.on_anomaly(lambda t, c, w: fired.append(t))
    bus.anomaly_detector._baseline   = 1.0
    bus.anomaly_detector._multiplier = 2.0
    for _ in range(2):
        bus.emit(OrganSignal(source="src", signal_type="organ_error", payload={}, priority=NORMAL))
    assert "organ_error" in fired
