from __future__ import annotations

import time

from identity_bus import IdentityBus, IdentitySignal


def test_publish_stores_signal(tmp_path):
    bus = IdentityBus(db_path=str(tmp_path / "bus.db"))
    signal = IdentitySignal("sport", "risk_override_tendency", 0.7, 0.8, time.time())
    bus.publish(signal)

    assert bus.aggregate("risk_override_tendency") != 0.5


def test_aggregate_returns_float(tmp_path):
    bus = IdentityBus(db_path=str(tmp_path / "bus.db"))
    bus.publish(IdentitySignal("sport", "aggression_index", 0.8, 0.9, time.time()))
    value = bus.aggregate("aggression_index")
    assert isinstance(value, float)
    assert 0.0 <= value <= 1.0


def test_subscriber_called(tmp_path):
    bus = IdentityBus(db_path=str(tmp_path / "bus.db"))
    received = []
    bus.subscribe("consistency_score", received.append)
    bus.publish(IdentitySignal("sport", "consistency_score", 0.6, 0.7, time.time()))
    assert len(received) == 1
    assert received[0].signal_id == "consistency_score"


def test_cross_domain_profile_all_keys(tmp_path):
    bus = IdentityBus(db_path=str(tmp_path / "bus.db"))
    profile = bus.cross_domain_profile()
    assert set(profile) == {
        "risk_override_tendency",
        "time_pressure_response",
        "data_reliance",
        "consistency_score",
        "aggression_index",
    }
