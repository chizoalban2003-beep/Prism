from __future__ import annotations

import time

from digital_identity import CrystallisationEngine, DigitalIdentity, DomainProfile
from identity_bus import IdentityBus


def _engine(tmp_path):
    bus = IdentityBus(db_path=str(tmp_path / "identity_bus.db"))
    return CrystallisationEngine(
        "TestUser",
        bus,
        db_path=str(tmp_path / "identity.db"),
    )


def test_observe_updates_fulcrum(tmp_path):
    engine = _engine(tmp_path)
    engine.observe("sport", 0.2, 0.7)
    first = engine.get_identity().domains["sport"].fixed_fulcrum
    engine.observe("sport", 0.8, 0.7)
    second = engine.get_identity().domains["sport"].fixed_fulcrum
    assert second != first


def test_crystallise_after_threshold(tmp_path):
    engine = _engine(tmp_path)
    for _ in range(25):
        engine.observe("sport", 0.55, 0.8)
    identity = engine.get_identity()
    assert identity is not None
    assert identity.domains["sport"].crystallised is True


def test_emergent_insight_compartmentaliser():
    identity = DigitalIdentity(
        user_name="TestUser",
        domains={
            "sport": DomainProfile("sport", 0.8, 0.01, 25, True, time.time()),
            "medical": DomainProfile("medical", 0.3, 0.01, 25, True, time.time()),
        },
        cross_signals={"time_pressure_response": 0.5},
        overall_risk=0.55,
        consistency=0.9,
        n_total=50,
        created_at=time.time(),
        last_updated=time.time(),
    )
    assert identity.emergent_insight() == "Compartmentaliser"


def test_to_card_data_structure():
    identity = DigitalIdentity(
        user_name="TestUser",
        domains={"sport": DomainProfile("sport", 0.6, 0.01, 25, True, time.time())},
        cross_signals={"time_pressure_response": 0.5},
        overall_risk=0.6,
        consistency=0.9,
        n_total=25,
        created_at=time.time(),
        last_updated=time.time(),
    )
    card = identity.to_card_data()
    assert "domains" in card
    assert "insight" in card
    assert "confidence" in card


def test_reset_domain_clears(tmp_path):
    engine = _engine(tmp_path)
    engine.observe("sport", 0.6, 0.8)
    engine.reset_domain("sport")
    identity = engine.get_identity()
    assert identity is not None
    assert identity.domains["sport"].n_observations == 0
