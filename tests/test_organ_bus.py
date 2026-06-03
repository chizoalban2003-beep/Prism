"""Tests for prism_organ_bus — LLM-mediated inter-engine communication."""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from prism_organ_bus import (
    HIGH,
    LOW,
    NORMAL,
    OrganBus,
    OrganSignal,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def bus(tmp_path):
    return OrganBus(db_path=str(tmp_path / "test_bus.db"))


@pytest.fixture
def bus_with_router(tmp_path):
    router = MagicMock()
    router.call.return_value = (
        json.dumps({"adjustment": "reduce_load", "factor": 0.6, "duration_days": 3}),
        {},
    )
    return OrganBus(llm_router=router, db_path=str(tmp_path / "test_bus.db")), router


# ── OrganSignal ───────────────────────────────────────────────────────────────

class TestOrganSignal:
    def test_defaults(self):
        sig = OrganSignal(source="physics", signal_type="injury_risk", payload={"risk": 0.7})
        assert sig.priority == NORMAL
        assert len(sig.signal_id) == 8
        assert sig.timestamp > 0

    def test_invalid_priority_raises(self):
        with pytest.raises(ValueError, match="priority"):
            OrganSignal(source="x", signal_type="y", payload={}, priority=99)

    def test_priority_constants(self):
        assert LOW < NORMAL < HIGH


# ── Registration ──────────────────────────────────────────────────────────────

class TestRegistration:
    def test_register(self, bus):
        received = []
        bus.register("policy", ["injury_risk"], lambda p: received.append(p))
        assert bus.subscribers_for("injury_risk") == ["policy"]

    def test_register_multiple_signal_types(self, bus):
        bus.register("calendar", ["injury_risk", "performance_plateau"],
                     lambda p: None)
        assert "calendar" in bus.subscribers_for("injury_risk")
        assert "calendar" in bus.subscribers_for("performance_plateau")

    def test_register_replaces_existing(self, bus):
        called = []
        bus.register("policy", ["injury_risk"], lambda p: called.append("v1"))
        bus.register("policy", ["injury_risk"], lambda p: called.append("v2"))
        bus.emit(OrganSignal("physics", "injury_risk", {"r": 0.8}))
        assert called == ["v2"]

    def test_unregister(self, bus):
        bus.register("policy", ["injury_risk"], lambda p: None)
        bus.unregister("policy")
        assert bus.subscribers_for("injury_risk") == []

    def test_source_does_not_receive_own_signal(self, bus):
        received = []
        bus.register("physics", ["injury_risk"], lambda p: received.append(p))
        bus.emit(OrganSignal("physics", "injury_risk", {"risk": 0.8}))
        assert received == []


# ── Direct routing (no LLM) ───────────────────────────────────────────────────

class TestDirectRouting:
    def test_no_router_always_direct(self, bus):
        received = []
        bus.register(
            "policy", ["injury_risk"],
            lambda p: received.append(p),
            vocabulary="risk float, muscle_group str",
        )
        sig = OrganSignal("physics", "injury_risk",
                          {"risk": 0.8, "muscle_group": "hamstring"}, priority=NORMAL)
        records = bus.emit(sig)
        assert len(records) == 1
        assert not records[0].via_llm
        assert received[0]["risk"] == 0.8

    def test_high_overlap_direct_route(self, bus):
        """Payload keys mentioned in vocabulary → direct route."""
        received = []
        bus.register(
            "policy", ["load_update"],
            lambda p: received.append(p),
            vocabulary="load, intensity, duration_days",  # all payload keys present
        )
        sig = OrganSignal("physics", "load_update",
                          {"load": 0.7, "intensity": 0.8, "duration_days": 2},
                          priority=NORMAL)
        records = bus.emit(sig)
        assert not records[0].via_llm

    def test_high_priority_forces_llm_when_router_present(self, tmp_path):
        router = MagicMock()
        router.call.return_value = (
            json.dumps({"adjustment": "rest"}), {},
        )
        b = OrganBus(llm_router=router, db_path=str(tmp_path / "b.db"))
        b.register(
            "policy", ["injury_risk"],
            lambda p: None,
            vocabulary="risk, muscle_group",
        )
        sig = OrganSignal("physics", "injury_risk",
                          {"risk": 0.9, "muscle_group": "hamstring"}, priority=HIGH)
        records = b.emit(sig)
        assert records[0].via_llm
        assert router.call.call_count == 1


# ── LLM translation ───────────────────────────────────────────────────────────

class TestLLMTranslation:
    def test_llm_translation_called_when_low_overlap(self, tmp_path):
        router = MagicMock()
        router.call.return_value = (
            json.dumps({"message": "hamstring risk elevated", "action": "rest"}), {},
        )
        b = OrganBus(llm_router=router, db_path=str(tmp_path / "b.db"))
        b.register(
            "calendar", ["injury_risk"],
            lambda p: None,
            vocabulary="message str, scheduled_date str, action str",
        )
        sig = OrganSignal("physics", "injury_risk",
                          {"risk": 0.8, "muscle_group": "hamstring", "confidence": 0.9})
        records = b.emit(sig)
        assert records[0].via_llm
        assert router.call.call_count == 1

    def test_llm_result_delivered_to_handler(self, tmp_path):
        router = MagicMock()
        router.call.return_value = (
            json.dumps({"adjustment": "reduce_load", "factor": 0.6}), {},
        )
        received = []
        b = OrganBus(llm_router=router, db_path=str(tmp_path / "b.db"))
        b.register(
            "policy", ["injury_risk"],
            lambda p: received.append(p),
            vocabulary="adjustment str, factor float",
        )
        b.emit(OrganSignal("physics", "injury_risk",
                           {"risk": 0.8, "muscle_group": "hamstring"}))
        assert received[0]["adjustment"] == "reduce_load"
        assert received[0]["factor"] == 0.6

    def test_llm_cache_used_on_second_signal(self, tmp_path):
        router = MagicMock()
        router.call.return_value = (
            json.dumps({"adjustment": "reduce_load", "factor": 0.5}), {},
        )
        b = OrganBus(llm_router=router, db_path=str(tmp_path / "b.db"))
        b.register(
            "policy", ["injury_risk"],
            lambda p: None,
            vocabulary="adjustment str, factor float",
        )
        # Emit same signal type twice
        b.emit(OrganSignal("physics", "injury_risk",
                           {"risk": 0.7, "muscle_group": "quad"}))
        b.emit(OrganSignal("physics", "injury_risk",
                           {"risk": 0.8, "muscle_group": "hamstring"}))
        # LLM should only be called once (second uses cache)
        assert router.call.call_count == 1

    def test_high_priority_bypasses_cache(self, tmp_path):
        router = MagicMock()
        router.call.return_value = (
            json.dumps({"adjustment": "rest"}), {},
        )
        b = OrganBus(llm_router=router, db_path=str(tmp_path / "b.db"))
        b.register(
            "policy", ["injury_risk"],
            lambda p: None,
            vocabulary="adjustment str",
        )
        b.emit(OrganSignal("physics", "injury_risk", {"risk": 0.7}, priority=HIGH))
        b.emit(OrganSignal("physics", "injury_risk", {"risk": 0.9}, priority=HIGH))
        assert router.call.call_count == 2   # cache bypassed each time

    def test_llm_failure_falls_back_to_direct(self, tmp_path):
        router = MagicMock()
        router.call.side_effect = RuntimeError("LLM unavailable")
        received = []
        b = OrganBus(llm_router=router, db_path=str(tmp_path / "b.db"))
        b.register(
            "policy", ["injury_risk"],
            lambda p: received.append(p),
            vocabulary="adjustment str",
        )
        b.emit(OrganSignal("physics", "injury_risk", {"risk": 0.8, "muscle_group": "x"}))
        # Should fall back to direct pass
        assert received == [{"risk": 0.8, "muscle_group": "x"}]


# ── Low-priority batching ─────────────────────────────────────────────────────

class TestBatching:
    def test_low_priority_queues_not_delivers(self, bus):
        received = []
        bus.register("policy", ["telemetry"], lambda p: received.append(p))
        bus.emit(OrganSignal("physics", "telemetry", {"fps": 60}, priority=LOW))
        assert received == []   # not delivered yet

    def test_flush_delivers_batched(self, bus):
        received = []
        bus.register("policy", ["telemetry"], lambda p: received.append(p))
        bus.emit(OrganSignal("physics", "telemetry", {"fps": 60}, priority=LOW))
        bus.emit(OrganSignal("physics", "telemetry", {"fps": 59}, priority=LOW))
        records = bus.flush_batch()
        assert len(records) == 2
        assert len(received) == 2

    def test_flush_clears_queue(self, bus):
        bus.register("policy", ["telemetry"], lambda p: None)
        bus.emit(OrganSignal("physics", "telemetry", {"fps": 60}, priority=LOW))
        bus.flush_batch()
        records2 = bus.flush_batch()
        assert records2 == []


# ── Fan-out (multiple subscribers) ───────────────────────────────────────────

class TestFanout:
    def test_signal_delivered_to_all_subscribers(self, bus):
        policy_received = []
        calendar_received = []
        horizon_received = []
        bus.register("policy",   ["injury_risk"], lambda p: policy_received.append(p))
        bus.register("calendar", ["injury_risk"], lambda p: calendar_received.append(p))
        bus.register("horizon",  ["injury_risk"], lambda p: horizon_received.append(p))
        bus.emit(OrganSignal("physics", "injury_risk", {"risk": 0.8}))
        assert len(policy_received) == 1
        assert len(calendar_received) == 1
        assert len(horizon_received) == 1

    def test_unrelated_subscriber_not_notified(self, bus):
        email_received = []
        bus.register("policy",   ["injury_risk"], lambda p: None)
        bus.register("email",    ["calendar_event"], lambda p: email_received.append(p))
        bus.emit(OrganSignal("physics", "injury_risk", {"risk": 0.8}))
        assert email_received == []


# ── DeliveryRecord ────────────────────────────────────────────────────────────

class TestDeliveryRecord:
    def test_successful_delivery(self, bus):
        bus.register("policy", ["sig"], lambda p: None)
        records = bus.emit(OrganSignal("src", "sig", {"k": "v"}))
        assert records[0].success is True
        assert records[0].error == ""

    def test_failed_handler_recorded(self, bus):
        def bad_handler(p):
            raise RuntimeError("handler crash")
        bus.register("policy", ["sig"], bad_handler)
        records = bus.emit(OrganSignal("src", "sig", {"k": "v"}))
        assert records[0].success is False
        assert "handler crash" in records[0].error

    def test_no_subscribers_returns_empty(self, bus):
        records = bus.emit(OrganSignal("src", "unknown_signal", {"k": "v"}))
        assert records == []


# ── Persistence ───────────────────────────────────────────────────────────────

class TestPersistence:
    def test_signal_persisted(self, bus):
        bus.register("policy", ["injury_risk"], lambda p: None)
        bus.emit(OrganSignal("physics", "injury_risk", {"risk": 0.8}))
        history = bus.history(n=5)
        assert len(history) == 1
        assert history[0]["signal_type"] == "injury_risk"

    def test_no_subscriber_persisted_as_no_subscribers(self, bus):
        bus.emit(OrganSignal("physics", "orphan_signal", {"x": 1}))
        history = bus.history(n=5)
        assert history[0]["status"] == "no_subscribers"

    def test_history_limit(self, bus):
        bus.register("policy", ["sig"], lambda p: None)
        for i in range(10):
            bus.emit(OrganSignal("src", "sig", {"i": i}))
        history = bus.history(n=3)
        assert len(history) == 3
