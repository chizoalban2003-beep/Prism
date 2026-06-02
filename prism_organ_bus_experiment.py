"""
prism_organ_bus_experiment.py
==============================
Experiment 4: Inter-Engine LLM Communication via OrganBus

Demonstrates the biological analogy in action:

    Physics Engine  (muscle)  → emits injury risk signal
           ↓   bloodstream (LLM bus)
    Policy Engine   (brain)   ← reduces training load
    Calendar Engine (planner) ← reschedules session
    Horizon Planner (memory)  ← creates recovery watch goal

No engine knows another engine's schema.  The LLM translates each signal
into the receiver's vocabulary — just like hormones carry meaning across
incompatible cell types.

Run with:
    python3 prism_organ_bus_experiment.py
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock

from prism_organ_bus import HIGH, LOW, OrganBus, OrganSignal


# ── Mock LLM router ───────────────────────────────────────────────────────────


def _make_smart_router() -> Any:
    """
    Router that returns vocabulary-appropriate translations.
    In production this is a real Ollama/OpenAI call.
    """
    router = MagicMock()

    def smart_call(prompt: str, **kwargs):
        p = prompt.lower()
        # Policy engine translation
        if "policy_engine" in p:
            return (
                json.dumps({
                    "adjustment":    "reduce_training_load",
                    "factor":        0.55,
                    "duration_days": 4,
                    "reason":        "Hamstring injury risk elevated at 78%",
                    "flag_for_physio": True,
                }), {}
            )
        # Calendar engine translation
        if "calendar_engine" in p:
            return (
                json.dumps({
                    "message":        "Hamstring risk 78% — consider rest or light session",
                    "action":         "reschedule_heavy_session",
                    "days_to_defer":  2,
                    "notify_coach":   True,
                }), {}
            )
        # Horizon planner translation
        if "horizon_planner" in p:
            return (
                json.dumps({
                    "intent":               "Monitor hamstring recovery until risk drops below 30%",
                    "trigger_condition":    "injury_risk < 0.30",
                    "completion_condition": "hamstring cleared by physio",
                }), {}
            )
        return (json.dumps({"raw": prompt[:80]}), {})

    router.call.side_effect = smart_call
    return router


# ── Mock engine "receptors" (handlers) ───────────────────────────────────────


@dataclass
class PolicyEngine:
    events: list = field(default_factory=list)

    def on_signal(self, payload: dict):
        self.events.append(payload)
        adj = payload.get("adjustment", "no change")
        fac = payload.get("factor", 1.0)
        print(f"  [PolicyEngine] Applied: {adj} (factor={fac:.0%})")
        if payload.get("flag_for_physio"):
            print("  [PolicyEngine] Flagged: send athlete to physio")


@dataclass
class CalendarEngine:
    events: list = field(default_factory=list)

    def on_signal(self, payload: dict):
        self.events.append(payload)
        msg    = payload.get("message", "")
        action = payload.get("action", "")
        defer  = payload.get("days_to_defer", 0)
        print(f"  [CalendarEngine] {msg}")
        if action:
            print(f"  [CalendarEngine] Action: {action} (defer {defer}d)")


@dataclass
class HorizonAdapter:
    """Wraps HorizonPlanner signals into goal creation calls."""
    events: list = field(default_factory=list)

    def on_signal(self, payload: dict):
        self.events.append(payload)
        intent = payload.get("intent", "")
        cond   = payload.get("trigger_condition", "")
        print(f"  [HorizonPlanner] Watch goal created: {intent!r}")
        print(f"  [HorizonPlanner] Trigger: {cond}")


# ── Experiment runner ─────────────────────────────────────────────────────────


def run_experiment(db_path: str = "/tmp/organ_bus_exp.db"):
    width = 72
    print()
    print("=" * width)
    print("PRISM ORGAN BUS — Experiment 4: Inter-Engine LLM Communication")
    print("Biological model: Logic Engines as Organs, LLM as Bloodstream")
    print("=" * width)

    router   = _make_smart_router()
    bus      = OrganBus(llm_router=router, db_path=db_path)
    policy   = PolicyEngine()
    calendar = CalendarEngine()
    horizon  = HorizonAdapter()

    # ── Register organs (with their vocabularies) ─────────────────────────────
    print("\n  Registering organs with their vocabularies...")

    bus.register(
        organ_name   = "policy_engine",
        signal_types = ["injury_risk_elevated", "performance_plateau"],
        handler      = policy.on_signal,
        vocabulary   = (
            "Understands: adjustment (str: reduce_training_load|rest|continue), "
            "factor (float 0-1), duration_days (int), reason (str), "
            "flag_for_physio (bool)"
        ),
    )
    bus.register(
        organ_name   = "calendar_engine",
        signal_types = ["injury_risk_elevated", "recovery_complete"],
        handler      = calendar.on_signal,
        vocabulary   = (
            "Understands: message (str notification text), "
            "action (str: reschedule_heavy_session|cancel|add_rest_day), "
            "days_to_defer (int), notify_coach (bool)"
        ),
    )
    bus.register(
        organ_name   = "horizon_planner",
        signal_types = ["injury_risk_elevated"],
        handler      = horizon.on_signal,
        vocabulary   = (
            "Understands: intent (str: goal description), "
            "trigger_condition (str: condition to watch for), "
            "completion_condition (str: what done looks like)"
        ),
    )

    # ── Scenario 1: Physics engine detects elevated injury risk ───────────────
    print("\n" + "─" * width)
    print("  Scenario 1: Physics engine emits injury_risk_elevated (HIGH priority)")
    print()

    t0 = time.time()
    signal = OrganSignal(
        source      = "physics_engine",
        signal_type = "injury_risk_elevated",
        payload     = {
            "risk":             0.78,
            "muscle_group":     "hamstring",
            "model_confidence": 0.92,
            "athlete_id":       "athlete_001",
            "session_type":     "sprint_training",
        },
        priority    = HIGH,   # always translate — important signal
    )
    records = bus.emit(signal)
    elapsed = (time.time() - t0) * 1000

    print(f"\n  Delivered to {len(records)} organs in {elapsed:.0f}ms")
    print(f"  LLM translations: {sum(1 for r in records if r.via_llm)}")
    print(f"  Direct routes:    {sum(1 for r in records if not r.via_llm)}")
    print(f"  LLM router calls: {router.call.call_count}")

    # ── Scenario 2: Low-priority telemetry (batched) ──────────────────────────
    print("\n" + "─" * width)
    print("  Scenario 2: Physics engine emits LOW-priority telemetry (batched)")
    print()
    bus.register(
        organ_name   = "policy_engine",
        signal_types = ["injury_risk_elevated", "performance_plateau", "telemetry"],
        handler      = policy.on_signal,
        vocabulary   = (
            "Understands: adjustment str, factor float, duration_days int, "
            "reason str, flag_for_physio bool, fps int, latency_ms float"
        ),
    )
    for i in range(3):
        bus.emit(OrganSignal(
            source      = "physics_engine",
            signal_type = "telemetry",
            payload     = {"fps": 60 - i, "latency_ms": 12.0 + i},
            priority    = LOW,
        ))
    print("  3 telemetry signals queued (not yet delivered)...")
    batch_records = bus.flush_batch()
    print(f"  Flushed: {len(batch_records)} deliveries")

    # ── Scenario 3: No subscriber ─────────────────────────────────────────────
    print("\n" + "─" * width)
    print("  Scenario 3: Signal with no registered subscriber (silent discard)")
    print()
    orphan_records = bus.emit(OrganSignal(
        source      = "vision_engine",
        signal_type = "video_highlight_detected",
        payload     = {"clip_id": "clip_042", "confidence": 0.85},
    ))
    print(f"  Delivered to {len(orphan_records)} organs (expected 0)")

    # ── Results ───────────────────────────────────────────────────────────────
    print("\n" + "=" * width)
    print("  RESULTS")
    print("─" * width)
    print(f"  Total LLM calls:         {router.call.call_count}")
    print(f"  Policy events received:  {len(policy.events)}")
    print(f"  Calendar events received:{len(calendar.events)}")
    print(f"  Horizon events received: {len(horizon.events)}")
    history = bus.history(n=10)
    print(f"  Signals persisted in DB: {len(history)}")
    print()

    print("  KEY INSIGHT")
    print("  ──────────")
    print("  The physics engine emitted one raw dict with 5 keys.")
    print("  The LLM translated it into 3 different vocabularies:")
    print("    policy   → reduction factor, duration, physio flag")
    print("    calendar → human message, action, defer days")
    print("    horizon  → goal intent, trigger condition, completion condition")
    print()
    print("  No engine knew any other engine's schema.")
    print("  The LLM (bloodstream) carried meaning across all three.")
    print("=" * width)
    print()

    return {
        "llm_calls":       router.call.call_count,
        "policy_events":   len(policy.events),
        "calendar_events": len(calendar.events),
        "horizon_events":  len(horizon.events),
        "persisted":       len(history),
    }


if __name__ == "__main__":
    run_experiment()
