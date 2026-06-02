"""Tests for prism_perception.py — PRISM Perceptual Context Engine."""
from __future__ import annotations

import queue
import time

import pytest

from prism_perception import (
    BiometricChannel,
    ContextFuser,
    ContextSignal,
    ContextState,
    PerceptionChannel,
    PrismPerception,
    ScreenContextChannel,
    SystemContextChannel,
    TypingPatternChannel,
    VoiceChannel,
)

# ---------------------------------------------------------------------------
# ContextSignal
# ---------------------------------------------------------------------------

def test_context_signal_defaults():
    sig = ContextSignal(channel="system", factor_id="cpu", value=0.5, confidence=0.8)
    assert sig.channel == "system"
    assert sig.factor_id == "cpu"
    assert sig.value == 0.5
    assert sig.confidence == 0.8
    assert sig.raw_label == ""
    assert sig.timestamp > 0


def test_context_signal_raw_label():
    sig = ContextSignal(channel="biometric", factor_id="hrv_recovery",
                        value=0.7, confidence=0.9, raw_label="HRV 65ms")
    assert sig.raw_label == "HRV 65ms"


# ---------------------------------------------------------------------------
# ContextState
# ---------------------------------------------------------------------------

def test_context_state_to_factor_updates_filters_low_confidence():
    state = ContextState(
        factors    = {"a": 0.9, "b": 0.5, "c": 0.1},
        confidence = {"a": 0.8, "b": 0.39, "c": 0.1},
        active_channels = [],
    )
    updates = state.to_factor_updates()
    assert "a" in updates
    assert "b" not in updates   # confidence 0.39 < 0.4
    assert "c" not in updates


def test_context_state_to_factor_updates_exact_threshold():
    state = ContextState(
        factors    = {"x": 0.5},
        confidence = {"x": 0.4},
        active_channels = [],
    )
    assert "x" in state.to_factor_updates()


def test_context_state_summary_default():
    state = ContextState(factors={}, confidence={}, active_channels=[])
    assert isinstance(state.summary, str)


def test_context_state_last_updated_is_float():
    state = ContextState(factors={}, confidence={}, active_channels=[])
    assert isinstance(state.last_updated, float)


# ---------------------------------------------------------------------------
# PerceptionChannel (base)
# ---------------------------------------------------------------------------

class _ConcreteChannel(PerceptionChannel):
    NAME = "test"

    def _run(self):
        self._emit("factor_a", 0.5, 0.8, "test")
        self._stop.wait()


def test_perception_channel_emit_clamps_above_one():
    q = queue.Queue()
    ch = _ConcreteChannel(q)
    ch._emit("f", 1.5, 0.9)
    sig = q.get_nowait()
    assert sig.value == 1.0


def test_perception_channel_emit_clamps_below_zero():
    q = queue.Queue()
    ch = _ConcreteChannel(q)
    ch._emit("f", -0.5, 0.9)
    sig = q.get_nowait()
    assert sig.value == 0.0


def test_perception_channel_emit_sets_channel_name():
    q = queue.Queue()
    ch = _ConcreteChannel(q)
    ch._emit("f", 0.3, 0.7, "label")
    sig = q.get_nowait()
    assert sig.channel == "test"


def test_perception_channel_disabled_does_not_start():
    q = queue.Queue()
    ch = _ConcreteChannel(q, enabled=False)
    ch.start()
    assert ch._thread is None


def test_perception_channel_pause_resume():
    q = queue.Queue()
    ch = _ConcreteChannel(q)
    ch.pause()
    assert not ch._enabled
    ch.resume()
    assert ch._enabled


def test_perception_channel_stop_sets_event():
    q = queue.Queue()
    ch = _ConcreteChannel(q)
    ch.stop()
    assert ch._stop.is_set()


def test_base_channel_run_raises():
    q = queue.Queue()
    ch = PerceptionChannel(q)
    with pytest.raises(NotImplementedError):
        ch._run()


# ---------------------------------------------------------------------------
# SystemContextChannel
# ---------------------------------------------------------------------------

def test_system_channel_name():
    assert SystemContextChannel.NAME == "system"


def test_system_channel_emit_time_context():
    q = queue.Queue()
    ch = SystemContextChannel(q)
    ch._emit_time_context()
    signals = []
    while not q.empty():
        signals.append(q.get_nowait())
    factor_ids = {s.factor_id for s in signals}
    assert "circadian_energy" in factor_ids
    assert "work_context" in factor_ids


def test_system_channel_circadian_energy_in_range():
    q = queue.Queue()
    ch = SystemContextChannel(q)
    ch._emit_time_context()
    signals = []
    while not q.empty():
        signals.append(q.get_nowait())
    circadian = [s for s in signals if s.factor_id == "circadian_energy"]
    assert len(circadian) == 1
    assert 0.0 <= circadian[0].value <= 1.0


def test_system_channel_emit_battery_no_crash():
    q = queue.Queue()
    ch = SystemContextChannel(q)
    ch._emit_battery()   # may or may not emit signals depending on OS


def test_system_channel_emit_system_load_no_crash():
    q = queue.Queue()
    ch = SystemContextChannel(q)
    ch._emit_system_load()   # psutil may or may not be available


# ---------------------------------------------------------------------------
# TypingPatternChannel
# ---------------------------------------------------------------------------

def test_typing_channel_name():
    assert TypingPatternChannel.NAME == "typing"


def test_typing_channel_record_keypress_stores_timestamps():
    q = queue.Queue()
    ch = TypingPatternChannel(q)
    for _ in range(5):
        ch.record_keypress()
        time.sleep(0.01)
    assert len(ch._key_times) == 5


def test_typing_channel_max_samples_enforced():
    q = queue.Queue()
    ch = TypingPatternChannel(q)
    for _ in range(40):
        ch.record_keypress()
    assert len(ch._key_times) <= ch._max_samples


def test_typing_channel_analyse_emits_signals():
    q = queue.Queue()
    ch = TypingPatternChannel(q)
    now = time.time()
    # Simulate 10 regular keypresses 0.15s apart
    ch._key_times = [now + i * 0.15 for i in range(10)]
    ch._analyse()
    signals = []
    while not q.empty():
        signals.append(q.get_nowait())
    factor_ids = {s.factor_id for s in signals}
    assert "typing_speed" in factor_ids
    assert "typing_regularity" in factor_ids
    assert "keyboard_active" in factor_ids


def test_typing_channel_analyse_keyboard_active_is_one():
    q = queue.Queue()
    ch = TypingPatternChannel(q)
    now = time.time()
    ch._key_times = [now + i * 0.15 for i in range(10)]
    ch._analyse()
    sigs = {}
    while not q.empty():
        s = q.get_nowait()
        sigs[s.factor_id] = s
    assert sigs["keyboard_active"].value == 1.0


def test_typing_channel_analyse_skips_if_too_few():
    q = queue.Queue()
    ch = TypingPatternChannel(q)
    ch._key_times = [time.time()]   # only 1 timestamp
    ch._analyse()
    assert q.empty()


def test_typing_channel_speed_normalised():
    q = queue.Queue()
    ch = TypingPatternChannel(q)
    now = time.time()
    ch._key_times = [now + i * 0.1 for i in range(10)]
    ch._analyse()
    sigs = {}
    while not q.empty():
        s = q.get_nowait()
        sigs[s.factor_id] = s
    assert 0.0 <= sigs["typing_speed"].value <= 1.0


def test_typing_channel_regularity_normalised():
    q = queue.Queue()
    ch = TypingPatternChannel(q)
    now = time.time()
    ch._key_times = [now + i * 0.2 for i in range(10)]
    ch._analyse()
    sigs = {}
    while not q.empty():
        s = q.get_nowait()
        sigs[s.factor_id] = s
    assert 0.0 <= sigs["typing_regularity"].value <= 1.0


# ---------------------------------------------------------------------------
# BiometricChannel
# ---------------------------------------------------------------------------

def test_biometric_channel_name():
    assert BiometricChannel.NAME == "biometric"


def test_biometric_ingest_hrv_high_recovery():
    q = queue.Queue()
    ch = BiometricChannel(q)
    ch.ingest(hrv_ms=100.0)   # >80ms = well recovered
    sigs = {}
    while not q.empty():
        s = q.get_nowait()
        sigs[s.factor_id] = s
    assert "hrv_recovery" in sigs
    assert sigs["hrv_recovery"].value == 1.0
    assert "stress_level" in sigs
    assert sigs["stress_level"].value == 0.0


def test_biometric_ingest_hrv_low_stressed():
    q = queue.Queue()
    ch = BiometricChannel(q)
    ch.ingest(hrv_ms=10.0)    # <20ms = max stress
    sigs = {}
    while not q.empty():
        s = q.get_nowait()
        sigs[s.factor_id] = s
    assert sigs["hrv_recovery"].value == 0.0
    assert sigs["stress_level"].value == 1.0


def test_biometric_ingest_sleep_poor():
    q = queue.Queue()
    ch = BiometricChannel(q)
    ch.ingest(sleep_hrs=3.0)   # <4hrs → 0
    sigs = {}
    while not q.empty():
        s = q.get_nowait()
        sigs[s.factor_id] = s
    assert sigs["sleep_quality"].value == 0.0
    assert "cognitive_readiness" in sigs


def test_biometric_ingest_sleep_great():
    q = queue.Queue()
    ch = BiometricChannel(q)
    ch.ingest(sleep_hrs=9.0)   # >9hrs → capped at 1.0
    sigs = {}
    while not q.empty():
        s = q.get_nowait()
        sigs[s.factor_id] = s
    assert sigs["sleep_quality"].value == 1.0


def test_biometric_ingest_steps():
    q = queue.Queue()
    ch = BiometricChannel(q)
    ch.ingest(steps=5000)
    sigs = {}
    while not q.empty():
        s = q.get_nowait()
        sigs[s.factor_id] = s
    assert sigs["activity_today"].value == pytest.approx(0.5)


def test_biometric_ingest_steps_capped():
    q = queue.Queue()
    ch = BiometricChannel(q)
    ch.ingest(steps=20000)
    sigs = {}
    while not q.empty():
        s = q.get_nowait()
        sigs[s.factor_id] = s
    assert sigs["activity_today"].value == 1.0


def test_biometric_ingest_soreness():
    q = queue.Queue()
    ch = BiometricChannel(q)
    ch.ingest(soreness=5)
    sigs = {}
    while not q.empty():
        s = q.get_nowait()
        sigs[s.factor_id] = s
    assert sigs["physical_soreness"].value == pytest.approx(0.5)


def test_biometric_ingest_heart_rate_athletic():
    q = queue.Queue()
    ch = BiometricChannel(q)
    ch.ingest(heart_rate=45)
    sigs = {}
    while not q.empty():
        s = q.get_nowait()
        sigs[s.factor_id] = s
    assert sigs["cardio_state"].value == 1.0


def test_biometric_ingest_heart_rate_elevated():
    q = queue.Queue()
    ch = BiometricChannel(q)
    ch.ingest(heart_rate=120)
    sigs = {}
    while not q.empty():
        s = q.get_nowait()
        sigs[s.factor_id] = s
    assert sigs["cardio_state"].value == 0.0


def test_biometric_ingest_no_args_emits_nothing():
    q = queue.Queue()
    ch = BiometricChannel(q)
    ch.ingest()
    assert q.empty()


def test_biometric_ingest_cognitive_readiness_range():
    q = queue.Queue()
    ch = BiometricChannel(q)
    ch.ingest(sleep_hrs=8.0)
    sigs = {}
    while not q.empty():
        s = q.get_nowait()
        sigs[s.factor_id] = s
    assert 0.0 <= sigs["cognitive_readiness"].value <= 1.0


# ---------------------------------------------------------------------------
# VoiceChannel
# ---------------------------------------------------------------------------

def test_voice_channel_name():
    assert VoiceChannel.NAME == "voice"


def test_voice_channel_wake_word_lowercased():
    q = queue.Queue()
    ch = VoiceChannel(q, wake_word="Hey PRISM")
    assert ch._wake_word == "hey prism"


def test_voice_channel_try_init_returns_false_without_pyaudio():
    q = queue.Queue()
    ch = VoiceChannel(q)
    # pyaudio is not installed in the test environment
    result = ch._try_init()
    assert result is False


# ---------------------------------------------------------------------------
# ScreenContextChannel
# ---------------------------------------------------------------------------

def test_screen_channel_name():
    assert ScreenContextChannel.NAME == "screen"


def test_screen_channel_disabled_by_default():
    q = queue.Queue()
    ch = ScreenContextChannel(q)
    assert not ch._enabled


# ---------------------------------------------------------------------------
# ContextFuser
# ---------------------------------------------------------------------------

def _make_signal(factor_id: str, value: float, confidence: float,
                 channel: str = "test", age_seconds: float = 0.0) -> ContextSignal:
    return ContextSignal(
        channel    = channel,
        factor_id  = factor_id,
        value      = value,
        confidence = confidence,
        timestamp  = time.time() - age_seconds,
    )


def test_fuser_empty_state():
    fuser = ContextFuser(queue.Queue())
    state = fuser.current_state()
    assert isinstance(state, ContextState)
    assert state.factors == {}
    assert state.confidence == {}


def test_fuser_ingests_signal_via_queue():
    q = queue.Queue()
    fuser = ContextFuser(q)
    fuser.start()
    q.put(_make_signal("stress_level", 0.8, 0.9))
    time.sleep(0.3)
    state = fuser.current_state()
    fuser.stop()
    assert "stress_level" in state.factors
    assert 0.0 <= state.factors["stress_level"] <= 1.0


def test_fuser_single_signal_value_preserved():
    q = queue.Queue()
    fuser = ContextFuser(q)
    fuser.start()
    q.put(_make_signal("hrv_recovery", 0.75, 0.9))
    time.sleep(0.3)
    state = fuser.current_state()
    fuser.stop()
    assert state.factors["hrv_recovery"] == pytest.approx(0.75, abs=0.01)


def test_fuser_confidence_recorded():
    q = queue.Queue()
    fuser = ContextFuser(q)
    fuser.start()
    q.put(_make_signal("sleep_quality", 0.6, 0.95))
    time.sleep(0.3)
    state = fuser.current_state()
    fuser.stop()
    assert state.confidence["sleep_quality"] == pytest.approx(0.95, abs=0.01)


def test_fuser_old_signals_excluded():
    q = queue.Queue()
    fuser = ContextFuser(q)
    # Inject signal directly as too-old (beyond window)
    old_sig = _make_signal("old_factor", 0.9, 0.9,
                           age_seconds=ContextFuser.WINDOW_SECONDS + 10)
    with fuser._lock:
        fuser._signals["old_factor"] = [old_sig]
    state = fuser.current_state()
    assert "old_factor" not in state.factors


def test_fuser_multiple_factors():
    q = queue.Queue()
    fuser = ContextFuser(q)
    fuser.start()
    q.put(_make_signal("a", 0.3, 0.8))
    q.put(_make_signal("b", 0.7, 0.6))
    time.sleep(0.3)
    state = fuser.current_state()
    fuser.stop()
    assert "a" in state.factors
    assert "b" in state.factors


def test_fuser_active_channels_populated():
    q = queue.Queue()
    fuser = ContextFuser(q)
    fuser.start()
    q.put(_make_signal("x", 0.5, 0.8, channel="biometric"))
    time.sleep(0.3)
    state = fuser.current_state()
    fuser.stop()
    assert "biometric" in state.active_channels


def test_fuser_summarise_high_stress():
    summary = ContextFuser._summarise({"stress_level": 0.8})
    assert "high stress" in summary


def test_fuser_summarise_well_recovered():
    summary = ContextFuser._summarise({"hrv_recovery": 0.8})
    assert "well recovered" in summary


def test_fuser_summarise_sleep_deprived():
    summary = ContextFuser._summarise({"sleep_quality": 0.3})
    assert "sleep-deprived" in summary


def test_fuser_summarise_focused():
    summary = ContextFuser._summarise({"screen_focus": 0.9})
    assert "focused" in summary


def test_fuser_summarise_speaking():
    summary = ContextFuser._summarise({"voice_active": 0.8})
    assert "actively speaking" in summary


def test_fuser_summarise_normal_context():
    # Explicitly set factors that don't trigger any condition
    summary = ContextFuser._summarise({
        "stress_level": 0.3,
        "hrv_recovery": 0.5,
        "sleep_quality": 0.6,
        "screen_focus": 0.5,
        "voice_active": 0.2,
    })
    assert summary == "normal context"


def test_fuser_summarise_multiple_conditions():
    summary = ContextFuser._summarise({"stress_level": 0.8, "sleep_quality": 0.2})
    assert "high stress" in summary
    assert "sleep-deprived" in summary


def test_fuser_weighted_average_two_signals():
    q = queue.Queue()
    fuser = ContextFuser(q)
    fuser.start()
    # Same factor, two signals — result should be between the two values
    q.put(_make_signal("factor_x", 0.2, 0.9))
    q.put(_make_signal("factor_x", 0.8, 0.9))
    time.sleep(0.3)
    state = fuser.current_state()
    fuser.stop()
    val = state.factors["factor_x"]
    assert 0.2 <= val <= 0.8


# ---------------------------------------------------------------------------
# PrismPerception (orchestrator)
# ---------------------------------------------------------------------------

def test_prism_perception_instantiates():
    p = PrismPerception(
        enable_voice=False, enable_screen=False,
        enable_biometric=False, enable_system=False, enable_typing=False,
    )
    assert p is not None


def test_prism_perception_setup_classmethod():
    p = PrismPerception.setup(
        enable_voice=False, enable_screen=False,
        enable_biometric=False, enable_system=False, enable_typing=False,
    )
    assert isinstance(p, PrismPerception)


def test_prism_perception_current_context_returns_state():
    p = PrismPerception(
        enable_voice=False, enable_screen=False,
        enable_biometric=False, enable_system=False, enable_typing=False,
    )
    p.start()
    state = p.current_context()
    p.stop()
    assert isinstance(state, ContextState)


def test_prism_perception_status_keys():
    p = PrismPerception(
        enable_voice=False, enable_screen=False,
        enable_biometric=False, enable_system=False, enable_typing=False,
    )
    p.start()
    status = p.status()
    p.stop()
    assert "active_channels" in status
    assert "factor_count" in status
    assert "summary" in status
    assert "factors" in status


def test_prism_perception_ingest_biometrics():
    p = PrismPerception(
        enable_voice=False, enable_screen=False,
        enable_biometric=True, enable_system=False, enable_typing=False,
    )
    p.start()
    p.ingest_biometrics(hrv_ms=60.0, sleep_hrs=7.5, steps=8000)
    # Poll until all expected factors appear (or timeout after 2s)
    deadline = time.time() + 2.0
    state = None
    while time.time() < deadline:
        state = p.current_context()
        if all(k in state.factors for k in ("hrv_recovery", "sleep_quality", "activity_today")):
            break
        time.sleep(0.05)
    p.stop()
    assert state is not None
    assert "hrv_recovery" in state.factors
    assert "sleep_quality" in state.factors
    assert "activity_today" in state.factors


def test_prism_perception_record_keypress_with_typing_enabled():
    p = PrismPerception(
        enable_voice=False, enable_screen=False,
        enable_biometric=False, enable_system=False, enable_typing=True,
    )
    for _ in range(5):
        p.record_keypress()
    assert len(p._typing._key_times) == 5


def test_prism_perception_record_keypress_without_typing():
    p = PrismPerception(
        enable_voice=False, enable_screen=False,
        enable_biometric=False, enable_system=False, enable_typing=False,
    )
    p.record_keypress()   # should not raise


def test_prism_perception_channels_registered():
    p = PrismPerception(
        enable_voice=False, enable_screen=False,
        enable_biometric=True, enable_system=True, enable_typing=True,
    )
    names = [c.NAME for c in p._channels]
    assert "system" in names
    assert "typing" in names
    assert "biometric" in names
    assert "voice" not in names
    assert "screen" not in names


def test_prism_perception_voice_channel_registered_when_enabled():
    p = PrismPerception(
        enable_voice=True, enable_screen=False,
        enable_biometric=False, enable_system=False, enable_typing=False,
    )
    names = [c.NAME for c in p._channels]
    assert "voice" in names


def test_prism_perception_screen_channel_registered_when_enabled():
    p = PrismPerception(
        enable_voice=False, enable_screen=True,
        enable_biometric=False, enable_system=False, enable_typing=False,
    )
    names = [c.NAME for c in p._channels]
    assert "screen" in names


def test_prism_perception_ingest_biometrics_no_biometric_channel():
    # Should not raise even if biometric channel is not present
    p = PrismPerception(
        enable_voice=False, enable_screen=False,
        enable_biometric=False, enable_system=False, enable_typing=False,
    )
    p.ingest_biometrics(hrv_ms=50.0)


def test_prism_perception_stop_does_not_raise():
    p = PrismPerception(
        enable_voice=False, enable_screen=False,
        enable_biometric=False, enable_system=False, enable_typing=False,
    )
    p.start()
    p.stop()


def test_prism_perception_factors_in_range():
    p = PrismPerception(
        enable_voice=False, enable_screen=False,
        enable_biometric=True, enable_system=False, enable_typing=False,
    )
    p.start()
    p.ingest_biometrics(hrv_ms=55.0, sleep_hrs=7.0, steps=6000,
                        heart_rate=65, soreness=3)
    time.sleep(0.3)
    state = p.current_context()
    p.stop()
    for factor_id, value in state.factors.items():
        assert 0.0 <= value <= 1.0, f"{factor_id}={value} out of range"
