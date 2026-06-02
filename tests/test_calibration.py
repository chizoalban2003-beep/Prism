import tempfile

from prism_calibration import PrismCalibration


def _tmp_cal():
    tmp = tempfile.mktemp(suffix=".db")
    return PrismCalibration(db_path=tmp)

def test_detect_too_aggressive():
    c = _tmp_cal()
    assert c.detect("that was too aggressive") == "too_aggressive"

def test_detect_correct():
    c = _tmp_cal()
    assert c.detect("that was right") == "correct"

def test_detect_returns_none_normal():
    c = _tmp_cal()
    assert c.detect("what time is my meeting") is None

def test_process_stores_event():
    c = _tmp_cal()
    c.process("too risky", "too_aggressive", {}, None)
    history = c.history()
    assert len(history) == 1

def test_summary_non_empty():
    c = _tmp_cal()
    c.process("good call", "correct", {}, None)
    assert c.summary() != "No calibration history yet."

def test_adjustment_direction():
    c = _tmp_cal()
    event = c.process("too aggressive", "too_aggressive", {}, None)
    assert event.adjustment < 0

def test_history_filtered_by_domain():
    c = _tmp_cal()
    c.process("good call", "correct", {"domain": "sport"}, None)
    c.process("too risky", "too_aggressive", {"domain": "financial"}, None)
    sport_history = c.history(domain="sport")
    assert all(e.domain == "sport" for e in sport_history)
