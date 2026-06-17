"""M12b — denial → standing-rule extraction.

Covers:
  * PrismInstructions.classify_denial classifier (pure)
  * PrismAgent.record_denial dual-write (task-scoped + standing)
  * POST /device/approve confirmation note varies by classification
"""
from __future__ import annotations

import types
from unittest.mock import MagicMock

import pytest

from prism_instructions import PrismInstructions

# ---------------------------------------------------------------------------
# classify_denial — pure classifier
# ---------------------------------------------------------------------------

def test_classify_empty_returns_none():
    assert PrismInstructions.classify_denial("send_email", "") == (None, None)
    assert PrismInstructions.classify_denial("send_email", "   ") == (None, None)


def test_classify_one_shot_returns_none():
    text, trig = PrismInstructions.classify_denial("send_email", "not right now")
    assert text is None and trig is None


def test_classify_never_email_picks_email_category():
    text, trig = PrismInstructions.classify_denial(
        "send_email", "never send work emails after 8pm"
    )
    assert trig == "email"
    assert "never send" in text


def test_classify_always_calendar():
    text, trig = PrismInstructions.classify_denial(
        "schedule_event", "always ask before booking meetings"
    )
    assert trig == "calendar"
    assert "always ask" in text


def test_classify_falls_back_to_task_slug():
    # Reason has a marker but no category keyword; task slug carries it.
    text, trig = PrismInstructions.classify_denial(
        "send_email", "never do this"
    )
    assert trig == "email"
    assert text == "never do this"


def test_classify_falls_back_to_always():
    # Neither reason nor task slug carries a TRIGGER_MAP category keyword.
    text, trig = PrismInstructions.classify_denial(
        "weird_thing", "from now on be careful"
    )
    assert trig == "always"
    assert text.startswith("from now on")


def test_classify_truncates_long_text():
    long = "never " + ("x" * 1000)
    text, trig = PrismInstructions.classify_denial("send_email", long)
    assert trig == "email"
    assert len(text) == 300


# ---------------------------------------------------------------------------
# record_denial — agent dual-write
# ---------------------------------------------------------------------------

@pytest.fixture
def stub_agent_with_instr(tmp_path):
    from prism_agent import PrismAgent

    agent = types.SimpleNamespace()
    agent._instructions = PrismInstructions(db_path=str(tmp_path / "instr.db"))
    agent.record_denial = types.MethodType(PrismAgent.record_denial, agent)
    return agent


def test_record_denial_returns_dict(stub_agent_with_instr):
    result = stub_agent_with_instr.record_denial("send_email", {}, "not now")
    assert isinstance(result, dict)
    assert result["task_scoped"] is True
    assert result["standing_trigger"] is None
    assert result["standing_text"] is None


def test_record_denial_extracts_standing_rule(stub_agent_with_instr):
    result = stub_agent_with_instr.record_denial(
        "send_email", {}, "never send work emails after 8pm"
    )
    assert result["task_scoped"] is True
    assert result["standing_trigger"] == "email"
    assert "never send" in result["standing_text"]

    # Both rules should be queryable.
    active = stub_agent_with_instr._instructions.all_active()
    triggers = {i.trigger for i in active}
    assert "email" in triggers
    assert "send_email" in triggers


def test_record_denial_no_reason_skips_standing(stub_agent_with_instr):
    result = stub_agent_with_instr.record_denial("send_email", {}, "")
    assert result["task_scoped"] is True
    assert result["standing_trigger"] is None


def test_record_denial_no_instructions_infra_returns_empty():
    from prism_agent import PrismAgent

    agent = types.SimpleNamespace(_instructions=None)
    agent.record_denial = types.MethodType(PrismAgent.record_denial, agent)
    result = agent.record_denial("send_email", {}, "never do this")
    assert result == {
        "task_scoped": False,
        "standing_trigger": None,
        "standing_text": None,
    }


# ---------------------------------------------------------------------------
# /device/approve — confirmation note varies by classification
# ---------------------------------------------------------------------------

def _client_with_agent(agent):
    from fastapi.testclient import TestClient

    from prism_asgi import app
    from prism_state import _set_state
    _set_state(agent=agent)
    return TestClient(app, raise_server_exceptions=False)


def test_deny_with_standing_rule_says_saved_as_rule():
    agent = MagicMock()
    agent.status.return_value = {"ok": True}
    agent.record_denial.return_value = {
        "task_scoped": True,
        "standing_trigger": "email",
        "standing_text": "never send work emails after 8pm",
    }
    client = _client_with_agent(agent)
    r = client.post(
        "/device/approve",
        json={
            "approved": False,
            "task": "send_email",
            "params": {},
            "instructions": "never send work emails after 8pm",
        },
    )
    assert r.status_code == 200
    body = r.json().get("body", "")
    assert "Saved as a rule for all email requests" in body
    assert "never send work emails after 8pm" in body


def test_deny_with_one_shot_says_noted():
    agent = MagicMock()
    agent.status.return_value = {"ok": True}
    agent.record_denial.return_value = {
        "task_scoped": True,
        "standing_trigger": None,
        "standing_text": None,
    }
    client = _client_with_agent(agent)
    r = client.post(
        "/device/approve",
        json={
            "approved": False,
            "task": "send_email",
            "params": {},
            "instructions": "not now, busy",
        },
    )
    assert r.status_code == 200
    body = r.json().get("body", "")
    assert "Noted" in body
    assert "not now, busy" in body
    assert "Saved as a rule" not in body


def test_deny_with_always_trigger_says_all_requests():
    agent = MagicMock()
    agent.status.return_value = {"ok": True}
    agent.record_denial.return_value = {
        "task_scoped": True,
        "standing_trigger": "always",
        "standing_text": "from now on confirm everything",
    }
    client = _client_with_agent(agent)
    r = client.post(
        "/device/approve",
        json={
            "approved": False,
            "task": "weird_task",
            "params": {},
            "instructions": "from now on confirm everything",
        },
    )
    assert r.status_code == 200
    body = r.json().get("body", "")
    assert "Saved as a rule for all requests" in body
