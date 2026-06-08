"""
tests/test_approval_flow.py
===========================
Tests for the approval flow:
  - DeviceTaskResult.needs_approval field
  - PrismDeviceAgent._check_policy returns (bool, bool)
  - device_result_card returns CardType.APPROVAL when needs_approval
  - POST /device/approve endpoint (approve and deny paths)
"""

from __future__ import annotations

import json
import sys
from unittest.mock import MagicMock

import pytest

from prism_device_agent import CapabilityMap, PrismDeviceAgent
from prism_device_executor import DeviceTaskResult
from prism_responses import CardType, approval_card, device_result_card

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_agent() -> PrismDeviceAgent:
    caps = CapabilityMap(
        cli_tools={},
        py_packages=[],
        platform=sys.platform,
        has_browser=False,
    )
    return PrismDeviceAgent(capabilities=caps)


def _needs_approval_result(task: str = "delete file /tmp/x") -> DeviceTaskResult:
    return DeviceTaskResult(
        success=False,
        output=f"Approval required for: {task}",
        tool_used="pending_approval",
        needs_approval=True,
        undo_command=json.dumps({"task": task, "params": {}}),
    )


# ---------------------------------------------------------------------------
# Unit tests — DeviceTaskResult
# ---------------------------------------------------------------------------

def test_needs_approval_flag_default_false():
    result = DeviceTaskResult(success=True, output="ok")
    assert result.needs_approval is False


def test_needs_approval_flag_set():
    result = _needs_approval_result()
    assert result.needs_approval is True


# ---------------------------------------------------------------------------
# Unit tests — _check_policy returns tuple[bool, bool]
# ---------------------------------------------------------------------------

def test_check_policy_returns_tuple():
    agent = _make_agent()
    result = agent._check_policy("list files in /tmp")
    assert isinstance(result, tuple)
    assert len(result) == 2
    allowed, needs_approval = result
    assert isinstance(allowed, bool)
    assert isinstance(needs_approval, bool)


def test_check_policy_no_policy_engine_allows():
    agent = _make_agent()
    allowed, needs_approval = agent._check_policy("list files")
    assert allowed is True
    assert needs_approval is False


def test_check_policy_policy_reject_returns_false_false():
    try:
        from prism_policy import PolicyEngine
    except ImportError:
        pytest.skip("prism_policy not available in this env")

    mock_policy = MagicMock()
    mock_policy.evaluate.return_value = (PolicyEngine.Verdict.REJECT, "Rejected")

    caps = CapabilityMap(cli_tools={}, py_packages=[], platform=sys.platform, has_browser=False)
    agent = PrismDeviceAgent(capabilities=caps, policy_engine=mock_policy)
    allowed, needs_approval = agent._check_policy("delete all files")
    assert allowed is False
    assert needs_approval is False


def test_check_policy_policy_escalate_returns_false_true():
    try:
        from prism_policy import PolicyEngine
    except ImportError:
        pytest.skip("prism_policy not available in this env")

    mock_policy = MagicMock()
    mock_policy.evaluate.return_value = (PolicyEngine.Verdict.ESCALATE, "Needs review")

    caps = CapabilityMap(cli_tools={}, py_packages=[], platform=sys.platform, has_browser=False)
    agent = PrismDeviceAgent(capabilities=caps, policy_engine=mock_policy)
    allowed, needs_approval = agent._check_policy("risky task")
    assert allowed is False
    assert needs_approval is True


# ---------------------------------------------------------------------------
# Unit tests — approval_card builder
# ---------------------------------------------------------------------------

def test_approval_card_type():
    card = approval_card("delete /tmp/x", "reason")
    assert card.card_type == CardType.APPROVAL


def test_approval_card_has_task_and_task_id():
    card = approval_card("delete /tmp/x", "reason", {"path": "/tmp/x"})
    assert "task" in card.card_data
    assert "task_id" in card.card_data
    assert card.card_data["task"] == "delete /tmp/x"
    assert len(card.card_data["task_id"]) == 8


def test_approval_card_actions():
    card = approval_card("some task", "reason")
    assert "Approve" in card.actions
    assert "Deny" in card.actions


# ---------------------------------------------------------------------------
# Unit tests — device_result_card with needs_approval
# ---------------------------------------------------------------------------

def test_approval_card_type_from_device_result_card():
    result = _needs_approval_result("delete /tmp/test")
    card = device_result_card(result, "delete /tmp/test")
    assert card.card_type == CardType.APPROVAL


def test_approval_card_has_task_from_device_result_card():
    result = _needs_approval_result("delete /tmp/test")
    card = device_result_card(result, "delete /tmp/test")
    assert "task" in card.card_data
    assert "task_id" in card.card_data


def test_normal_result_not_approval_card():
    result = DeviceTaskResult(success=True, output="done", tool_used="python_glob")
    card = device_result_card(result, "list files")
    assert card.card_type != CardType.APPROVAL


# ---------------------------------------------------------------------------
# Integration tests — POST /device/approve endpoint (FastAPI TestClient)
# ---------------------------------------------------------------------------

def _make_approval_client():
    """Build a TestClient with a mock agent wired for /device/approve."""
    from fastapi.testclient import TestClient

    from prism_asgi import app
    from prism_state import _set_state

    agent = MagicMock()
    agent.status.return_value = {"ok": True}
    agent._assistant = MagicMock()
    agent._assistant.history.return_value = []
    agent._hub = MagicMock()
    agent._hub.list_devices.return_value = []
    agent._profile = MagicMock()
    agent._profile.name = "Tester"

    # Wire a device agent that returns a denial card on approved=False
    from prism_responses import text_card
    device_agent = MagicMock()
    device_agent.execute.return_value = text_card("Execution result", "done")
    agent._device_agent = device_agent

    _set_state(agent=agent)
    return TestClient(app, raise_server_exceptions=False)


def test_deny_endpoint_cancels():
    client = _make_approval_client()
    r = client.post(
        "/device/approve",
        json={"approved": False, "task": "delete /tmp/x", "params": {}},
    )
    assert r.status_code == 200
    data = r.json()
    assert "cancel" in data.get("body", "").lower() or data.get("type") == "approval"


def test_approve_endpoint_executes():
    client = _make_approval_client()
    r = client.post(
        "/device/approve",
        json={"approved": True, "task": "list files in /tmp", "params": {}},
    )
    assert r.status_code == 200
    data = r.json()
    assert "type" in data
    assert data.get("type") != "approval"
