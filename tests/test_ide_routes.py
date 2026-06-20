"""Tests for prism_routes_ide.py — IDE integration routes."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from prism_state import _set_state

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clean_state():
    _set_state(agent=None)
    yield
    _set_state(agent=None)


@pytest.fixture
def client():
    from prism_asgi import app

    return TestClient(app)


def _make_llm_router(reply: str = "LLM response text"):
    llm = MagicMock()
    llm.call.return_value = reply
    return llm


def _make_agent(llm_reply: str = "LLM response text"):
    agent = MagicMock()
    agent._router = _make_llm_router(llm_reply)
    agent._phase = None
    agent._soul = None
    return agent


# ---------------------------------------------------------------------------
# GET /ide/status
# ---------------------------------------------------------------------------


class TestIdeStatus:
    def test_returns_200(self, client):
        r = client.get("/ide/status")
        assert r.status_code == 200

    def test_ok_true(self, client):
        r = client.get("/ide/status")
        assert r.json()["ok"] is True

    def test_agent_not_ready_without_state(self, client):
        r = client.get("/ide/status")
        d = r.json()
        assert d["agent_ready"] is False
        assert d["llm_ready"] is False

    def test_agent_ready_with_agent(self, client):
        _set_state(agent=_make_agent())
        r = client.get("/ide/status")
        d = r.json()
        assert d["agent_ready"] is True
        assert d["llm_ready"] is True

    def test_server_field_present(self, client):
        r = client.get("/ide/status")
        assert r.json()["server"] == "prism-asgi"

    def test_timestamp_present(self, client):
        r = client.get("/ide/status")
        assert r.json()["timestamp"] > 0

    def test_phase_field_present(self, client):
        r = client.get("/ide/status")
        assert "phase" in r.json()


# ---------------------------------------------------------------------------
# POST /ide/explain
# ---------------------------------------------------------------------------


class TestIdeExplain:
    def test_returns_200(self, client):
        _set_state(agent=_make_agent())
        r = client.post("/ide/explain", json={"code": "x = 1", "language": "python"})
        assert r.status_code == 200

    def test_returns_explanation_key(self, client):
        _set_state(agent=_make_agent("This assigns 1 to x."))
        r = client.post("/ide/explain", json={"code": "x = 1", "language": "python"})
        assert "explanation" in r.json()
        assert r.json()["explanation"] == "This assigns 1 to x."

    def test_unwraps_router_tuple(self, client):
        # The real LLMRouter.call returns a (text, model) tuple. The endpoint
        # must return the text, not a stringified tuple like "('...', 'mistral')".
        agent = MagicMock()
        agent._phase = None
        agent._soul = None
        agent._router = MagicMock()
        agent._router.call.return_value = ("This assigns 1 to x.", "mistral")
        _set_state(agent=agent)
        r = client.post("/ide/explain", json={"code": "x = 1", "language": "python"})
        assert r.json()["explanation"] == "This assigns 1 to x."

    def test_empty_llm_does_not_leak_tuple(self, client):
        agent = MagicMock()
        agent._phase = None
        agent._soul = None
        agent._router = MagicMock()
        agent._router.call.return_value = ("", "none")
        _set_state(agent=agent)
        r = client.post("/ide/explain", json={"code": "x = 1", "language": "python"})
        assert r.json()["explanation"] == ""
        assert "none" not in r.json()["explanation"]

    def test_missing_code_returns_400(self, client):
        r = client.post("/ide/explain", json={"language": "python"})
        assert r.status_code == 400
        assert "code" in r.json()["error"]

    def test_empty_code_returns_400(self, client):
        r = client.post("/ide/explain", json={"code": "", "language": "python"})
        assert r.status_code == 400

    def test_no_agent_still_returns_200(self, client):
        """Without agent, LLM helper returns a placeholder string — not a 5xx."""
        r = client.post("/ide/explain", json={"code": "x = 1", "language": "python"})
        assert r.status_code == 200
        assert "explanation" in r.json()

    def test_language_echoed(self, client):
        _set_state(agent=_make_agent())
        r = client.post("/ide/explain", json={"code": "x = 1", "language": "python"})
        assert r.json()["language"] == "python"

    def test_optional_question_accepted(self, client):
        _set_state(agent=_make_agent())
        r = client.post(
            "/ide/explain",
            json={"code": "x = 1", "language": "python", "question": "What is x?"},
        )
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# POST /ide/complete
# ---------------------------------------------------------------------------


class TestIdeComplete:
    def test_returns_200(self, client):
        _set_state(agent=_make_agent())
        r = client.post(
            "/ide/complete",
            json={"code": "def foo():", "cursor_line": 1, "cursor_col": 10, "language": "python"},
        )
        assert r.status_code == 200

    def test_missing_code_returns_400(self, client):
        r = client.post("/ide/complete", json={"language": "python", "cursor_line": 0, "cursor_col": 0})
        assert r.status_code == 400

    def test_completion_key_present(self, client):
        _set_state(agent=_make_agent("    pass"))
        r = client.post(
            "/ide/complete",
            json={"code": "def foo():", "cursor_line": 1, "cursor_col": 10, "language": "python"},
        )
        assert "completion" in r.json()

    def test_optional_filename_accepted(self, client):
        _set_state(agent=_make_agent())
        r = client.post(
            "/ide/complete",
            json={
                "code": "x = ",
                "cursor_line": 1,
                "cursor_col": 4,
                "language": "python",
                "filename": "main.py",
            },
        )
        assert r.status_code == 200
        assert r.json()["filename"] == "main.py"


# ---------------------------------------------------------------------------
# POST /ide/review
# ---------------------------------------------------------------------------


class TestIdeReview:
    def test_returns_200(self, client):
        _set_state(agent=_make_agent())
        r = client.post("/ide/review", json={"code": "x = 1\nprint(x)", "language": "python"})
        assert r.status_code == 200

    def test_review_key_present(self, client):
        _set_state(agent=_make_agent("Looks good."))
        r = client.post("/ide/review", json={"code": "x = 1\nprint(x)", "language": "python"})
        assert r.json()["review"] == "Looks good."

    def test_missing_code_returns_400(self, client):
        r = client.post("/ide/review", json={"language": "python"})
        assert r.status_code == 400

    def test_filename_echoed(self, client):
        _set_state(agent=_make_agent())
        r = client.post(
            "/ide/review",
            json={"code": "x = 1", "language": "python", "filename": "app.py"},
        )
        assert r.json()["filename"] == "app.py"


# ---------------------------------------------------------------------------
# POST /ide/fix
# ---------------------------------------------------------------------------


class TestIdeFix:
    def test_returns_200(self, client):
        _set_state(agent=_make_agent())
        r = client.post(
            "/ide/fix",
            json={"code": "print(x)", "error_message": "NameError: name 'x' is not defined", "language": "python"},
        )
        assert r.status_code == 200

    def test_fix_key_present(self, client):
        _set_state(agent=_make_agent("x = 0\nprint(x)"))
        r = client.post(
            "/ide/fix",
            json={"code": "print(x)", "error_message": "NameError", "language": "python"},
        )
        assert "fix" in r.json()

    def test_missing_code_returns_400(self, client):
        r = client.post("/ide/fix", json={"error_message": "NameError", "language": "python"})
        assert r.status_code == 400

    def test_missing_error_message_returns_400(self, client):
        r = client.post("/ide/fix", json={"code": "print(x)", "language": "python"})
        assert r.status_code == 400

    def test_error_message_echoed(self, client):
        _set_state(agent=_make_agent())
        r = client.post(
            "/ide/fix",
            json={"code": "print(x)", "error_message": "NameError", "language": "python"},
        )
        assert r.json()["error_message"] == "NameError"


# ---------------------------------------------------------------------------
# POST /ide/chat
# ---------------------------------------------------------------------------


class TestIdeChat:
    def test_returns_200(self, client):
        _set_state(agent=_make_agent())
        r = client.post("/ide/chat", json={"message": "Hello PRISM"})
        assert r.status_code == 200

    def test_reply_key_present(self, client):
        _set_state(agent=_make_agent("Hi there!"))
        r = client.post("/ide/chat", json={"message": "Hello PRISM"})
        assert r.json()["reply"] == "Hi there!"

    def test_missing_message_returns_400(self, client):
        r = client.post("/ide/chat", json={})
        assert r.status_code == 400
        assert "message" in r.json()["error"]

    def test_empty_message_returns_400(self, client):
        r = client.post("/ide/chat", json={"message": ""})
        assert r.status_code == 400

    def test_optional_code_context_accepted(self, client):
        _set_state(agent=_make_agent())
        r = client.post(
            "/ide/chat",
            json={"message": "Explain this", "code_context": "for i in range(10): pass"},
        )
        assert r.status_code == 200

    def test_no_agent_returns_200_with_placeholder(self, client):
        r = client.post("/ide/chat", json={"message": "Hello?"})
        assert r.status_code == 200
        assert "reply" in r.json()


# ---------------------------------------------------------------------------
# GET /ide/context
# ---------------------------------------------------------------------------


class TestIdeContext:
    def test_returns_200(self, client):
        r = client.get("/ide/context")
        assert r.status_code == 200

    def test_agent_ready_false_without_state(self, client):
        r = client.get("/ide/context")
        d = r.json()
        assert d["agent_ready"] is False

    def test_agent_ready_true_with_agent(self, client):
        _set_state(agent=_make_agent())
        r = client.get("/ide/context")
        assert r.json()["agent_ready"] is True

    def test_phase_field_present(self, client):
        r = client.get("/ide/context")
        assert "phase" in r.json()

    def test_timestamp_present(self, client):
        r = client.get("/ide/context")
        assert r.json()["timestamp"] > 0
