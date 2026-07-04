"""Shared pytest fixtures for the PRISM test suite."""
import os
import pathlib
import sys

# Disable HTTP bearer auth in tests. The middleware enforces auth when a
# token is configured via env or ~/.prism/auth_token; tests would fail if
# a developer has the daemon running locally and the file exists. Setting
# this before any prism_* import ensures prism_asgi sees the override.
os.environ["PRISM_AUTH_DISABLE"] = "1"

# Hermetic config: don't read the developer's untracked prism_config.toml
# or ~/.prism/prism_config.toml. Their contents (a preferred cloud provider,
# a rotated-out key, an uninstalled ollama model) leak network latency and
# nondeterminism into every test that constructs a PrismAgent. CI has
# neither file, so this makes local runs match CI.
os.environ["PRISM_HERMETIC_CONFIG"] = "1"

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import pytest


@pytest.fixture(autouse=True)
def _isolate_session_manager(tmp_path_factory):
    """Reset the global SessionManager singleton (and any leaked active session)
    to a fresh per-test DB. The singleton was shared across tests, causing an
    intermittent ordering flake in the WS/chat session-persistence tests."""
    try:
        import prism_session_manager
        d = tmp_path_factory.mktemp("sessions")
        prism_session_manager.reset_session_manager(db_path=str(d / "sessions.db"))
    except Exception:
        pass
    try:
        import prism_state
        prism_state._state.pop("active_session_id", None)
    except Exception:
        pass
    yield


@pytest.fixture
def temp_db(tmp_path):
    """A temporary SQLite database path for tests that use SQLite."""
    return str(tmp_path / "test.db")


@pytest.fixture
def temp_dir(tmp_path):
    """A temporary directory for file operation tests."""
    return str(tmp_path)


@pytest.fixture
def mock_llm_router():
    """A mock LLM router that returns predictable responses."""
    class MockRouter:
        def call(self, prompt, **kwargs):
            return '{"result": "mock"}', "mock/model"
        def discover(self, force=False):
            return []
        def best(self, min_capability=1):
            return None
    return MockRouter()


class _OfflineLLM:
    """Handle returned by the `offline_llm` fixture.

    Tests can inspect calls and inject deterministic replies:

        def test_x(offline_llm):
            offline_llm.set_reply('{"findings": {"phone_number": "555"}}')
            ...
            assert offline_llm.calls  # list of (prompt, kwargs)
    """
    def __init__(self):
        self.calls: list[tuple[str, dict]] = []
        self._default_reply = ""
        self._reply_queue: list[str] = []

    def set_reply(self, text: str) -> None:
        self._default_reply = text

    def queue_reply(self, text: str) -> None:
        self._reply_queue.append(text)

    def _next_reply(self) -> str:
        if self._reply_queue:
            return self._reply_queue.pop(0)
        return self._default_reply


@pytest.fixture
def offline_llm(monkeypatch):
    """Run a test without any real LLM or outbound HTTP.

    Patches every LLM call site (LLMRouter, PrismPlanner, PrismCollaborator
    web search) and blocks urllib outbound traffic from those modules so a
    missing Ollama/Claude can no longer hang or time out.

    Opt-in: declare `offline_llm` as a fixture argument. Returns an
    `_OfflineLLM` handle for inspecting calls and queueing replies.
    """
    handle = _OfflineLLM()

    # --- LLMRouter: discover/best/call ---------------------------------
    from prism_llm_router import LLMOption, LLMRouter

    stub_option = LLMOption(
        provider="stdlib", model="stdlib", endpoint="",
        available=True, capability=0, notes="offline_llm fixture",
    )

    def _discover(self, force=False):
        self._options = [stub_option]
        self._discovered = True
        return self._options

    def _best(self, min_capability=1, phase_hint=None):
        return stub_option if min_capability <= 0 else None

    def _call(self, prompt, **kwargs):
        handle.calls.append((prompt, dict(kwargs)))
        return handle._next_reply(), "stdlib/stdlib"

    monkeypatch.setattr(LLMRouter, "discover", _discover, raising=True)
    monkeypatch.setattr(LLMRouter, "best", _best, raising=True)
    monkeypatch.setattr(LLMRouter, "call", _call, raising=True)

    # --- PrismPlanner: bypasses LLMRouter, talks to urllib directly ----
    try:
        from prism_planner import PrismPlanner
        monkeypatch.setattr(
            PrismPlanner, "_call_llm",
            lambda self, prompt: handle._next_reply(),
            raising=True,
        )
    except Exception:
        pass

    # --- Block outbound HTTP from LLM call sites -----------------------
    # We monkeypatch the module-level `urllib.request.urlopen` references
    # rather than the global, so unrelated code (e.g. fixtures themselves)
    # is unaffected.
    class _Blocked(RuntimeError):
        pass

    def _blocked(*_a, **_k):
        raise _Blocked("outbound HTTP blocked by offline_llm fixture")

    for mod_name in ("prism_llm_router", "prism_planner", "prism_collaborator"):
        try:
            mod = __import__(mod_name)
            if hasattr(mod, "urllib"):
                monkeypatch.setattr(
                    mod.urllib.request, "urlopen", _blocked, raising=True
                )
        except Exception:
            pass

    yield handle
