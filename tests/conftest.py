"""Shared pytest fixtures for the PRISM test suite."""
import os
import pathlib
import sys

# Disable HTTP bearer auth in tests. The middleware enforces auth when a
# token is configured via env or ~/.prism/auth_token; tests would fail if
# a developer has the daemon running locally and the file exists. Setting
# this before any prism_* import ensures prism_asgi sees the override.
os.environ["PRISM_AUTH_DISABLE"] = "1"

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
