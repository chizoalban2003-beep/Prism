"""Shared pytest fixtures for the PRISM test suite."""
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import pytest


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
