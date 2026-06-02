import json
import tempfile
from unittest.mock import MagicMock, patch
from prism_autonomous import PrismAutonomous, _is_safe_code, AcquiredTool


def _make_engine(synthesised_code=None):
    router = MagicMock()
    if synthesised_code:
        router.call.return_value = (json.dumps({
            "name": "test_tool",
            "description": "test tool for unit tests",
            "requirements": [],
            "code": synthesised_code,
        }), {})
    eng = PrismAutonomous(
        llm_router=router,
        push=MagicMock(configured=False),
    )
    # Override tool dir to temp
    import pathlib, os
    eng.TOOL_DIR = pathlib.Path(tempfile.mkdtemp())
    return eng


def test_safe_code_passes():
    code = "import json\ndef execute(task, params):\n    return 'ok'"
    safe, _ = _is_safe_code(code)
    assert safe


def test_unsafe_code_blocked_eval():
    code = "def execute(task, params):\n    return eval(task)"
    safe, reason = _is_safe_code(code)
    assert not safe
    assert "eval" in reason


def test_unsafe_code_blocked_os_system():
    code = "import os\ndef execute(task, params):\n    os.system('ls')"
    safe, _ = _is_safe_code(code)
    assert not safe


def test_synthesise_and_run():
    code = "def execute(task, params):\n    return f'did: {task}'"
    eng  = _make_engine(synthesised_code=code)
    result = eng.execute_sync("test task", {})
    assert "test task" in result


def test_cached_tool_reused():
    code = "def execute(task, params):\n    return 'cached'"
    eng  = _make_engine(synthesised_code=code)
    eng.execute_sync("test caching task", {})
    # Second call should use cache (router not called again)
    call_count_before = eng._router.call.call_count
    eng.execute_sync("test caching task", {})
    assert eng._router.call.call_count == call_count_before


def test_list_tools_empty_initially():
    eng = _make_engine()
    # Fresh engine with temp dir has no tools
    assert eng.list_tools() == []


def test_can_handle_false_initially():
    eng = _make_engine()
    assert not eng.can_handle("some brand new task nobody has done before xyz123")


def test_ast_blocks_obfuscated_import():
    # String-based bypass attempt that regex would miss
    code = """
def execute(task, params):
    m = __import__('os')
    return m.getcwd()
"""
    safe, reason = _is_safe_code(code)
    assert not safe


def test_ast_blocks_attribute_access():
    code = """
import pathlib
def execute(task, params):
    p = pathlib.Path('/tmp/x')
    p.unlink()
    return 'done'
"""
    safe, reason = _is_safe_code(code)
    assert not safe


def test_subprocess_isolation():
    # Tool that would fail if run in-process (no os import allowed)
    # but succeeds in subprocess because it only uses stdlib json
    code = "import json\ndef execute(task, params):\n    return json.dumps({'task': task})"
    eng  = _make_engine(synthesised_code=code)
    result = eng.execute_sync("hello world", {})
    assert "hello" in result or "task" in result


def test_subprocess_timeout():
    # Tool that hangs — should timeout
    code = "import time\ndef execute(task, params):\n    time.sleep(999)\n    return 'done'"
    # Override safety check since time is allowed
    from prism_autonomous import _is_safe_code
    # time.sleep is fine, but we need to monkeypatch timeout to be short
    eng = _make_engine(synthesised_code=code)
    # Patch timeout to 2s for test speed
    import unittest.mock as mock
    with mock.patch('subprocess.run', side_effect=__import__('subprocess').TimeoutExpired('cmd', 2)):
        result = eng.execute_sync("hang", {})
    assert "timed out" in result.lower() or "failed" in result.lower()
