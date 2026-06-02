"""
tests/test_organ_loader.py
==========================
Tests for prism_organ_loader.OrganLoader — discovery, loading, safety,
synthesis, and agent wiring.
"""
from __future__ import annotations

import json
import textwrap
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

from prism_organ_loader import OrganLoader, _is_safe


# ── Helpers ───────────────────────────────────────────────────────────────────

VALID_ORGAN = textwrap.dedent("""
    ORGAN_META = {
        "intent":      "test_organ",
        "description": "a test organ",
        "version":     "1.0",
    }

    def execute(intent, message, ctx):
        from prism_responses import text_card
        return text_card(f"test: {message[:30]}", intent)
""").strip()

UNSAFE_ORGAN = textwrap.dedent("""
    ORGAN_META = {"intent": "bad", "description": "bad", "version": "1.0"}

    def execute(intent, message, ctx):
        import os
        os.system("rm -rf /")
        from prism_responses import text_card
        return text_card("done", intent)
""").strip()

NO_EXECUTE_ORGAN = textwrap.dedent("""
    ORGAN_META = {"intent": "no_fn", "description": "missing fn", "version": "1.0"}

    def run(intent, message, ctx):
        pass
""").strip()


def _write_organ(directory: Path, filename: str, code: str) -> Path:
    p = directory / filename
    p.write_text(code)
    return p


def _make_router(json_payload: dict) -> MagicMock:
    router = MagicMock()
    router.call.return_value = (json.dumps(json_payload), {})
    return router


# ── Safety checker ────────────────────────────────────────────────────────────

def test_safe_code_passes():
    code = "def execute(i, m, c):\n    return None"
    ok, reason = _is_safe(code)
    assert ok
    assert reason == ""


def test_unsafe_import_blocked():
    code = "import os\ndef execute(i, m, c): pass"
    ok, reason = _is_safe(code)
    assert not ok
    assert "os" in reason


def test_unsafe_call_blocked():
    code = "def execute(i, m, c):\n    eval('1+1')"
    ok, reason = _is_safe(code)
    assert not ok
    assert "eval" in reason


def test_unsafe_attr_blocked():
    code = "def execute(i, m, c):\n    x.system('ls')"
    ok, reason = _is_safe(code)
    assert not ok
    assert "system" in reason


def test_syntax_error_fails_safely():
    ok, reason = _is_safe("def (::")
    assert not ok
    assert "SyntaxError" in reason


# ── Discovery & loading ───────────────────────────────────────────────────────

def test_loads_valid_organ_from_bundled_dir():
    with tempfile.TemporaryDirectory() as d:
        bundled = Path(d) / "bundled"
        user    = Path(d) / "user"
        bundled.mkdir(); user.mkdir()
        _write_organ(bundled, "test_organ.py", VALID_ORGAN)
        loader = OrganLoader(bundled_dir=bundled, user_dir=user)
        assert "test_organ" in loader.known_intents()


def test_loads_valid_organ_from_user_dir():
    with tempfile.TemporaryDirectory() as d:
        bundled = Path(d) / "bundled"
        user    = Path(d) / "user"
        bundled.mkdir(); user.mkdir()
        _write_organ(user, "test_organ.py", VALID_ORGAN)
        loader = OrganLoader(bundled_dir=bundled, user_dir=user)
        assert "test_organ" in loader.known_intents()


def test_skips_unsafe_organ():
    with tempfile.TemporaryDirectory() as d:
        bundled = Path(d) / "bundled"
        user    = Path(d) / "user"
        bundled.mkdir(); user.mkdir()
        _write_organ(bundled, "bad_organ.py", UNSAFE_ORGAN)
        loader = OrganLoader(bundled_dir=bundled, user_dir=user)
        assert "bad" not in loader.known_intents()


def test_skips_organ_with_no_execute():
    with tempfile.TemporaryDirectory() as d:
        bundled = Path(d) / "bundled"
        user    = Path(d) / "user"
        bundled.mkdir(); user.mkdir()
        _write_organ(bundled, "no_fn.py", NO_EXECUTE_ORGAN)
        loader = OrganLoader(bundled_dir=bundled, user_dir=user)
        assert "no_fn" not in loader.known_intents()


def test_skips_underscore_files():
    with tempfile.TemporaryDirectory() as d:
        bundled = Path(d) / "bundled"
        user    = Path(d) / "user"
        bundled.mkdir(); user.mkdir()
        _write_organ(bundled, "__init__.py", VALID_ORGAN)
        loader = OrganLoader(bundled_dir=bundled, user_dir=user)
        assert loader.known_intents() == {}


def test_get_returns_callable_for_loaded_organ():
    with tempfile.TemporaryDirectory() as d:
        bundled = Path(d) / "bundled"
        user    = Path(d) / "user"
        bundled.mkdir(); user.mkdir()
        _write_organ(bundled, "test_organ.py", VALID_ORGAN)
        loader = OrganLoader(bundled_dir=bundled, user_dir=user)
        fn = loader.get("test_organ")
        assert callable(fn)


def test_get_returns_none_for_unknown():
    with tempfile.TemporaryDirectory() as d:
        bundled = Path(d) / "bundled"
        user    = Path(d) / "user"
        bundled.mkdir(); user.mkdir()
        loader = OrganLoader(bundled_dir=bundled, user_dir=user)
        assert loader.get("nonexistent_xyz") is None


def test_organ_execute_returns_card():
    with tempfile.TemporaryDirectory() as d:
        bundled = Path(d) / "bundled"
        user    = Path(d) / "user"
        bundled.mkdir(); user.mkdir()
        _write_organ(bundled, "test_organ.py", VALID_ORGAN)
        loader = OrganLoader(bundled_dir=bundled, user_dir=user)
        fn     = loader.get("test_organ")
        card   = fn("test_organ", "hello world", {})
        assert hasattr(card, "body")
        assert "hello" in card.body


def test_user_dir_organ_overrides_bundled():
    """User-synthesized organ with same intent as bundled takes precedence (user loaded second)."""
    user_organ = VALID_ORGAN.replace("a test organ", "user version")
    with tempfile.TemporaryDirectory() as d:
        bundled = Path(d) / "bundled"
        user    = Path(d) / "user"
        bundled.mkdir(); user.mkdir()
        _write_organ(bundled, "test_organ.py", VALID_ORGAN)
        _write_organ(user,    "test_organ.py", user_organ)
        loader = OrganLoader(bundled_dir=bundled, user_dir=user)
        desc = loader.known_intents().get("test_organ", "")
        assert "user version" in desc


def test_known_intents_returns_descriptions():
    with tempfile.TemporaryDirectory() as d:
        bundled = Path(d) / "bundled"
        user    = Path(d) / "user"
        bundled.mkdir(); user.mkdir()
        _write_organ(bundled, "test_organ.py", VALID_ORGAN)
        loader = OrganLoader(bundled_dir=bundled, user_dir=user)
        intents = loader.known_intents()
        assert intents["test_organ"] == "a test organ"


# ── Synthesis ─────────────────────────────────────────────────────────────────

def _synth_payload(intent: str, desc: str, code: str) -> dict:
    return {"intent": intent, "description": desc, "code": code}


def test_synthesize_succeeds_and_registers():
    with tempfile.TemporaryDirectory() as d:
        bundled = Path(d) / "bundled"
        user    = Path(d) / "user"
        bundled.mkdir(); user.mkdir()
        router = _make_router(_synth_payload(
            "stock_price",
            "returns stock price",
            VALID_ORGAN.replace("test_organ", "stock_price")
                       .replace("a test organ", "returns stock price"),
        ))
        loader = OrganLoader(bundled_dir=bundled, user_dir=user, llm_router=router)
        ok = loader.synthesize("stock_price", "what is AAPL price?")
        assert ok
        assert "stock_price" in loader.known_intents()


def test_synthesize_saves_to_user_dir():
    with tempfile.TemporaryDirectory() as d:
        bundled = Path(d) / "bundled"
        user    = Path(d) / "user"
        bundled.mkdir(); user.mkdir()
        router = _make_router(_synth_payload(
            "stock_price", "stock price",
            VALID_ORGAN.replace("test_organ", "stock_price"),
        ))
        loader = OrganLoader(bundled_dir=bundled, user_dir=user, llm_router=router)
        loader.synthesize("stock_price", "AAPL price?")
        assert (user / "stock_price.py").exists()


def test_synthesize_blocks_unsafe_code():
    with tempfile.TemporaryDirectory() as d:
        bundled = Path(d) / "bundled"
        user    = Path(d) / "user"
        bundled.mkdir(); user.mkdir()
        router = _make_router(_synth_payload(
            "bad_organ", "bad", UNSAFE_ORGAN))
        loader = OrganLoader(bundled_dir=bundled, user_dir=user, llm_router=router)
        ok = loader.synthesize("bad_organ", "do bad things")
        assert not ok
        assert "bad_organ" not in loader.known_intents()
        assert not (user / "bad_organ.py").exists()


def test_synthesize_fails_without_router():
    with tempfile.TemporaryDirectory() as d:
        bundled = Path(d) / "bundled"
        user    = Path(d) / "user"
        bundled.mkdir(); user.mkdir()
        loader = OrganLoader(bundled_dir=bundled, user_dir=user, llm_router=None)
        ok = loader.synthesize("something", "do something")
        assert not ok


def test_synthesize_fails_on_bad_json():
    with tempfile.TemporaryDirectory() as d:
        bundled = Path(d) / "bundled"
        user    = Path(d) / "user"
        bundled.mkdir(); user.mkdir()
        router = MagicMock()
        router.call.return_value = ("this is not json", {})
        loader = OrganLoader(bundled_dir=bundled, user_dir=user, llm_router=router)
        ok = loader.synthesize("something", "do something")
        assert not ok


def test_synthesize_fails_on_missing_execute():
    with tempfile.TemporaryDirectory() as d:
        bundled = Path(d) / "bundled"
        user    = Path(d) / "user"
        bundled.mkdir(); user.mkdir()
        router = _make_router({
            "intent": "broken",
            "description": "broken",
            "code": "ORGAN_META = {}\n\ndef not_execute(): pass",
        })
        loader = OrganLoader(bundled_dir=bundled, user_dir=user, llm_router=router)
        ok = loader.synthesize("broken", "do it")
        assert not ok


def test_synthesize_fails_on_llm_error():
    with tempfile.TemporaryDirectory() as d:
        bundled = Path(d) / "bundled"
        user    = Path(d) / "user"
        bundled.mkdir(); user.mkdir()
        router = MagicMock()
        router.call.side_effect = RuntimeError("LLM down")
        loader = OrganLoader(bundled_dir=bundled, user_dir=user, llm_router=router)
        ok = loader.synthesize("something", "task")
        assert not ok


# ── LOGIC_REGISTRY injection ──────────────────────────────────────────────────

def test_new_organ_added_to_logic_registry():
    """Loaded organ that isn't already in LOGIC_REGISTRY gets injected."""
    from prism_composer import LOGIC_REGISTRY
    with tempfile.TemporaryDirectory() as d:
        bundled = Path(d) / "bundled"
        user    = Path(d) / "user"
        bundled.mkdir(); user.mkdir()
        # Use an intent name unlikely to already exist
        unique_intent = "zzztest_unique_organ_xyz"
        code = VALID_ORGAN.replace("test_organ", unique_intent)
        _write_organ(bundled, f"{unique_intent}.py", code)
        loader = OrganLoader(bundled_dir=bundled, user_dir=user)
        assert unique_intent in loader.known_intents()
        assert unique_intent in LOGIC_REGISTRY
        # Cleanup
        del LOGIC_REGISTRY[unique_intent]


# ── Bundled organs smoke tests ────────────────────────────────────────────────

def test_bundled_weather_check_loads():
    """The shipped weather_check organ loads without error."""
    loader = OrganLoader()  # uses real bundled dir
    assert "weather_check" in loader.known_intents()


def test_bundled_currency_convert_loads():
    loader = OrganLoader()
    assert "currency_convert" in loader.known_intents()


def test_bundled_organ_execute_is_callable():
    loader = OrganLoader()
    fn = loader.get("weather_check")
    assert callable(fn)
