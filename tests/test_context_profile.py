"""Tests for prism_context_profile.ContextManager"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

from prism_context_profile import BUILTIN_CONTEXTS, ContextManager, ContextProfile


def _manager() -> tuple[ContextManager, Path]:
    d = tempfile.mkdtemp()
    p = Path(d) / "contexts.json"
    return ContextManager(db_path=str(p)), p


# ── Defaults ──────────────────────────────────────────────────────────────────

def test_default_context_exists():
    cm, _ = _manager()
    assert cm.active_id == "default"
    profile = cm.active()
    assert profile.context_id == "default"


def test_default_context_has_empty_overrides():
    cm, _ = _manager()
    assert cm.active().policy_overrides == {}


# ── create() ─────────────────────────────────────────────────────────────────

def test_create_new_context():
    cm, _ = _manager()
    p = cm.create("work", description="Work mode")
    assert p.context_id == "work"
    assert p.description == "Work mode"


def test_create_with_policy_overrides():
    cm, _ = _manager()
    p = cm.create("focus", policy_overrides={"email_send": False, "send_push": False})
    assert p.policy_overrides["email_send"] is False


def test_create_with_organ_priorities():
    cm, _ = _manager()
    p = cm.create("personal", organ_priorities={"health_summary": 10})
    assert p.organ_priorities["health_summary"] == 10


# ── switch() ─────────────────────────────────────────────────────────────────

def test_switch_to_existing():
    cm, _ = _manager()
    cm.create("work")
    cm.switch("work")
    assert cm.active_id == "work"


def test_switch_to_builtin_creates_profile():
    cm, _ = _manager()
    cm.switch("work")  # not explicitly created — should load from BUILTIN_CONTEXTS
    assert "work" in [p.context_id for p in cm.list_profiles()]


def test_switch_to_unknown_raises():
    cm, _ = _manager()
    try:
        cm.switch("nonexistent_context_xyz")
        assert False, "should have raised"
    except ValueError:
        pass


def test_switch_persists_to_file():
    cm, path = _manager()
    cm.create("work")
    cm.switch("work")
    data = json.loads(path.read_text())
    assert data["active_id"] == "work"


# ── list_profiles() ──────────────────────────────────────────────────────────

def test_list_profiles_includes_default():
    cm, _ = _manager()
    ids = [p.context_id for p in cm.list_profiles()]
    assert "default" in ids


def test_list_profiles_after_create():
    cm, _ = _manager()
    cm.create("work")
    cm.create("personal")
    ids = [p.context_id for p in cm.list_profiles()]
    assert "work" in ids
    assert "personal" in ids


# ── delete() ─────────────────────────────────────────────────────────────────

def test_delete_custom_context():
    cm, _ = _manager()
    cm.create("tmp")
    assert cm.delete("tmp")
    assert "tmp" not in [p.context_id for p in cm.list_profiles()]


def test_delete_default_context_refused():
    cm, _ = _manager()
    assert cm.delete("default") is False


def test_delete_active_falls_back_to_default():
    cm, _ = _manager()
    cm.create("tmp")
    cm.switch("tmp")
    cm.delete("tmp")
    assert cm.active_id == "default"


# ── persistence across instances ─────────────────────────────────────────────

def test_persists_and_reloads():
    d = tempfile.mkdtemp()
    p = Path(d) / "ctx.json"
    cm1 = ContextManager(db_path=str(p))
    cm1.create("work", description="Work mode")
    cm1.switch("work")

    cm2 = ContextManager(db_path=str(p))
    assert cm2.active_id == "work"
    assert "work" in [pr.context_id for pr in cm2.list_profiles()]


# ── inject helpers ────────────────────────────────────────────────────────────

def test_inject_into_chain_sets_context_id():
    cm, _ = _manager()
    cm.create("focus")
    cm.switch("focus")
    chain = MagicMock()
    chain._context_id = "default"
    cm.inject_into_chain(chain)
    assert chain._context_id == "focus"


def test_inject_into_chain_ctx_adds_keys():
    cm, _ = _manager()
    ctx = {}
    cm.inject_into_chain_ctx(ctx)
    assert "context_id" in ctx
    assert "organ_priorities" in ctx


def test_apply_to_policy_calls_set_allowance():
    cm, _ = _manager()
    cm.create("focus", policy_overrides={"email_send": False})
    cm.switch("focus")
    policy = MagicMock()
    policy.set_allowance = MagicMock()
    cm.apply_to_policy(policy)
    policy.set_allowance.assert_called_with("email_send", False)


# ── ContextProfile serialisation ─────────────────────────────────────────────

def test_context_profile_round_trip():
    p = ContextProfile(
        context_id       = "test",
        description      = "Test context",
        soul_lens_ids    = ["lens1"],
        policy_overrides = {"email_send": True},
        organ_priorities = {"health_summary": 5},
    )
    d = p.to_dict()
    p2 = ContextProfile.from_dict(d)
    assert p2.context_id == "test"
    assert p2.soul_lens_ids == ["lens1"]
    assert p2.policy_overrides["email_send"] is True


# ── builtin contexts ──────────────────────────────────────────────────────────

def test_builtin_work_has_email_allowed():
    assert BUILTIN_CONTEXTS["work"]["policy_overrides"]["email_send"] is True


def test_builtin_focus_has_email_denied():
    assert BUILTIN_CONTEXTS["focus"]["policy_overrides"]["email_send"] is False


def test_builtin_focus_has_push_denied():
    assert BUILTIN_CONTEXTS["focus"]["policy_overrides"]["send_push"] is False
