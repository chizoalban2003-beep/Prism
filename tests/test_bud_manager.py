"""Tests for BudManager — filling the gap identified in the full repo audit."""
from unittest.mock import MagicMock

import pytest

from prism_bud_manager import BudHandle, BudManager, BudStatus, _scoped_ctx

# ── _scoped_ctx helper ────────────────────────────────────────────────────────

class TestScopedCtx:
    def test_always_keys_always_included(self):
        full = {"organ_loader": "ol", "router": "r", "secret_cred": "x"}
        scoped = _scoped_ctx(full, [])
        assert "organ_loader" in scoped
        assert "router" in scoped

    def test_non_allowed_key_excluded(self):
        full = {"organ_loader": "ol", "secret_cred": "x"}
        scoped = _scoped_ctx(full, [])
        assert "secret_cred" not in scoped

    def test_capability_unlocks_extra_keys(self):
        full = {"twilio_config": "tc", "organ_loader": "ol"}
        scoped = _scoped_ctx(full, ["telephony"])
        assert "twilio_config" in scoped

    def test_approval_flags_always_pass_through(self):
        full = {"_approved_shell_run": True, "organ_loader": "ol"}
        scoped = _scoped_ctx(full, [])
        assert "_approved_shell_run" in scoped

    def test_internet_write_grants_mail_handles(self):
        # email/calendar/contacts/router-style mail handles are granted to
        # internet_write organs, not to everyone.
        full = {"email": "E", "calendar": "C", "contacts": "K", "organ_loader": "ol"}
        scoped = _scoped_ctx(full, ["internet_write"])
        assert scoped.get("email") == "E"
        assert scoped.get("calendar") == "C"
        assert scoped.get("contacts") == "K"

    def test_mail_handles_not_in_always(self):
        # A read-only / unrelated organ must NOT receive the mailbox/calendar.
        full = {"email": "E", "calendar": "C", "contacts": "K", "organ_loader": "ol"}
        scoped = _scoped_ctx(full, ["internet_read"])
        assert "email" not in scoped
        assert "calendar" not in scoped

    def test_sensitive_context_not_leaked_to_organs(self):
        # Conversation history / persona / recalled memory are never exposed.
        full = {
            "organ_loader": "ol", "history": [1], "persona_context": "p",
            "memory_context": [{}], "perception": {}, "standing_instructions": "x",
        }
        scoped = _scoped_ctx(full, [])
        for leaked in ("history", "persona_context", "memory_context",
                       "perception", "standing_instructions"):
            assert leaked not in scoped

    def test_telephony_cap_does_not_leak_shell_runner(self):
        full = {"shell_runner": "sr", "twilio_config": "tc", "organ_loader": "ol"}
        scoped = _scoped_ctx(full, ["telephony"])
        assert "shell_runner" not in scoped
        assert "twilio_config" in scoped

    def test_empty_capabilities_only_always_keys(self):
        full = {"organ_loader": "ol", "shell_runner": "sr", "twilio_config": "tc"}
        scoped = _scoped_ctx(full, [])
        assert "shell_runner" not in scoped
        assert "twilio_config" not in scoped


# ── BudManager spawn ──────────────────────────────────────────────────────────

class TestBudManagerSpawn:
    def setup_method(self):
        self.mgr = BudManager()

    def test_spawn_returns_handle(self):
        h = self.mgr.spawn("web_search", "find X", {}, ["internet_read"])
        assert isinstance(h, BudHandle)

    def test_spawned_handle_pending_status(self):
        h = self.mgr.spawn("web_search", "find X", {}, [])
        assert h.status == BudStatus.PENDING

    def test_spawn_injects_bud_id_into_scoped_ctx(self):
        h = self.mgr.spawn("web_search", "q", {}, [])
        assert "_bud_id" in h.scoped_ctx
        assert h.scoped_ctx["_bud_id"] == h.bud_id

    def test_spawn_injects_capabilities(self):
        h = self.mgr.spawn("phone_call", "call mom", {}, ["telephony"])
        assert h.scoped_ctx["_bud_capabilities"] == ["telephony"]

    def test_active_count_increments(self):
        self.mgr.spawn("a", "m", {}, [])
        self.mgr.spawn("b", "m", {}, [])
        assert self.mgr.active_count() == 2

    def test_session_count_increments(self):
        self.mgr.spawn("a", "m", {}, [])
        self.mgr.spawn("b", "m", {}, [])
        assert self.mgr.session_stats()["total_spawned"] == 2

    def test_unique_bud_ids(self):
        handles = [self.mgr.spawn("a", "m", {}, []) for _ in range(10)]
        ids = {h.bud_id for h in handles}
        assert len(ids) == 10


# ── BudManager execute ────────────────────────────────────────────────────────

class TestBudManagerExecute:
    def setup_method(self):
        self.mgr = BudManager()

    def test_execute_calls_organ_fn(self):
        organ = MagicMock(return_value="result_card")
        h = self.mgr.spawn("web_search", "q", {}, [])
        result = self.mgr.execute(h, organ)
        organ.assert_called_once()
        assert result == "result_card"

    def test_execute_passes_intent_message_ctx(self):
        captured = {}
        def organ(intent, msg, ctx):
            captured["intent"] = intent
            captured["msg"] = msg
            captured["ctx"] = ctx
            return "card"
        full_ctx = {"organ_loader": "ol"}
        h = self.mgr.spawn("weather_check", "London weather", full_ctx, [])
        self.mgr.execute(h, organ)
        assert captured["intent"] == "weather_check"
        assert captured["msg"] == "London weather"

    def test_execute_sets_completed_status(self):
        h = self.mgr.spawn("a", "m", {}, [])
        self.mgr.execute(h, lambda i, m, c: None)
        assert h.status == BudStatus.COMPLETED

    def test_execute_decommissions_on_completion(self):
        h = self.mgr.spawn("a", "m", {}, [])
        self.mgr.execute(h, lambda i, m, c: None)
        assert self.mgr.active_count() == 0

    def test_execute_sets_failed_status_on_error(self):
        def bad_organ(i, m, c):
            raise ValueError("organ blew up")
        h = self.mgr.spawn("a", "m", {}, [])
        with pytest.raises(ValueError):
            self.mgr.execute(h, bad_organ)
        assert h.status == BudStatus.FAILED
        assert "organ blew up" in h.error

    def test_execute_decommissions_on_error(self):
        def bad_organ(i, m, c):
            raise RuntimeError("oops")
        h = self.mgr.spawn("a", "m", {}, [])
        with pytest.raises(RuntimeError):
            self.mgr.execute(h, bad_organ)
        assert self.mgr.active_count() == 0

    def test_elapsed_ms_recorded(self):
        h = self.mgr.spawn("a", "m", {}, [])
        self.mgr.execute(h, lambda i, m, c: None)
        assert h.elapsed_ms >= 0


# ── Decommission ──────────────────────────────────────────────────────────────

class TestDecommission:
    def setup_method(self):
        self.mgr = BudManager()

    def test_decommission_clears_bud_id_from_ctx(self):
        h = self.mgr.spawn("a", "m", {}, [])
        self.mgr.decommission(h)
        assert "_bud_id" not in h.scoped_ctx

    def test_decommission_clears_capabilities_from_ctx(self):
        h = self.mgr.spawn("a", "m", {}, ["internet_read"])
        self.mgr.decommission(h)
        assert "_bud_capabilities" not in h.scoped_ctx

    def test_decommission_idempotent(self):
        h = self.mgr.spawn("a", "m", {}, [])
        self.mgr.decommission(h)
        self.mgr.decommission(h)  # must not raise
        assert self.mgr.active_count() == 0

    def test_active_count(self):
        h1 = self.mgr.spawn("a", "m", {}, [])
        self.mgr.spawn("b", "m", {}, [])
        assert self.mgr.active_count() == 2
        self.mgr.decommission(h1)
        assert self.mgr.active_count() == 1


# ── Per-session rate limits (L2 max_per_session / L1 organ ceiling) ───────────

class TestSessionRateLimits:
    def setup_method(self):
        self.mgr = BudManager()

    def test_session_intent_count_starts_zero(self):
        assert self.mgr.session_intent_count("email_send") == 0

    def test_session_intent_count_increments_per_intent(self):
        self.mgr.spawn("email_send", "m", {}, ["internet_write"])
        self.mgr.spawn("email_send", "m", {}, ["internet_write"])
        self.mgr.spawn("web_search", "m", {}, ["internet_read"])
        assert self.mgr.session_intent_count("email_send") == 2
        assert self.mgr.session_intent_count("web_search") == 1

    def test_session_organ_total_counts_all_intents(self):
        self.mgr.spawn("a", "m", {}, [])
        self.mgr.spawn("b", "m", {}, [])
        assert self.mgr.session_organ_total() == 2

    def test_organ_budget_not_exceeded_without_guard(self):
        assert self.mgr.organ_budget_exceeded() is False

    def test_organ_budget_exceeded_at_ceiling(self):
        guard = MagicMock()
        guard.max_organs_per_session.return_value = 2
        mgr = BudManager(constitution_guard=guard)
        assert mgr.organ_budget_exceeded() is False
        mgr.spawn("a", "m", {}, [])
        mgr.spawn("b", "m", {}, [])
        assert mgr.organ_budget_exceeded() is True


# ── Synthesis limits ──────────────────────────────────────────────────────────

class TestSynthesisLimits:
    def test_synthesis_allowed_without_guard(self):
        mgr = BudManager(constitution_guard=None)
        assert mgr.synthesis_allowed()

    def test_synthesis_allowed_below_limit(self):
        guard = MagicMock()
        guard.max_synthesis_per_session.return_value = 5
        mgr = BudManager(constitution_guard=guard)
        for _ in range(4):
            mgr.record_synthesis()
        assert mgr.synthesis_allowed()

    def test_synthesis_blocked_at_limit(self):
        guard = MagicMock()
        guard.max_synthesis_per_session.return_value = 3
        mgr = BudManager(constitution_guard=guard)
        for _ in range(3):
            mgr.record_synthesis()
        assert not mgr.synthesis_allowed()

    def test_record_synthesis_increments_counter(self):
        mgr = BudManager()
        mgr.record_synthesis()
        mgr.record_synthesis()
        assert mgr.session_stats()["synthesis_this_session"] == 2

    def test_session_stats_structure(self):
        mgr = BudManager()
        s = mgr.session_stats()
        assert "total_spawned" in s
        assert "currently_active" in s
        assert "synthesis_this_session" in s
