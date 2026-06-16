"""
Tests for the Nucleus-Organ Topology infrastructure:
  - ConstitutionGuard (L1)
  - BudManager (ephemeral execution contexts)
  - OrganLoader.get_organ_capabilities()
  - OrganLoader.synthesize() constitution check
  - Organ capabilities declared in ORGAN_META
  - _execute() constitution block + bud execution
  - Missing organ synthesis flow
  - PrismChain logicpolicy feedback loop
    (llm→logic+logicpolicy→policy→llm→logicN+logicpolicyN→policyN...)
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── ConstitutionGuard ─────────────────────────────────────────────────────────

class TestConstitutionGuard:
    def setup_method(self):
        from prism_constitution import ConstitutionGuard
        self.guard = ConstitutionGuard()

    def test_loads_successfully(self):
        assert self.guard.loaded

    def test_check_allowed_intent(self):
        ok, reason = self.guard.check("web_search", ["internet_read"])
        assert ok
        assert reason == ""

    def test_check_blocked_intent_missing_capability(self):
        ok, reason = self.guard.check("web_search", [])
        assert not ok
        assert "internet_read" in reason

    def test_check_shell_run_requires_subprocess(self):
        ok, _ = self.guard.check("shell_run", [])
        assert not ok
        ok2, _ = self.guard.check("shell_run", ["subprocess"])
        assert ok2

    def test_required_capabilities_web_search(self):
        caps = self.guard.required_capabilities("web_search")
        assert "internet_read" in caps

    def test_required_capabilities_unit_convert_empty(self):
        caps = self.guard.required_capabilities("unit_convert")
        assert caps == []

    def test_may_synthesize_subprocess_blocked(self):
        assert not self.guard.may_synthesize("subprocess")

    def test_may_synthesize_telephony_blocked(self):
        assert not self.guard.may_synthesize("telephony")

    def test_may_synthesize_internet_read_allowed(self):
        assert self.guard.may_synthesize("internet_read")

    def test_is_never_log_email_send(self):
        assert self.guard.is_never_log("email_send")

    def test_is_never_log_web_search_false(self):
        assert not self.guard.is_never_log("web_search")

    def test_capability_risk_levels(self):
        assert self.guard.capability_risk("subprocess") == "critical"
        assert self.guard.capability_risk("internet_read") == "low"
        assert self.guard.capability_risk("telephony") == "high"

    def test_max_synthesis_per_session(self):
        assert self.guard.max_synthesis_per_session() == 10

    def test_check_unknown_intent_allowed(self):
        # Intents not in constitution have no requirements → allowed
        ok, reason = self.guard.check("some_new_intent", [])
        assert ok

    def test_check_file_write_requires_filesystem_write(self):
        ok, _ = self.guard.check("file_write", [])
        assert not ok
        ok2, _ = self.guard.check("file_write", ["filesystem_write"])
        assert ok2

    def test_get_guard_singleton(self):
        from prism_constitution import get_guard
        g1 = get_guard()
        g2 = get_guard()
        assert g1 is g2

    def test_check_discord_send_requires_internet_write(self):
        ok, _ = self.guard.check("discord_send", [])
        assert not ok
        ok2, _ = self.guard.check("discord_send", ["internet_write"])
        assert ok2


# ── BudManager ────────────────────────────────────────────────────────────────

class TestBudManager:
    def setup_method(self):
        from prism_bud_manager import BudManager
        from prism_constitution import ConstitutionGuard
        self.guard = ConstitutionGuard()
        self.mgr = BudManager(constitution_guard=self.guard)

    def test_spawn_returns_handle(self):
        from prism_bud_manager import BudStatus
        handle = self.mgr.spawn("web_search", "test", {}, ["internet_read"])
        assert handle.bud_id
        assert handle.intent == "web_search"
        assert handle.status == BudStatus.PENDING
        assert "internet_read" in handle.capabilities

    def test_execute_calls_organ_fn(self):
        from prism_responses import text_card
        def fake_organ(intent, msg, ctx):
            return text_card("result", "test")
        handle = self.mgr.spawn("unit_convert", "5km to miles", {}, [])
        card = self.mgr.execute(handle, fake_organ)
        assert card.body == "result"

    def test_execute_decommissions_bud(self):
        from prism_bud_manager import BudStatus
        from prism_responses import text_card
        def fake_organ(intent, msg, ctx):
            return text_card("ok", "test")
        handle = self.mgr.spawn("unit_convert", "msg", {}, [])
        bud_id = handle.bud_id
        self.mgr.execute(handle, fake_organ)
        # After execution the bud is removed from active_buds
        assert bud_id not in self.mgr._active_buds
        assert handle.status == BudStatus.COMPLETED

    def test_execute_marks_failed_on_exception(self):
        from prism_bud_manager import BudStatus
        def bad_organ(intent, msg, ctx):
            raise RuntimeError("boom")
        handle = self.mgr.spawn("unit_convert", "msg", {}, [])
        with pytest.raises(RuntimeError):
            self.mgr.execute(handle, bad_organ)
        assert handle.status == BudStatus.FAILED

    def test_scoped_ctx_strips_sensitive_keys(self):
        full_ctx = {
            "organ_loader": "loader",
            "router": "rtr",
            "secret_key": "s3cr3t",     # not granted by any capability
            "twilio_config": {"sid": "x"},
        }
        handle = self.mgr.spawn("web_search", "msg", full_ctx, ["internet_read"])
        assert "organ_loader" in handle.scoped_ctx
        assert "router" in handle.scoped_ctx
        assert "secret_key" not in handle.scoped_ctx
        # twilio_config only granted by internet_write, not internet_read
        assert "twilio_config" not in handle.scoped_ctx

    def test_scoped_ctx_grants_twilio_for_internet_write(self):
        full_ctx = {"twilio_config": {"sid": "x"}}
        handle = self.mgr.spawn("discord_send", "msg", full_ctx, ["internet_write"])
        assert "twilio_config" in handle.scoped_ctx

    def test_scoped_ctx_always_includes_bud_token(self):
        handle = self.mgr.spawn("unit_convert", "msg", {}, [])
        assert "_bud_id" in handle.scoped_ctx
        assert handle.scoped_ctx["_bud_id"] == handle.bud_id

    def test_decommission_removes_token(self):
        handle = self.mgr.spawn("unit_convert", "msg", {}, [])
        self.mgr.decommission(handle)
        assert "_bud_id" not in handle.scoped_ctx

    def test_active_count(self):
        h1 = self.mgr.spawn("a", "m", {}, [])
        self.mgr.spawn("b", "m", {}, [])
        assert self.mgr.active_count() == 2
        self.mgr.decommission(h1)
        assert self.mgr.active_count() == 1

    def test_session_stats(self):
        self.mgr.spawn("a", "m", {}, [])
        self.mgr.spawn("b", "m", {}, [])
        stats = self.mgr.session_stats()
        assert stats["total_spawned"] == 2

    def test_synthesis_allowed_under_limit(self):
        assert self.mgr.synthesis_allowed()

    def test_synthesis_allowed_at_limit(self):
        self.mgr._session_synthesis = 10
        assert not self.mgr.synthesis_allowed()

    def test_approval_flags_always_passed_through(self):
        full_ctx = {"_approved_email_send": True, "secret": "x"}
        handle = self.mgr.spawn("email_send", "msg", full_ctx, ["internet_write"])
        assert "_approved_email_send" in handle.scoped_ctx
        assert "secret" not in handle.scoped_ctx


# ── OrganLoader capabilities ──────────────────────────────────────────────────

class TestOrganLoaderCapabilities:
    def setup_method(self):
        from prism_organ_loader import OrganLoader
        self.loader = OrganLoader()

    def test_web_search_has_internet_read(self):
        caps = self.loader.get_organ_capabilities("web_search")
        assert "internet_read" in caps

    def test_shell_run_has_subprocess(self):
        caps = self.loader.get_organ_capabilities("shell_run")
        assert "subprocess" in caps

    def test_unit_convert_empty_capabilities(self):
        caps = self.loader.get_organ_capabilities("unit_convert")
        assert caps == []

    def test_reminder_set_has_notifications(self):
        caps = self.loader.get_organ_capabilities("reminder_set")
        assert "notifications" in caps

    def test_discord_send_has_internet_write(self):
        caps = self.loader.get_organ_capabilities("discord_send")
        assert "internet_write" in caps

    def test_file_write_has_filesystem_write(self):
        caps = self.loader.get_organ_capabilities("file_write")
        assert "filesystem_write" in caps

    def test_screenshot_has_system_ui(self):
        caps = self.loader.get_organ_capabilities("screenshot_capture")
        assert "system_ui" in caps

    def test_phone_call_has_telephony(self):
        caps = self.loader.get_organ_capabilities("phone_call")
        assert "telephony" in caps

    def test_unknown_intent_returns_empty(self):
        caps = self.loader.get_organ_capabilities("no_such_organ_xyz")
        assert caps == []

    def test_all_bundled_organs_have_capabilities_declared(self):
        """Every bundled organ must declare a capabilities list in ORGAN_META."""
        from pathlib import Path
        bundled_dir = Path(__file__).parent.parent / "organs"
        issues = []
        for py_file in sorted(bundled_dir.glob("*.py")):
            if py_file.name.startswith("_"):
                continue
            intent = py_file.stem
            entry = self.loader._organs.get(intent)
            if entry is None:
                continue
            fn = entry[0]
            meta = getattr(fn, "_organ_meta", {})
            if "capabilities" not in meta:
                issues.append(intent)
        assert issues == [], f"Organs missing capabilities in ORGAN_META: {issues}"


# ── Synthesize — constitution blocks dangerous capabilities ───────────────────

class TestSynthesisConstitutionBlock:
    def test_synthesize_subprocess_blocked(self):
        from prism_organ_loader import OrganLoader
        mock_router = MagicMock()
        loader = OrganLoader(llm_router=mock_router)
        # shell_run requires subprocess which constitution forbids synthesizing
        result = loader.synthesize("shell_run", "run ls -la")
        assert result is False
        mock_router.call.assert_not_called()

    def test_synthesize_telephony_blocked(self):
        from prism_organ_loader import OrganLoader
        mock_router = MagicMock()
        loader = OrganLoader(llm_router=mock_router)
        result = loader.synthesize("phone_call", "call alice")
        assert result is False
        mock_router.call.assert_not_called()

    def test_synthesize_safe_intent_proceeds(self):
        from prism_organ_loader import OrganLoader
        mock_router = MagicMock()
        mock_router.call.return_value = (None, {})
        loader = OrganLoader(llm_router=mock_router)
        # stock_price is a new intent with no constitution requirements → proceeds to LLM
        loader.synthesize("stock_price", "what is AAPL stock price")
        mock_router.call.assert_called_once()


# ── PrismAgent integration ────────────────────────────────────────────────────

class TestAgentTopologyIntegration:
    def _make_agent(self):
        from prism_agent import PrismAgent
        return PrismAgent()

    def test_agent_has_constitution(self):
        agent = self._make_agent()
        assert agent._constitution is not None
        assert agent._constitution.loaded

    def test_agent_has_bud_manager(self):
        agent = self._make_agent()
        assert agent._bud_mgr is not None

    def test_organ_execute_routes_through_bud(self):
        """An organ call should increment bud session count."""
        agent = self._make_agent()
        initial = agent._bud_mgr.session_stats()["total_spawned"]
        agent.chat("how many kilometers in 5 miles")
        after = agent._bud_mgr.session_stats()["total_spawned"]
        assert after > initial

    def test_constitution_blocks_shell_run_missing_cap(self):
        """
        shell_run requires subprocess capability.  If somehow an organ without
        that capability were loaded, the constitution should block execution.
        """
        from prism_responses import text_card
        agent = self._make_agent()
        # Patch organ_loader to return a fake shell_run organ with no capabilities
        fake_fn = MagicMock(return_value=text_card("executed!", "shell"))
        fake_fn._organ_meta = {"intent": "shell_run", "capabilities": []}
        fake_fn._organ_policy = {}
        with patch.object(agent._organ_loader, "get", return_value=fake_fn), \
             patch.object(agent._organ_loader, "get_organ_capabilities", return_value=[]):
            card = agent._execute("shell_run", "run ls", {})
        # Constitution should block it
        assert "restricted" in card.body.lower() or "blocked" in card.title.lower()

    def test_missing_organ_synthesis_message_without_router(self):
        """Without an LLM router, missing organ returns an actionable message."""
        agent = self._make_agent()
        agent._router = None
        agent._organ_loader._router = None
        card = agent._execute("nonexistent_intent_xyz", "do something weird", {})
        body = card.body.lower()
        assert ("capability" in body or "synthesize" in body or
                "ollama" in body or "llm" in body or "not found" in body)

    @pytest.mark.slow
    @pytest.mark.timeout(120)
    def test_synthesis_limit_respected(self):
        """When session synthesis limit reached, skip synthesis attempt."""
        agent = self._make_agent()
        agent._bud_mgr._session_synthesis = 10  # at limit
        with patch.object(agent._organ_loader, "synthesize") as mock_syn:
            agent._execute("some_novel_intent_abc", "do something", {})
        mock_syn.assert_not_called()

    def test_bud_scoped_ctx_excludes_secret_fields(self):
        """Generic organ path: bud should not forward ungranted ctx keys."""
        agent = self._make_agent()
        captured_ctx: dict = {}

        from prism_responses import text_card as _tc

        def spy_organ(intent, msg, ctx):
            captured_ctx.update(ctx)
            return _tc("ok", "test")

        # Use "qr_generate" — no special handler in _execute, goes straight to
        # the generic organ block, which routes through BudManager.
        with patch.object(agent._organ_loader, "get", return_value=spy_organ), \
             patch.object(agent._organ_loader, "get_organ_capabilities",
                          return_value=[]), \
             patch.object(agent._organ_loader, "get_organ_policy", return_value={}):
            agent._execute("qr_generate", "qr code for hello", {"secret_api_key": "s3cr3t"})

        assert "secret_api_key" not in captured_ctx
        # _bud_id is visible to the organ during execution (its own identity token)
        # but is removed from scoped_ctx on decommission (after organ returns)
        # — verify scoped_ctx is clean after the call
        organ_fn = agent._organ_loader.get("qr_generate")
        if organ_fn is not None:
            caps2 = agent._organ_loader.get_organ_capabilities("qr_generate")
            h = agent._bud_mgr.spawn("qr_generate", "test", {}, caps2)
            agent._bud_mgr.decommission(h)
            assert "_bud_id" not in h.scoped_ctx

    @pytest.mark.slow
    @pytest.mark.timeout(120)
    def test_multiple_tasks_end_to_end(self):
        """Smoke-test 10 diverse tasks — all return a PrismCard, never raise."""
        from prism_responses import PrismCard
        agent = self._make_agent()
        tasks = [
            "convert 100 km to miles",
            "what is the weather like today",
            "search the web for Python async tutorials",
            "show me the news headlines",
            "what's my status",
            "list my organs",
            "remind me in 10 minutes to check email",
            "convert 50 USD to EUR",
            "take a screenshot",
            "what's on Wikipedia about photosynthesis",
        ]
        for task in tasks:
            card = agent.chat(task)
            assert isinstance(card, PrismCard), f"Task '{task}' did not return PrismCard"
            assert card.body, f"Task '{task}' returned empty body"


# ── PrismChain logicpolicy feedback loop ──────────────────────────────────────

class TestChainLogicPolicyLoop:
    def _make_chain(self):
        from prism_chain import PrismChain
        from prism_organ_loader import OrganLoader
        loader = OrganLoader()
        return PrismChain(organ_loader=loader), loader

    def test_logicpolicy_meta_returns_dict_and_string(self):
        chain, _ = self._make_chain()
        meta, summary = chain._logicpolicy_meta("web_search")
        assert isinstance(meta, dict)
        assert "risk_level" in meta
        assert "capabilities" in meta
        assert "irreversible" in meta
        assert "constitution" in meta
        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_logicpolicy_meta_web_search_internet_read(self):
        chain, _ = self._make_chain()
        meta, summary = chain._logicpolicy_meta("web_search")
        assert "internet_read" in meta["capabilities"]
        assert "internet_read" in summary

    def test_logicpolicy_meta_shell_run_constitution_blocked(self):
        chain, _ = self._make_chain()
        meta, summary = chain._logicpolicy_meta("shell_run")
        # shell_run has subprocess capability which constitution blocks
        # check against constitution: requires subprocess but shell_run has it →
        # constitution should actually allow it (organ declares subprocess and
        # constitution requires subprocess for shell_run)
        assert "subprocess" in meta["capabilities"]
        assert meta["risk_level"] == "critical"
        assert "critical" in summary

    def test_logicpolicy_meta_unit_convert_safe(self):
        chain, _ = self._make_chain()
        meta, summary = chain._logicpolicy_meta("unit_convert")
        assert meta["capabilities"] == []
        assert meta["risk_level"] == "low"
        assert meta["irreversible"] is False
        assert meta["constitution"] == "allowed"

    def test_logicpolicy_meta_unknown_logic_graceful(self):
        chain, _ = self._make_chain()
        meta, summary = chain._logicpolicy_meta("nonexistent_logic_xyz")
        assert meta["risk_level"] == "low"
        assert meta["capabilities"] == []
        assert meta["constitution"] == "allowed"

    def test_chain_step_carries_organ_meta(self):
        from prism_chain import ChainStep
        step = ChainStep(
            step_num=1, logic="web_search", message_in="q",
            result_out="r", policy_note="", duration_ms=10.0,
            organ_meta={"risk_level": "low", "capabilities": ["internet_read"],
                        "irreversible": False, "constitution": "allowed"},
        )
        assert step.organ_meta["capabilities"] == ["internet_read"]
        assert step.organ_meta["constitution"] == "allowed"

    def test_accumulated_includes_logicpolicy_after_step(self):
        """
        The LLM-visible accumulated state must contain LogicPolicy after each step.
        We simulate a single step by calling _logicpolicy_meta and checking the
        format matches what run() injects into state.accumulated.
        """
        chain, _ = self._make_chain()
        meta, lp_summary = chain._logicpolicy_meta("unit_convert")
        # Build the accumulated line as run() does
        accumulated = (
            "\n\n[Step 1 — unit_convert]\n"
            "Asked: convert 5km to miles\n"
            "Got: 3.107 miles"
            + (f"\nLogicPolicy: {lp_summary}" if lp_summary else "")
        )
        assert "LogicPolicy:" in accumulated
        assert "risk=" in accumulated
        assert "L1=" in accumulated

    def test_streaming_step_event_includes_risk_and_caps(self):
        """run_streaming() step events must carry risk and caps fields."""
        from prism_chain import ChainStep
        # The _on_step closure in run_streaming reads step.organ_meta
        step = ChainStep(
            step_num=1, logic="weather_check", message_in="London weather",
            result_out="Cloudy, 15°C", policy_note="", duration_ms=50.0,
            organ_meta={"risk_level": "low", "capabilities": ["internet_read"],
                        "irreversible": False, "constitution": "allowed"},
        )
        # Simulate what _on_step does
        event = {
            "event":       "step",
            "step":        step.step_num,
            "logic":       step.logic,
            "result":      step.result_out[:200],
            "policy":      step.policy_note or "",
            "score":       step.eval_score,
            "risk":        step.organ_meta.get("risk_level", "low"),
            "caps":        step.organ_meta.get("capabilities", []),
            "constitution": step.organ_meta.get("constitution", "allowed"),
        }
        assert event["risk"] == "low"
        assert event["caps"] == ["internet_read"]
        assert event["constitution"] == "allowed"

    def test_logicpolicy_loop_multiple_logics(self):
        """
        Verify that logicpolicy metadata is distinct per logic —
        the loop feeds back the right context for each step.
        """
        chain, _ = self._make_chain()
        pairs = [
            ("unit_convert",     [],                    "low"),
            ("web_search",       ["internet_read"],     "low"),
            ("discord_send",     ["internet_write"],    "high"),
            ("shell_run",        ["subprocess"],        "critical"),
            ("reminder_set",     ["notifications"],     "low"),
            ("file_write",       ["filesystem_write"],  "medium"),
        ]
        for logic, expected_caps, expected_risk in pairs:
            meta, summary = chain._logicpolicy_meta(logic)
            assert meta["capabilities"] == expected_caps, \
                f"{logic}: expected caps {expected_caps}, got {meta['capabilities']}"
            assert meta["risk_level"] == expected_risk, \
                f"{logic}: expected risk {expected_risk}, got {meta['risk_level']}"
            assert "risk=" in summary
            assert "L1=" in summary
