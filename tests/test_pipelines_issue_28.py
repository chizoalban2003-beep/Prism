"""
tests/test_pipelines_issue_28.py
================================
Persistent pipelines (#28-116, gap 2): store CRUD, NL parsing of
"save pipeline <name>: <steps> [every N ...]", scheduling, and the
handler/agent wiring that runs a pipeline through the tool loop.

Relies on conftest's hermetic HOME: ~/.prism/pipelines.db is throwaway.
"""
from __future__ import annotations

from prism_intents import INTENTS
from prism_pipelines import PipelineStore, human_schedule, parse_save
from prism_responses import text_card
from prism_routing import route_intent


def _route(m):
    return route_intent(m, INTENTS, lambda _m: None)


class TestParse:
    def test_basic_name_and_steps(self):
        name, instr, secs = parse_save(
            "save pipeline morning: check the weather and my calendar")
        assert name == "morning"
        assert "weather" in instr and "calendar" in instr
        assert secs == 0

    def test_every_n_minutes(self):
        _, _, secs = parse_save(
            "save pipeline watch: check bitcoin price every 15 minutes")
        assert secs == 15 * 60

    def test_daily_adverb(self):
        _, instr, secs = parse_save(
            "create routine brief: summarise my inbox every day")
        assert secs == 86400
        assert "every day" not in instr.lower()

    def test_missing_colon_raises(self):
        try:
            parse_save("save pipeline broken no separator")
            assert False, "expected ValueError"
        except ValueError:
            pass

    def test_human_schedule(self):
        assert human_schedule(0) == "manual"
        assert human_schedule(3600) == "every 1h"
        assert human_schedule(900) == "every 15m"
        assert human_schedule(172800) == "every 2d"


class TestStore:
    def test_save_get_list_delete(self, tmp_path):
        s = PipelineStore(tmp_path / "p.db")
        s.save("Morning Brief", "check weather", 3600)
        p = s.get("morning brief")   # name is normalised lower
        assert p is not None
        assert p.schedule_secs == 3600
        assert [x.name for x in s.list_all()] == ["morning brief"]
        assert s.delete("morning brief") is True
        assert s.get("morning brief") is None

    def test_save_replaces_and_preserves_created_at(self, tmp_path):
        s = PipelineStore(tmp_path / "p.db")
        first = s.save("x", "step one")
        second = s.save("x", "step two")
        assert second.instruction == "step two"
        assert second.created_at == first.created_at
        assert len(s.list_all()) == 1

    def test_due_only_scheduled_and_elapsed(self, tmp_path):
        s = PipelineStore(tmp_path / "p.db")
        s.save("manual", "do it")                 # no schedule
        s.save("fast", "poll", 1)                 # scheduled
        import time
        time.sleep(1.05)
        due = [p.name for p in s.due()]
        assert "fast" in due and "manual" not in due

    def test_empty_name_or_instruction_rejected(self, tmp_path):
        s = PipelineStore(tmp_path / "p.db")
        for args in [("", "steps"), ("name", "")]:
            try:
                s.save(*args)
                assert False
            except ValueError:
                pass


class TestRouting:
    def test_intents(self):
        assert _route("save pipeline morning: do things") == "pipeline_save"
        assert _route("run pipeline morning") == "pipeline_run"
        assert _route("list my pipelines") == "pipeline_list"
        assert _route("delete pipeline morning") == "pipeline_delete"

    def test_does_not_steal_ordinary_tasks(self):
        assert _route("add task buy milk") == "add_task"
        assert _route("list my tasks") == "list_tasks"


class TestChatGuards:
    """chat()-level: a control-plane command must not be intercepted by
    task-oriented pre-processing just because its saved instruction mentions
    an unconfigured service (calendar/email) or reads as multi-step."""

    def _agent(self):
        from prism_agent import PrismAgent
        return PrismAgent()

    def test_calendar_mention_in_pipeline_does_not_raise_setup_card(
            self, offline_llm):
        agent = self._agent()
        # calendar is unconfigured in the hermetic HOME; the mention here is
        # part of the instruction, not a live request — must still save.
        card = agent.chat(
            "save pipeline dawn: check the weather and my calendar")
        assert card.title == "Pipeline saved", card.title
        assert agent._pipelines.get("dawn") is not None

    def test_and_in_instruction_does_not_fold_into_the_loop(self, offline_llm):
        agent = self._agent()
        card = agent.chat(
            "save pipeline umbrella: check the weather and tell me to pack")
        assert card.title == "Pipeline saved", card.title


class TestHandlerWiring:
    def _agent(self, tmp_path, monkeypatch):
        import types

        from prism_pa_intents import handle_pa_intent
        ran = {}

        agent = types.SimpleNamespace()
        agent._pipelines = PipelineStore(tmp_path / "p.db")

        def run_pipeline(instruction, ctx=None):
            ran["instruction"] = instruction
            return text_card("pipeline output", "Done")
        agent.run_pipeline = run_pipeline
        return agent, handle_pa_intent, ran

    def test_save_then_run_invokes_agent_pipeline(self, tmp_path, monkeypatch):
        agent, handle, ran = self._agent(tmp_path, monkeypatch)
        c1 = handle(agent, "pipeline_save",
                    "save pipeline brief: check the weather then note it", {})
        assert "Saved pipeline" in c1.title or "brief" in c1.body.lower()

        c2 = handle(agent, "pipeline_run", "run pipeline brief", {})
        assert c2.body == "pipeline output"
        assert "weather" in ran["instruction"]
        assert agent._pipelines.get("brief").run_count == 1

    def test_run_unknown_pipeline_is_honest(self, tmp_path, monkeypatch):
        agent, handle, _ = self._agent(tmp_path, monkeypatch)
        c = handle(agent, "pipeline_run", "run pipeline nonexistent", {})
        assert "No pipeline named" in c.body

    def test_list_empty_and_populated(self, tmp_path, monkeypatch):
        agent, handle, _ = self._agent(tmp_path, monkeypatch)
        assert "No pipelines yet" in handle(
            agent, "pipeline_list", "list pipelines", {}).body
        handle(agent, "pipeline_save",
               "save pipeline a: do a thing every 2 hours", {})
        listed = handle(agent, "pipeline_list", "list pipelines", {}).body
        assert "**a**" in listed and "every 2h" in listed
