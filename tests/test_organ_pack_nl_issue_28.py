"""
tests/test_organ_pack_nl_issue_28.py
====================================
Natural-language access to organ packs (Limit 3): export the user-synthesised
organs into a hash-verified pack and list packs/shareable organs from chat.
Import stays API/approval-gated by design. Uses hermetic HOME (conftest) so
~/.prism/packs is throwaway.
"""
from __future__ import annotations

import types
from pathlib import Path

from prism_intents import INTENTS
from prism_pa_intents import handle_pa_intent
from prism_routing import route_intent


def _route(m):
    return route_intent(m, INTENTS, lambda _m: "")


class _FakeLoader:
    """Minimal loader stub exposing the surface build_pack + the handler use."""

    def __init__(self, user_intents):
        self._user = set(user_intents)

    def list_organ_details(self):
        out = []
        for i in ("file_read", "web_search"):        # bundled
            out.append({"intent": i, "description": f"{i} desc",
                        "source": "bundled"})
        for i in sorted(self._user):                 # user-synthesised
            out.append({"intent": i, "description": f"{i} desc",
                        "source": "user"})
        return out

    def organ_source(self, intent):
        return (f"ORGAN_META = {{'intent': '{intent}'}}\n"
                "def execute(intent, message, ctx):\n    return None\n")

    def organ_details(self, intent):
        return {"intent": intent, "description": f"{intent} desc",
                "version": "1.0", "capabilities": [], "risk_level": "low"}


class TestRouting:
    def test_export_and_list_route(self):
        assert _route("export my organs as a pack") == "organ_pack_export"
        assert _route("make a pack called tools") == "organ_pack_export"
        assert _route("list my packs") == "organ_pack_list"
        assert _route("what packs do I have") == "organ_pack_list"
        assert _route("shareable capabilities") == "organ_pack_list"

    def test_does_not_steal_tasks_or_pipelines(self):
        assert _route("list my tasks") == "list_tasks"
        assert _route("list my pipelines") == "pipeline_list"


class TestExport:
    def test_export_writes_hash_verified_pack(self):
        agent = types.SimpleNamespace(
            _organ_loader=_FakeLoader(["my_synth_a", "my_synth_b"]))
        card = handle_pa_intent(agent, "organ_pack_export",
                                "export my organs as a pack called mine", {})
        assert "Exported" in card.title or "Exported" in card.body
        assert "my_synth_a" in card.body and "my_synth_b" in card.body
        # the pack file exists and verifies
        path = Path("~/.prism/packs/mine.json").expanduser()
        assert path.exists()
        import prism_organ_pack as pack
        ok, _ = pack.verify_pack(pack.read_pack(path))
        assert ok is True

    def test_export_with_no_user_organs_is_honest(self):
        agent = types.SimpleNamespace(_organ_loader=_FakeLoader([]))
        card = handle_pa_intent(agent, "organ_pack_export",
                                "export a pack", {})
        assert "no user" in card.body.lower()


class TestList:
    def test_list_shows_user_organs(self):
        agent = types.SimpleNamespace(_organ_loader=_FakeLoader(["synth_x"]))
        card = handle_pa_intent(agent, "organ_pack_list", "list my packs", {})
        assert "synth_x" in card.body
        # bundled organs are NOT listed as shareable
        assert "file_read" not in card.body

    def test_list_empty_is_honest(self):
        agent = types.SimpleNamespace(_organ_loader=_FakeLoader([]))
        card = handle_pa_intent(agent, "organ_pack_list", "list my packs", {})
        assert "No user-synth" in card.body or "bundled" in card.body
