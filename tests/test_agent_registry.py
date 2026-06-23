"""
tests/test_agent_registry.py
============================
Tests for the unified agent registry:
  - prism_agent_registry.inventory()  (aggregator)
  - GET /agents                        (route)
  - organs/agents_inventory.execute    (chat-routable)

All four upstream sources are stubbed so the aggregator's normalization
can be exercised without spinning up a real PrismAgent.
"""
from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field

# ── Stubs that mimic the four upstream surfaces ──────────────────────────────

class _StubLLMRouter:
    def status_summary(self):
        return {
            "best": "ollama/llama3.2:3b",
            "available": [
                {"provider": "ollama", "model": "llama3.2:3b",
                 "available": True, "capability": 2, "latency_ms": 12.3},
                {"provider": "claude", "model": "claude-sonnet",
                 "available": False, "capability": 0, "latency_ms": 0.0},
            ],
            "stdlib_only": False,
        }


class _StubOrganFn:
    def __init__(self, policy):
        self._organ_policy = policy


class _StubOrganLoader:
    def __init__(self):
        fn_a = _StubOrganFn({"risk_level": "low",  "requires_approval": False})
        fn_b = _StubOrganFn({"risk_level": "high", "requires_approval": True})
        self._organs = {
            "weather_check":   (fn_a, {"description": "weather", "capabilities": ["internet_read"]}),
            "github_issue":    (fn_b, {"description": "issues",  "capabilities": ["internet_write"]}),
        }
        self._disabled = set()


class _StubMCP:
    def status(self):
        return [
            {"name": "filesystem", "alive": True,  "initialized": True,
             "tool_count": 2, "resource_count": 0, "prompt_count": 0},
            {"name": "broken",     "alive": False, "initialized": False,
             "tool_count": 0, "resource_count": 0, "prompt_count": 0},
        ]
    def list_tools(self):
        return [
            {"server": "filesystem", "name": "read_file"},
            {"server": "filesystem", "name": "write_file"},
        ]


@dataclass
class _StubPeer:
    name: str = "laptop"
    host: str = "192.168.1.5"
    port: int = 8742
    last_seen: float = 1234567.0
    capabilities: dict = field(default_factory=lambda: {"browser": True, "ffmpeg": True, "git": False})


class _StubMesh:
    def list_peers(self):
        return [_StubPeer()]


def _stub_state():
    return {
        "agent":       None,
        "llm_router":  _StubLLMRouter(),
        "mcp":         _StubMCP(),
        "organ_loader": _StubOrganLoader(),
    }


# ── inventory() ───────────────────────────────────────────────────────────────

class TestInventory:
    def setup_method(self):
        import sys

        # Patch the lazy peer import so the test does not touch the real mesh
        from prism_agent_registry import inventory
        fake_module = type(sys)("prism_mesh")
        fake_module.get_mesh = lambda: _StubMesh()
        sys.modules["prism_mesh"] = fake_module
        self.inventory = inventory

    def test_total_count(self):
        inv = self.inventory(_stub_state())
        assert inv["summary"]["total"] == 7   # 2 llm + 2 organ + 2 mcp + 1 peer

    def test_kind_counts(self):
        inv = self.inventory(_stub_state())
        s = inv["summary"]
        assert s["llm"] == 2
        assert s["organ"] == 2
        assert s["mcp"] == 2
        assert s["peer"] == 1

    def test_ready_count(self):
        inv = self.inventory(_stub_state())
        # available: ollama (ready), claude (offline), 2 organs (loaded),
        # filesystem mcp (ready), broken mcp (offline), 1 peer (online, last_seen>0)
        # ready states: ready, loaded, online → ollama + 2 organs + filesystem + peer = 5
        assert inv["summary"]["ready"] == 5

    def test_llm_entry_shape(self):
        inv = self.inventory(_stub_state())
        llms = [a for a in inv["agents"] if a["kind"] == "llm"]
        ollama = next(a for a in llms if "ollama" in a["name"])
        assert ollama["status"] == "ready"
        assert "reasoning" in ollama["capabilities"]
        assert ollama["latency_ms"] == 12.3
        assert ollama["provider"] == "ollama"

    def test_organ_entry_shape(self):
        inv = self.inventory(_stub_state())
        organs = [a for a in inv["agents"] if a["kind"] == "organ"]
        gh = next(a for a in organs if a["name"] == "github_issue")
        assert gh["status"] == "loaded"
        assert gh["risk"] == "high"
        assert "internet_write" in gh["capabilities"]

    def test_mcp_entry_shape(self):
        inv = self.inventory(_stub_state())
        mcps = [a for a in inv["agents"] if a["kind"] == "mcp"]
        fs = next(a for a in mcps if a["name"] == "filesystem")
        assert fs["status"] == "ready"
        assert "read_file" in fs["capabilities"]
        assert "write_file" in fs["capabilities"]
        assert fs["tool_count"] == 2

    def test_peer_entry_shape(self):
        inv = self.inventory(_stub_state())
        peers = [a for a in inv["agents"] if a["kind"] == "peer"]
        assert len(peers) == 1
        p = peers[0]
        assert p["status"] == "online"
        assert "browser" in p["capabilities"]
        assert "git" not in p["capabilities"]   # capability=False excluded
        assert "192.168.1.5" in p["host"]

    def test_capability_filter_match(self):
        inv = self.inventory(_stub_state(), capability="internet_write")
        names = [a["name"] for a in inv["agents"]]
        assert "github_issue" in names
        assert "weather_check" not in names

    def test_capability_filter_substring(self):
        inv = self.inventory(_stub_state(), capability="internet")
        kinds = {a["kind"] for a in inv["agents"]}
        assert kinds == {"organ"}    # both organs match "internet"
        assert len(inv["agents"]) == 2

    def test_capability_filter_no_match(self):
        inv = self.inventory(_stub_state(), capability="nonexistent_cap")
        assert inv["agents"] == []
        assert inv["summary"]["total"] == 0

    def test_empty_state(self):
        inv = self.inventory({})
        # only peer source is global; with stub_mesh it returns 1 peer
        # but here state is empty so llm/organ/mcp empty
        s = inv["summary"]
        assert s["llm"] == 0
        assert s["organ"] == 0
        assert s["mcp"] == 0

    def test_broken_source_isolated(self):
        class _Broken:
            def status_summary(self): raise RuntimeError("boom")
        state = {"llm_router": _Broken()}
        inv = self.inventory(state)
        # broken LLM source returns empty, but does not raise
        assert inv["summary"]["llm"] == 0


# ── organs/agents_inventory.execute ───────────────────────────────────────────

def _load_organ(name: str):
    spec = importlib.util.spec_from_file_location(name, f"organs/{name}.py")
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestAgentsInventoryOrgan:
    def setup_method(self):
        import sys
        fake_module = type(sys)("prism_mesh")
        fake_module.get_mesh = lambda: _StubMesh()
        sys.modules["prism_mesh"] = fake_module
        self.organ = _load_organ("agents_inventory")

    def test_meta_shape(self):
        assert self.organ.ORGAN_META["intent"] == "agents_inventory"
        assert self.organ.ORGAN_POLICY["risk_level"] == "low"

    def test_renders_card(self):
        ctx = {"state": _stub_state()}
        card = self.organ.execute("agents_inventory", "what agents do you have?", ctx)
        body = card.body
        assert "Agents" in body
        assert "ollama" in body
        assert "weather_check" in body
        assert "filesystem" in body
        assert "laptop" in body

    def test_capability_filter_from_message(self):
        ctx = {"state": _stub_state()}
        card = self.organ.execute("agents_inventory", "agents for internet_write", ctx)
        assert "github_issue" in card.body
        assert "weather_check" not in card.body

    def test_empty_result(self):
        ctx = {"state": _stub_state()}
        card = self.organ.execute("agents_inventory", "agents for capability: zzz_nope", ctx)
        assert "No agents" in card.body


# ── GET /agents route ────────────────────────────────────────────────────────

class TestAgentsRoute:
    def test_returns_inventory_shape(self):
        import sys
        fake_module = type(sys)("prism_mesh")
        fake_module.get_mesh = lambda: _StubMesh()
        sys.modules["prism_mesh"] = fake_module

        # Inject stubs into the shared _state
        from prism_state import _state
        _saved = dict(_state)
        try:
            _state.clear()
            _state.update(_stub_state())
            # Import the route module after state is primed
            import asyncio

            from prism_routes_agents import agents_list
            result = asyncio.run(agents_list(capability=None))
            assert "agents" in result
            assert "summary" in result
            assert result["summary"]["llm"] == 2
        finally:
            _state.clear()
            _state.update(_saved)
