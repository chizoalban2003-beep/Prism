"""Tests for prism_mcp — MCP client, organ bridge, and routes (mock stdio server)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

import prism_mcp
from prism_mcp import MCPManager, MCPServer, MCPTool, extract_text

_MOCK = str(Path(__file__).parent / "_mcp_mock_server.py")
_CMD = [sys.executable, _MOCK]


@pytest.fixture()
def manager():
    mgr = MCPManager()
    mgr.add_server("mock", _CMD)
    mgr.connect("mock")
    yield mgr
    mgr.shutdown()


# ── server handshake + tools ────────────────────────────────────────────────────

class TestServer:
    def test_connect_lists_tools(self, manager):
        tools = manager.list_tools()
        names = {t["name"] for t in tools}
        assert {"echo", "add"} <= names
        assert all(t["server"] == "mock" for t in tools)

    def test_status_alive(self, manager):
        st = manager.status()
        assert st[0]["alive"] is True
        assert st[0]["initialized"] is True
        assert st[0]["tool_count"] == 2

    def test_call_echo(self, manager):
        out = manager.call_text("mock", "echo", {"text": "hello mcp"})
        assert out == "hello mcp"

    def test_call_add(self, manager):
        out = manager.call_text("mock", "add", {"a": 2, "b": 40})
        assert out == "42.0"

    def test_unknown_tool_errors(self, manager):
        from prism_mcp import MCPError
        with pytest.raises(MCPError):
            manager.call_tool("mock", "does_not_exist", {})

    def test_unknown_server_errors(self, manager):
        from prism_mcp import MCPError
        with pytest.raises(MCPError):
            manager.call_tool("nope", "echo", {})


# ── config parsing ──────────────────────────────────────────────────────────────

class TestFromConfig:
    def test_disabled_when_absent(self):
        mgr = MCPManager.from_config({})
        assert mgr.server_names() == []

    def test_disabled_flag(self):
        mgr = MCPManager.from_config({"mcp": {"enabled": False, "servers": [
            {"name": "x", "command": ["echo"]}]}})
        assert mgr.server_names() == []

    def test_enabled_parses_servers(self):
        mgr = MCPManager.from_config({"mcp": {"enabled": True, "servers": [
            {"name": "fs", "command": ["npx", "server"]},
            {"name": "bad"},  # missing command — skipped
        ]}})
        assert mgr.server_names() == ["fs"]

    def test_command_must_be_list(self):
        with pytest.raises(ValueError):
            MCPServer("x", command="not-a-list")  # type: ignore[arg-type]


# ── result extraction ────────────────────────────────────────────────────────────

class TestExtractText:
    def test_text_blocks(self):
        assert extract_text({"content": [
            {"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}) == "a\nb"

    def test_resource_block(self):
        assert "u.txt" in extract_text({"content": [
            {"type": "resource", "resource": {"uri": "u.txt"}}]})

    def test_non_dict(self):
        assert extract_text("plain") == "plain"


# ── organ bridge ─────────────────────────────────────────────────────────────────

class TestOrganBridge:
    def test_register_creates_organs(self, manager):
        import tempfile

        from prism_organ_loader import OrganLoader
        d = tempfile.mkdtemp()
        loader = OrganLoader(bundled_dir=Path(d) / "b", user_dir=Path(d) / "u")
        n = prism_mcp.register_mcp_organs(loader, manager)
        assert n == 2
        assert loader.get("mcp.mock.echo") is not None
        assert "mcp.mock.add" in loader.known_intents()

    def test_organ_executes_with_ctx_args(self, manager):
        _, fn, _ = prism_mcp.make_mcp_organ(manager, MCPTool("mock", "echo"))
        card = fn("mcp.mock.echo", "ignored", {"mcp_arguments": {"text": "via ctx"}})
        assert "via ctx" in card.body

    def test_organ_single_string_arg_from_message(self, manager):
        tool = manager.find_tool("echo")
        _, fn, _ = prism_mcp.make_mcp_organ(manager, tool)
        # echo's schema has a single required string 'text' → message maps into it
        card = fn("mcp.mock.echo", "mapped message", {})
        assert "mapped message" in card.body

    def test_organ_json_message_args(self, manager):
        tool = manager.find_tool("add")
        _, fn, _ = prism_mcp.make_mcp_organ(manager, tool)
        card = fn("mcp.mock.add", '{"a": 1, "b": 5}', {})
        assert "6.0" in card.body

    def test_llm_arg_synthesis_for_multi_prop_schema(self, manager):
        # 'add' has two numeric props → no deterministic single-string mapping;
        # a router synthesises the JSON arguments from the message.
        class _StubRouter:
            def call(self, prompt, **kw):
                return ('{"a": 4, "b": 6}', "stub")
        tool = manager.find_tool("add")
        _, fn, _ = prism_mcp.make_mcp_organ(manager, tool, router=_StubRouter())
        card = fn("mcp.mock.add", "add four and six", {})
        assert "10.0" in card.body


# ── HTTP routes ──────────────────────────────────────────────────────────────────

class TestRoutes:
    def _client(self, manager):
        from fastapi.testclient import TestClient

        from prism_asgi import app
        from prism_state import _set_state
        _set_state(agent=None, mcp=manager)
        return TestClient(app, raise_server_exceptions=False)

    def test_status_and_tools(self, manager):
        c = self._client(manager)
        assert c.get("/mcp/status").json()["enabled"] is True
        tools = c.get("/mcp/tools").json()
        assert tools["count"] == 2

    def test_call_route(self, manager):
        c = self._client(manager)
        r = c.post("/mcp/call", json={"server": "mock", "tool": "add",
                                      "arguments": {"a": 10, "b": 5}})
        assert r.status_code == 200
        assert r.json()["text"] == "15.0"

    def test_call_route_missing_fields(self, manager):
        c = self._client(manager)
        r = c.post("/mcp/call", json={"server": "mock"})
        assert r.status_code == 400


# ── resources + prompts ─────────────────────────────────────────────────────────

class TestResources:
    def test_list_resources(self, manager):
        res = manager.list_resources()
        assert any(r["uri"] == "mem://note" and r["server"] == "mock" for r in res)

    def test_read_resource(self, manager):
        out = manager.read_resource("mock", "mem://note")
        assert "mem://note" in extract_text(
            {"content": [{"type": "text", "text": out["contents"][0]["text"]}]})

    def test_status_counts(self, manager):
        st = manager.status()[0]
        assert st["resource_count"] == 1
        assert st["prompt_count"] == 1


class TestPrompts:
    def test_list_prompts(self, manager):
        prompts = manager.list_prompts()
        assert any(p["name"] == "greet" and p["server"] == "mock" for p in prompts)

    def test_get_prompt(self, manager):
        out = manager.get_prompt("mock", "greet", {"who": "PRISM"})
        assert "Hello, PRISM!" in out["messages"][0]["content"]["text"]


class TestHttpTransport:
    """End-to-end HTTP (Streamable HTTP) transport against a local mock server."""

    @pytest.fixture()
    def http_manager(self):
        import http.server
        import json as _json
        import threading

        tools = [{"name": "echo", "description": "echo",
                  "inputSchema": {"type": "object",
                                  "properties": {"text": {"type": "string"}},
                                  "required": ["text"]}}]

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _json(self, obj, extra=None):
                body = _json.dumps(obj).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                for k, v in (extra or {}).items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                ln = int(self.headers.get("Content-Length", "0"))
                req = _json.loads(self.rfile.read(ln) or b"{}")
                mid, method = req.get("id"), req.get("method")
                if mid is None:                       # notification
                    self.send_response(202)
                    self.end_headers()
                    return
                if method == "initialize":
                    res = {"protocolVersion": "2024-11-05",
                           "serverInfo": {"name": "http-mock", "version": "1.0"},
                           "capabilities": {}}
                    self._json({"jsonrpc": "2.0", "id": mid, "result": res},
                               extra={"Mcp-Session-Id": "sess-1"})
                    return
                if method == "tools/list":
                    res = {"tools": tools}
                elif method == "tools/call":
                    args = (req.get("params", {}) or {}).get("arguments", {})
                    res = {"content": [{"type": "text",
                                        "text": str(args.get("text", ""))}]}
                else:
                    self._json({"jsonrpc": "2.0", "id": mid,
                                "error": {"code": -32601, "message": "no"}})
                    return
                self._json({"jsonrpc": "2.0", "id": mid, "result": res})

        srv = http.server.HTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        mgr = MCPManager()
        mgr.add_server("http", url=f"http://127.0.0.1:{srv.server_address[1]}/")
        mgr.connect("http")
        yield mgr
        mgr.shutdown()
        srv.shutdown()

    def test_http_lists_tools(self, http_manager):
        assert any(t["name"] == "echo" for t in http_manager.list_tools())

    def test_http_calls_tool(self, http_manager):
        assert http_manager.call_text("http", "echo", {"text": "hi http"}) == "hi http"

    def test_http_status_alive(self, http_manager):
        st = http_manager.status()[0]
        assert st["alive"] is True and st["initialized"] is True


class TestConfigHttp:
    def test_from_config_url_server(self):
        mgr = MCPManager.from_config({"mcp": {"enabled": True, "servers": [
            {"name": "remote", "url": "https://example.com/mcp"}]}})
        assert mgr.server_names() == ["remote"]

    def test_from_config_mixed(self):
        mgr = MCPManager.from_config({"mcp": {"enabled": True, "servers": [
            {"name": "a", "command": ["x"]},
            {"name": "b", "url": "https://h/mcp"},
            {"name": "bad"}]}})
        assert sorted(mgr.server_names()) == ["a", "b"]


class TestChatRouting:
    """MCP tools registered as organs are reachable through the chat router."""

    def test_mcp_organ_executes_via_agent(self, manager):
        from prism_agent import PrismAgent
        agent = PrismAgent()
        prism_mcp.register_mcp_organs(agent._organ_loader, manager)
        card = agent._execute("mcp.mock.echo", "hello",
                              {"mcp_arguments": {"text": "routed via chat"}})
        assert "routed via chat" in card.body

    def test_mcp_intent_surfaced_to_classifier(self, manager):
        from prism_agent import PrismAgent
        agent = PrismAgent()
        prism_mcp.register_mcp_organs(agent._organ_loader, manager)

        class _StubRouter:
            def call(self, prompt, **kw):
                # The classifier prompt should list the mcp organ as a label.
                assert "mcp.mock.echo" in prompt
                return ("mcp.mock.echo", "stub")

        agent._router = _StubRouter()
        assert agent._llm_classify("please echo something") == "mcp.mock.echo"


class TestResourcePromptRoutes:
    def _client(self, manager):
        from fastapi.testclient import TestClient

        from prism_asgi import app
        from prism_state import _set_state
        _set_state(agent=None, mcp=manager)
        return TestClient(app, raise_server_exceptions=False)

    def test_resources_route(self, manager):
        c = self._client(manager)
        assert c.get("/mcp/resources").json()["count"] == 1

    def test_resource_read_route(self, manager):
        c = self._client(manager)
        r = c.post("/mcp/resource/read", json={"server": "mock", "uri": "mem://note"})
        assert r.status_code == 200
        assert "mem://note" in r.json()["result"]["contents"][0]["text"]

    def test_prompts_route(self, manager):
        c = self._client(manager)
        assert c.get("/mcp/prompts").json()["count"] == 1

    def test_prompt_get_route(self, manager):
        c = self._client(manager)
        r = c.post("/mcp/prompt/get",
                   json={"server": "mock", "name": "greet", "arguments": {"who": "X"}})
        assert r.status_code == 200
        assert "Hello, X!" in r.json()["result"]["messages"][0]["content"]["text"]
