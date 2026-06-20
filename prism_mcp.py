"""
prism_mcp.py
============
Model Context Protocol (MCP) client for PRISM.

MCP (https://modelcontextprotocol.io) is an open JSON-RPC 2.0 protocol that lets
an agent connect to external "servers" exposing **tools**, resources, and
prompts. This module implements the **client** side over the *stdio* transport
(newline-delimited JSON-RPC), so PRISM can borrow capabilities from any MCP
server (filesystem, GitHub, Slack, a database, …) the user configures — the
same way Hermes/Claude Desktop/Cursor do.

Design
------
* One ``MCPServer`` per configured server: a long-lived subprocess spoken to
  over stdin/stdout. A background reader thread demultiplexes responses by
  JSON-RPC id, so calls are safe to issue concurrently.
* ``MCPManager`` owns the set of servers, performs the ``initialize`` handshake,
  caches ``tools/list``, and dispatches ``tools/call``.
* Servers are **user-configured trusted commands** (like any local integration);
  we never use a shell — argv lists only.

Config (``prism_config.toml``)::

    [mcp]
    enabled = true

    [[mcp.servers]]
    name    = "filesystem"
    command = ["npx", "-y", "@modelcontextprotocol/server-filesystem", "/data"]
    # env   = { FOO = "bar" }    # optional

Programmatic::

    mgr = MCPManager.from_config(config)
    mgr.connect_all()
    tools = mgr.list_tools()                       # [{server, name, description, schema}]
    out   = mgr.call_tool("filesystem", "read_file", {"path": "/data/x.txt"})
"""
from __future__ import annotations

import json
import logging
import os
import queue
import subprocess
import threading
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

# MCP protocol version this client advertises. Servers may negotiate a
# different one in their initialize result; we accept whatever they return.
PROTOCOL_VERSION = "2024-11-05"
_CLIENT_INFO = {"name": "prism", "version": "0.1"}
_DEFAULT_TIMEOUT = 30.0


@dataclass
class MCPTool:
    server: str
    name: str
    description: str = ""
    input_schema: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "server": self.server,
            "name": self.name,
            "description": self.description,
            "schema": self.input_schema,
        }


class MCPError(Exception):
    """Raised on protocol / transport / tool errors."""


class MCPServer:
    """A single MCP server spoken to over stdio (newline-delimited JSON-RPC)."""

    def __init__(
        self,
        name: str,
        command: list[str],
        env: Optional[dict[str, str]] = None,
        cwd: Optional[str] = None,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        if not command or not isinstance(command, list):
            raise ValueError("MCP server 'command' must be a non-empty argv list")
        self.name = name
        self.command = command
        self.env = env or {}
        self.cwd = cwd
        self.timeout = timeout

        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()          # serialises writes
        self._id = 0
        self._pending: dict[int, queue.Queue] = {}
        self._pending_lock = threading.Lock()
        self._reader: Optional[threading.Thread] = None
        self._alive = False
        self.initialized = False
        self.server_info: dict = {}
        self.tools: list[MCPTool] = []

    # ── lifecycle ────────────────────────────────────────────────────────────

    def start(self) -> None:
        if self._alive:
            return
        full_env = {**os.environ, **self.env}
        try:
            self._proc = subprocess.Popen(
                self.command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                env=full_env,
                cwd=self.cwd,
                text=True,
                bufsize=1,                     # line-buffered
            )
        except FileNotFoundError as exc:
            raise MCPError(f"MCP server '{self.name}' command not found: {exc}") from exc
        except Exception as exc:
            raise MCPError(f"failed to start MCP server '{self.name}': {exc}") from exc
        self._alive = True
        self._reader = threading.Thread(
            target=self._read_loop, name=f"mcp-{self.name}", daemon=True
        )
        self._reader.start()

    def stop(self) -> None:
        self._alive = False
        proc = self._proc
        if proc is not None:
            try:
                proc.terminate()
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.kill()
            except Exception:
                pass
        self._proc = None
        self.initialized = False

    def is_alive(self) -> bool:
        return self._alive and self._proc is not None and self._proc.poll() is None

    # ── transport ────────────────────────────────────────────────────────────

    def _read_loop(self) -> None:
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            mid = msg.get("id")
            if mid is None:
                continue  # notification from server — ignored
            with self._pending_lock:
                q = self._pending.pop(mid, None)
            if q is not None:
                q.put(msg)
        self._alive = False

    def _notify(self, method: str, params: Optional[dict] = None) -> None:
        if self._proc is None or self._proc.stdin is None:
            raise MCPError(f"MCP server '{self.name}' not running")
        payload: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            payload["params"] = params
        with self._lock:
            self._proc.stdin.write(json.dumps(payload) + "\n")
            self._proc.stdin.flush()

    def _request(self, method: str, params: Optional[dict] = None,
                 timeout: Optional[float] = None) -> Any:
        if not self.is_alive():
            raise MCPError(f"MCP server '{self.name}' is not running")
        if self._proc is None or self._proc.stdin is None:
            raise MCPError(f"MCP server '{self.name}' has no stdin")
        with self._lock:
            self._id += 1
            mid = self._id
            q: queue.Queue = queue.Queue(maxsize=1)
            with self._pending_lock:
                self._pending[mid] = q
            payload: dict[str, Any] = {"jsonrpc": "2.0", "id": mid, "method": method}
            if params is not None:
                payload["params"] = params
            try:
                self._proc.stdin.write(json.dumps(payload) + "\n")
                self._proc.stdin.flush()
            except Exception as exc:
                with self._pending_lock:
                    self._pending.pop(mid, None)
                raise MCPError(f"write to '{self.name}' failed: {exc}") from exc
        try:
            msg = q.get(timeout=timeout or self.timeout)
        except queue.Empty:
            with self._pending_lock:
                self._pending.pop(mid, None)
            raise MCPError(f"MCP '{self.name}.{method}' timed out") from None
        if "error" in msg and msg["error"]:
            err = msg["error"]
            raise MCPError(f"MCP '{self.name}.{method}' error: "
                           f"{err.get('message', err)}")
        return msg.get("result")

    # ── handshake + capabilities ─────────────────────────────────────────────

    def initialize(self) -> dict:
        result = self._request("initialize", {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": _CLIENT_INFO,
        })
        self.server_info = (result or {}).get("serverInfo", {})
        # Per spec, the client signals readiness with this notification.
        try:
            self._notify("notifications/initialized")
        except Exception:
            pass
        self.initialized = True
        return result or {}

    def refresh_tools(self) -> list[MCPTool]:
        result = self._request("tools/list")
        tools: list[MCPTool] = []
        for t in (result or {}).get("tools", []) or []:
            if not isinstance(t, dict) or not t.get("name"):
                continue
            tools.append(MCPTool(
                server=self.name,
                name=str(t["name"]),
                description=str(t.get("description", "")),
                input_schema=t.get("inputSchema") or t.get("input_schema") or {},
            ))
        self.tools = tools
        return tools

    def call_tool(self, tool_name: str, arguments: Optional[dict] = None) -> dict:
        result = self._request("tools/call", {
            "name": tool_name,
            "arguments": arguments or {},
        })
        return result or {}


# ── result helpers ────────────────────────────────────────────────────────────

def extract_text(call_result: dict) -> str:
    """Flatten an MCP tools/call result's content blocks into plain text."""
    if not isinstance(call_result, dict):
        return str(call_result)
    parts: list[str] = []
    for block in call_result.get("content", []) or []:
        if not isinstance(block, dict):
            parts.append(str(block))
            continue
        btype = block.get("type")
        if btype == "text":
            parts.append(str(block.get("text", "")))
        elif btype == "resource":
            res = block.get("resource", {})
            parts.append(str(res.get("text") or res.get("uri") or res))
        else:
            parts.append(json.dumps(block))
    return "\n".join(p for p in parts if p)


# ── manager ─────────────────────────────────────────────────────────────────

class MCPManager:
    """Owns the configured MCP servers and routes tool calls."""

    def __init__(self) -> None:
        self._servers: dict[str, MCPServer] = {}

    # construction -------------------------------------------------------------

    def add_server(
        self,
        name: str,
        command: list[str],
        env: Optional[dict[str, str]] = None,
        cwd: Optional[str] = None,
    ) -> MCPServer:
        srv = MCPServer(name=name, command=command, env=env, cwd=cwd)
        self._servers[name] = srv
        return srv

    @classmethod
    def from_config(cls, config: dict) -> MCPManager:
        """Build a manager from the ``[mcp]`` section of prism_config.toml.

        Returns an empty (disabled) manager when MCP is absent or disabled.
        """
        mgr = cls()
        mcp_cfg = (config or {}).get("mcp", {})
        if not isinstance(mcp_cfg, dict) or not mcp_cfg.get("enabled", False):
            return mgr
        servers = mcp_cfg.get("servers", []) or []
        # Accept either a list of tables or a dict-of-tables.
        if isinstance(servers, dict):
            servers = [{"name": k, **v} for k, v in servers.items()]
        for entry in servers:
            if not isinstance(entry, dict):
                continue
            name = str(entry.get("name", "")).strip()
            command = entry.get("command")
            if not name or not isinstance(command, list) or not command:
                logger.warning("[mcp] skipping malformed server entry: %r", entry)
                continue
            mgr.add_server(
                name=name,
                command=[str(c) for c in command],
                env={str(k): str(v) for k, v in (entry.get("env") or {}).items()},
                cwd=entry.get("cwd"),
            )
        return mgr

    # connection ---------------------------------------------------------------

    def connect(self, name: str) -> dict:
        srv = self._servers.get(name)
        if srv is None:
            raise MCPError(f"unknown MCP server: {name}")
        srv.start()
        srv.initialize()
        srv.refresh_tools()
        logger.info("[mcp] connected '%s' (%d tool(s))", name, len(srv.tools))
        return {"server": name, "server_info": srv.server_info,
                "tools": [t.name for t in srv.tools]}

    def connect_all(self) -> dict[str, Any]:
        results: dict[str, Any] = {}
        for name in list(self._servers):
            try:
                results[name] = self.connect(name)
            except Exception as exc:
                logger.warning("[mcp] connect '%s' failed: %s", name, exc)
                results[name] = {"error": str(exc)}
        return results

    def shutdown(self) -> None:
        for srv in self._servers.values():
            srv.stop()

    # introspection ------------------------------------------------------------

    def server_names(self) -> list[str]:
        return list(self._servers)

    def status(self) -> list[dict]:
        return [
            {
                "name": s.name,
                "alive": s.is_alive(),
                "initialized": s.initialized,
                "tool_count": len(s.tools),
                "server_info": s.server_info,
            }
            for s in self._servers.values()
        ]

    def list_tools(self) -> list[dict]:
        out: list[dict] = []
        for s in self._servers.values():
            out.extend(t.to_dict() for t in s.tools)
        return out

    def find_tool(self, tool_name: str) -> Optional[MCPTool]:
        """Locate a tool by ``server.tool`` or bare ``tool`` (first match)."""
        if "." in tool_name:
            sname, _, tname = tool_name.partition(".")
            srv = self._servers.get(sname)
            if srv:
                for t in srv.tools:
                    if t.name == tname:
                        return t
            return None
        for s in self._servers.values():
            for t in s.tools:
                if t.name == tool_name:
                    return t
        return None

    # invocation ---------------------------------------------------------------

    def call_tool(self, server: str, tool: str,
                  arguments: Optional[dict] = None) -> dict:
        srv = self._servers.get(server)
        if srv is None:
            raise MCPError(f"unknown MCP server: {server}")
        if not srv.initialized:
            raise MCPError(f"MCP server '{server}' not connected")
        return srv.call_tool(tool, arguments)

    def call_text(self, server: str, tool: str,
                  arguments: Optional[dict] = None) -> str:
        return extract_text(self.call_tool(server, tool, arguments))


# ── organ bridge — expose MCP tools as PRISM organs ───────────────────────────

def _resolve_arguments(tool: MCPTool, message: str, ctx: Optional[dict]) -> dict:
    """Best-effort mapping of an organ call into MCP tool arguments.

    Priority: explicit ``ctx['mcp_arguments']`` → a JSON object in the message →
    a single required string property filled from the message → empty.
    """
    if isinstance(ctx, dict):
        explicit = ctx.get("mcp_arguments")
        if isinstance(explicit, dict):
            return explicit
    m = (message or "").strip()
    if m.startswith("{"):
        try:
            parsed = json.loads(m)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
    schema = tool.input_schema or {}
    props = schema.get("properties") or {}
    if not props:
        return {}
    required = schema.get("required") or list(props)
    string_props = [
        k for k in required
        if (props.get(k, {}) or {}).get("type", "string") == "string"
    ]
    if len(string_props) == 1:
        return {string_props[0]: message}
    return {}


def make_mcp_organ(manager: MCPManager, tool: MCPTool):
    """Return ``(intent, execute_fn, meta)`` exposing one MCP tool as an organ."""
    intent = f"mcp.{tool.server}.{tool.name}"
    meta = {
        "intent": intent,
        "description": f"[MCP:{tool.server}] {tool.description or tool.name}",
        "version": "1.0",
        "capabilities": ["internet_read"],
        "inputs": {"message": "str", "ctx": "dict"},
        "outputs": {"card": "PrismCard"},
    }

    def execute(intent: str, message: str, ctx: dict):
        from prism_responses import text_card
        try:
            args = _resolve_arguments(tool, message, ctx)
            result = manager.call_tool(tool.server, tool.name, args)
            text = extract_text(result)
            if result.get("isError"):
                return text_card(f"MCP tool reported an error:\n{text}",
                                 f"MCP · {tool.name}")
            return text_card(text or "(no output)", f"MCP · {tool.name}")
        except Exception as exc:
            return text_card(f"MCP tool '{tool.name}' failed: {exc}",
                             f"MCP · {tool.name}")

    execute._organ_meta = meta          # type: ignore[attr-defined]
    execute._organ_policy = {           # type: ignore[attr-defined]
        "risk_level": "medium",
        "requires_approval": False,
        "irreversible": False,
        "max_per_session": None,
    }
    return intent, execute, meta


def register_mcp_organs(loader: Any, manager: MCPManager) -> int:
    """Register every connected MCP tool as an organ on *loader*. Returns count."""
    count = 0
    for srv in manager._servers.values():
        for tool in srv.tools:
            intent, fn, meta = make_mcp_organ(manager, tool)
            try:
                loader._register(intent, fn, meta, source="mcp")
                count += 1
            except Exception as exc:
                logger.warning("[mcp] failed to register organ %s: %s", intent, exc)
    return count


# ── module singleton (mirrors prism_phase.get_engine) ─────────────────────────

_manager: Optional[MCPManager] = None


def get_manager() -> MCPManager:
    global _manager
    if _manager is None:
        _manager = MCPManager()
    return _manager


def set_manager(mgr: MCPManager) -> None:
    global _manager
    _manager = mgr
