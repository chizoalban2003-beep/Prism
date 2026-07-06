#!/usr/bin/env python3
"""Minimal stdio MCP server — zero dependencies, used to demo/verify the
MCP → organ → tool-loop path (docs/command-centre-assessment.md gap 1).

Wire it up in prism_config.toml:

    [mcp]
    enabled = true

    [[mcp.servers]]
    name    = "demo"
    command = ["python3", "demos/mcp_demo_server.py"]

Its tools register as organs ``mcp.demo.echo`` / ``mcp.demo.local_time``
and appear in the agentic loop's tool belt as ``mcp_demo_echo`` /
``mcp_demo_local_time``.
"""
import json
import sys
import time

TOOLS = [
    {
        "name": "echo",
        "description": "Echo the provided text back, prefixed with [echo].",
        "inputSchema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    },
    {
        "name": "local_time",
        "description": "Return the server's local date and time.",
        "inputSchema": {"type": "object", "properties": {}},
    },
]


def _result(mid, payload):
    return {"jsonrpc": "2.0", "id": mid, "result": payload}


def _call(name: str, args: dict) -> dict:
    if name == "echo":
        text = "[echo] " + str(args.get("text", ""))
    elif name == "local_time":
        text = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    else:
        return {"content": [{"type": "text", "text": f"unknown tool {name}"}],
                "isError": True}
    return {"content": [{"type": "text", "text": text}], "isError": False}


def main() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        mid, method = msg.get("id"), msg.get("method", "")
        if mid is None:
            continue  # notification — nothing to answer
        if method == "initialize":
            out = _result(mid, {
                "protocolVersion": msg.get("params", {}).get(
                    "protocolVersion", "2024-11-05"),
                "serverInfo": {"name": "prism-demo", "version": "1.0"},
                "capabilities": {"tools": {}},
            })
        elif method == "tools/list":
            out = _result(mid, {"tools": TOOLS})
        elif method == "tools/call":
            p = msg.get("params", {})
            out = _result(mid, _call(p.get("name", ""),
                                     p.get("arguments") or {}))
        else:
            out = _result(mid, {})
        sys.stdout.write(json.dumps(out) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
