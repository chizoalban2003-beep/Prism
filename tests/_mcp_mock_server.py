"""A minimal MCP stdio server for tests (newline-delimited JSON-RPC 2.0).

Implements initialize, tools/list, and tools/call for two tools: echo, add.
Not a test module itself (leading underscore keeps pytest from collecting it).
"""
import json
import sys

TOOLS = [
    {
        "name": "echo",
        "description": "echo the provided text",
        "inputSchema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    },
    {
        "name": "add",
        "description": "add two numbers a and b",
        "inputSchema": {
            "type": "object",
            "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
            "required": ["a", "b"],
        },
    },
]


def _send(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        mid = msg.get("id")
        method = msg.get("method")

        if method == "initialize":
            _send({"jsonrpc": "2.0", "id": mid, "result": {
                "protocolVersion": "2024-11-05",
                "serverInfo": {"name": "mock", "version": "1.0"},
                "capabilities": {"tools": {}},
            }})
        elif method == "notifications/initialized":
            continue  # notification — no response
        elif method == "tools/list":
            _send({"jsonrpc": "2.0", "id": mid, "result": {"tools": TOOLS}})
        elif method == "resources/list":
            _send({"jsonrpc": "2.0", "id": mid, "result": {"resources": [
                {"uri": "mem://note", "name": "note",
                 "description": "a note", "mimeType": "text/plain"},
            ]}})
        elif method == "resources/read":
            uri = (msg.get("params", {}) or {}).get("uri", "")
            _send({"jsonrpc": "2.0", "id": mid, "result": {"contents": [
                {"uri": uri, "mimeType": "text/plain", "text": f"contents of {uri}"},
            ]}})
        elif method == "prompts/list":
            _send({"jsonrpc": "2.0", "id": mid, "result": {"prompts": [
                {"name": "greet", "description": "a greeting prompt",
                 "arguments": [{"name": "who", "required": True}]},
            ]}})
        elif method == "prompts/get":
            params = msg.get("params", {}) or {}
            who = (params.get("arguments", {}) or {}).get("who", "world")
            _send({"jsonrpc": "2.0", "id": mid, "result": {
                "description": "greeting",
                "messages": [{"role": "user", "content": {
                    "type": "text", "text": f"Hello, {who}!"}}],
            }})
        elif method == "tools/call":
            params = msg.get("params", {})
            name = params.get("name")
            args = params.get("arguments", {})
            if name == "echo":
                _send({"jsonrpc": "2.0", "id": mid, "result": {
                    "content": [{"type": "text", "text": str(args.get("text", ""))}]
                }})
            elif name == "add":
                try:
                    total = float(args.get("a", 0)) + float(args.get("b", 0))
                    _send({"jsonrpc": "2.0", "id": mid, "result": {
                        "content": [{"type": "text", "text": str(total)}]
                    }})
                except Exception as exc:
                    _send({"jsonrpc": "2.0", "id": mid, "result": {
                        "content": [{"type": "text", "text": f"error: {exc}"}],
                        "isError": True,
                    }})
            else:
                _send({"jsonrpc": "2.0", "id": mid,
                       "error": {"code": -32601, "message": "unknown tool"}})
        elif mid is not None:
            _send({"jsonrpc": "2.0", "id": mid,
                   "error": {"code": -32601, "message": "method not found"}})


if __name__ == "__main__":
    main()
