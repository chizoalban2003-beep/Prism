"""
prism_chat_graph_bridge.py
==========================
Bridges PrismAgent chat turns to the WAL-backed memory graph.

Two helpers extracted from PrismAgent:

* :func:`recall_from_graph` — appends durable-graph hits onto the
  ``memory_context`` list already on a context dict. Flat semantic recall
  stays primary; this surfaces past observation-turn nodes the flat store
  may not return.
* :func:`mirror_turn_to_graph` — writes a conversation turn as a graph
  node (user or assistant), and links assistant→user with an
  ``answered_by`` edge when a prior user-turn node id is provided. Returns
  the new ``last_user_node`` value so the caller can persist it.

Both are best-effort and swallow exceptions: a missing or misbehaving
graph never breaks the chat path. They are no-ops when ``memory_graph``
is ``None`` (e.g. PrismAgent running outside the daemon).
"""
from __future__ import annotations

import re
from typing import Any, Optional


def recall_from_graph(
    memory_graph: Any,
    message: str,
    context: dict,
) -> None:
    """Merge durable memory-graph hits into ``context['memory_context']``."""
    if memory_graph is None or not message:
        return
    try:
        q_tokens = set(re.findall(r"[a-z0-9]{3,}", message.lower()))
        if not q_tokens:
            return
        existing = context.get("memory_context", []) or []
        seen = {(e.get("excerpt", "") or "")[:100] for e in existing}
        scored: list[tuple[int, Any, str]] = []
        for node in memory_graph.query_nodes(node_type="observation", limit=50):
            content = (getattr(node, "value", {}) or {}).get("content", "")
            if not content:
                continue
            overlap = len(q_tokens & set(re.findall(r"[a-z0-9]{3,}", content.lower())))
            if overlap:
                scored.append((overlap, node, content))
        scored.sort(key=lambda x: x[0], reverse=True)
        for _overlap, node, content in scored[:5]:
            key = content[:100]
            if key in seen:
                continue
            seen.add(key)
            existing.append({
                "title": content[:60] or getattr(node, "node_id", "memory"),
                "excerpt": content[:300],
                "source": (node.value or {}).get("source", getattr(node, "node_type", "graph")),
                "score": 0.5,
            })
        if existing:
            context["memory_context"] = existing[:5]
    except Exception:
        pass


def mirror_turn_to_graph(
    memory_graph: Any,
    role: str,
    content: str,
    entry_id: Any,
    last_user_node: Optional[str],
) -> Optional[str]:
    """Mirror a chat turn into the graph. Returns the new ``last_user_node``."""
    if memory_graph is None or not entry_id or not content:
        return last_user_node
    new_last = last_user_node
    try:
        from prism_memory_graph import GraphEdge, GraphNode
        node_id = f"conv_{entry_id}"
        memory_graph.write_node(GraphNode(
            node_id=node_id,
            node_type="observation",
            value={"role": role, "content": content[:2000],
                   "source": "conversation"},
        ))
        if role == "user":
            new_last = node_id
        elif role == "assistant" and last_user_node:
            memory_graph.write_edge(GraphEdge(
                src=last_user_node, dst=node_id, relation="answered_by"))
            new_last = None
    except Exception:
        pass
    return new_last
