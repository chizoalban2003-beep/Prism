"""
prism_organ_planner.py
======================
Composition planner — turns a set of organ intents into a wire diagram.

Each organ declares optional I/O schemas via ORGAN_META.inputs / outputs
(see prism_organ_loader.py). The planner uses OrganLoader.composable_with()
to draw arrows from producer organs to consumer organs by matching output
type → input type.

This is the foundation for the PowerBI-style arrow infrastructure the user
asked for: organs become nodes, types become edges, and a chain becomes a
DAG. Execution is *not* in scope here — this module only proves the wiring.

Usage
-----
    from prism_organ_loader import OrganLoader
    from prism_organ_planner import compose

    loader = OrganLoader()
    plan = compose(loader, ["weather_check", "translate_text"])

    # plan = {
    #   "nodes": ["weather_check", "translate_text"],
    #   "arrows": [{"from": "weather_check", "to": "translate_text",
    #               "matched_types": ["str"]}],
    #   "orphans": [],     # nodes with no inbound or outbound arrow
    #   "roots":   ["weather_check"],   # no inbound arrows
    #   "leaves":  ["translate_text"],  # no outbound arrows
    # }
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from prism_organ_loader import OrganLoader


def _matching_types(producer_outputs: dict, consumer_inputs: dict) -> list[str]:
    """Return the keys where producer outputs and consumer inputs agree on
    both name and type — PowerBI-style column matching. An arrow forms only
    when the consumer's declared input key appears in the producer's outputs
    with the same type. Empty list means no arrow."""
    matched = [
        key for key, t_in in consumer_inputs.items()
        if producer_outputs.get(key) == t_in
    ]
    return sorted(matched)


def compose(loader: OrganLoader, intents: list[str]) -> dict:
    """
    Return a wire-diagram for the given list of organ intents.

    Skips intents that are not loaded. Self-edges (where producer == consumer)
    are not emitted — they would always loop the default `card → card` arrow.
    """
    nodes = [i for i in intents if loader.get(i) is not None]
    if len(nodes) < 1:
        return {"nodes": [], "arrows": [], "orphans": [], "roots": [], "leaves": []}

    schemas = {i: loader.get_organ_schema(i) for i in nodes}

    arrows: list[dict] = []
    inbound: dict[str, int] = {i: 0 for i in nodes}
    outbound: dict[str, int] = {i: 0 for i in nodes}

    for producer in nodes:
        p_out = schemas[producer].get("outputs", {})
        if not p_out:
            continue
        for consumer in nodes:
            if consumer == producer:
                continue
            c_in = schemas[consumer].get("inputs", {})
            if not c_in:
                continue
            matched = _matching_types(p_out, c_in)
            if matched:
                arrows.append({
                    "from": producer,
                    "to":   consumer,
                    "matched_types": matched,
                })
                outbound[producer] += 1
                inbound[consumer]  += 1

    roots   = sorted([n for n in nodes if inbound[n]  == 0 and outbound[n] > 0])
    leaves  = sorted([n for n in nodes if outbound[n] == 0 and inbound[n]  > 0])
    orphans = sorted([n for n in nodes if inbound[n]  == 0 and outbound[n] == 0])

    return {
        "nodes":   nodes,
        "arrows":  arrows,
        "orphans": orphans,
        "roots":   roots,
        "leaves":  leaves,
    }


def has_cycle(plan: dict) -> bool:
    """Detect cycles in the wire diagram. Useful before execution scheduling."""
    arrows = plan.get("arrows", [])
    graph: dict[str, list[str]] = {}
    for a in arrows:
        graph.setdefault(a["from"], []).append(a["to"])

    WHITE, GRAY, BLACK = 0, 1, 2
    color: dict[str, int] = {n: WHITE for n in plan.get("nodes", [])}

    def dfs(node: str) -> bool:
        color[node] = GRAY
        for nxt in graph.get(node, []):
            if color.get(nxt, WHITE) == GRAY:
                return True
            if color.get(nxt, WHITE) == WHITE and dfs(nxt):
                return True
        color[node] = BLACK
        return False

    return any(dfs(n) for n in color if color[n] == WHITE)


def topological_order(plan: dict) -> list[str]:
    """Kahn's algorithm — returns nodes in execution order.

    Roots first, leaves last. Returns [] if the graph has a cycle.
    Orphan nodes (no arrows in or out) are appended in their original order.
    """
    nodes = list(plan.get("nodes", []))
    arrows = plan.get("arrows", [])
    if not nodes:
        return []

    indeg: dict[str, int] = {n: 0 for n in nodes}
    graph: dict[str, list[str]] = {n: [] for n in nodes}
    for a in arrows:
        f, t = a["from"], a["to"]
        if f in graph and t in indeg:
            graph[f].append(t)
            indeg[t] += 1

    ready = [n for n in nodes if indeg[n] == 0]
    order: list[str] = []
    while ready:
        n = ready.pop(0)
        order.append(n)
        for nxt in graph[n]:
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                ready.append(nxt)

    if len(order) != len(nodes):
        return []
    return order


def auto_select_organs(
    loader,
    message: str,
    llm_router,
    max_organs: int = 4,
) -> list[str]:
    """Ask the LLM to pick the smallest set of loaded organs that handle the
    user message. Returns a list of intent names that exist in the loader.

    Empty list means: the picker failed, or the LLM decided no organ fits.
    Callers should fall back to single-organ routing in that case.

    The prompt uses the LIVE loader's known_intents() — not a stale hard-
    coded registry — so newly installed or synthesised organs are reachable
    the moment they're loaded.
    """
    if llm_router is None or not message.strip():
        return []
    try:
        intents = loader.known_intents()
    except Exception:
        return []
    if not intents:
        return []

    catalog = "\n".join(f"  {k}: {v}" for k, v in sorted(intents.items()))
    prompt = (
        "You are picking organs (tools) to satisfy a user request.\n\n"
        f"Available organs:\n{catalog}\n\n"
        f"User request: \"{message}\"\n\n"
        f"Pick up to {max_organs} organs whose combined output answers the "
        "request. Prefer the smallest set. If exactly one organ is enough, "
        "return just that one. If nothing fits, return an empty list.\n\n"
        "Return ONLY valid JSON of this exact shape:\n"
        "{\"intents\": [\"organ_name\", ...]}\n"
    )
    try:
        raw, _ = llm_router.call(
            prompt, min_capability=1, max_tokens=200, json_mode=True
        )
    except Exception:
        return []
    if not raw:
        return []

    import json as _json
    text = raw.strip()
    for prefix in ("```json", "```"):
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    if text.endswith("```"):
        text = text[:-3].strip()
    try:
        data = _json.loads(text)
    except Exception:
        return []
    picked = data.get("intents") if isinstance(data, dict) else None
    if not isinstance(picked, list):
        return []

    seen: set[str] = set()
    out: list[str] = []
    for item in picked:
        name = str(item).strip()
        if name and name in intents and name not in seen:
            seen.add(name)
            out.append(name)
        if len(out) >= max_organs:
            break
    return out


def execute_plan(
    loader,
    plan: dict,
    message: str = "",
    initial_ctx: dict | None = None,
    max_steps: int = 50,
) -> dict:
    """Walk the DAG in topological order, executing each organ.

    Convention
    ----------
    Each organ runs with ``ctx`` plus a per-key entry ``ctx["_upstream"]``
    mapping intent → that organ's return value. Downstream organs that
    declared input types matching an upstream output can read structured
    data from there; organs that don't care just ignore it.

    The standard PrismCard return is preserved as the "result" — callers
    can render the final card (typically the leaf node's output) or
    inspect intermediate outputs in ``outputs[<intent>]``.

    Returns
    -------
        {
          "order":    [...],          # topological execution sequence
          "outputs":  {intent: any},  # whatever each organ returned
          "errors":   {intent: str},  # exceptions per organ (empty on success)
          "skipped":  [...],          # organs not run (cycle or hit max_steps)
          "executed": int,            # count of organs that ran
        }
    """
    order = topological_order(plan)
    nodes = plan.get("nodes", [])
    result: dict = {
        "order":    order,
        "outputs":  {},
        "errors":   {},
        "skipped":  [],
        "executed": 0,
    }
    if not order:
        result["skipped"] = list(nodes)
        if not nodes:
            result["errors"]["_plan"] = (
                "empty plan — no known organs matched the requested intents"
            )
        else:
            result["errors"]["_plan"] = "cycle detected among organs — refuse to execute"
        return result

    ctx = dict(initial_ctx or {})
    ctx.setdefault("_upstream", {})

    for i, intent in enumerate(order):
        if i >= max_steps:
            result["skipped"] = order[i:]
            break
        fn = loader.get(intent)
        if fn is None:
            result["errors"][intent] = "organ not loaded"
            result["skipped"].append(intent)
            continue
        try:
            out = fn(intent, message, ctx)
        except Exception as exc:
            result["errors"][intent] = f"{type(exc).__name__}: {exc}"
            continue
        result["outputs"][intent] = out
        ctx["_upstream"][intent] = out
        result["executed"] += 1

    return result
