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
    """Return the set of types that appear in both producer outputs and
    consumer inputs. Empty list means the arrow can't be drawn."""
    out_types = set(producer_outputs.values())
    in_types = set(consumer_inputs.values())
    return sorted(out_types & in_types)


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
