"""
Canary Check organ — synthetic pipeline health probe.

Injects a fixed synthetic node through the full write→WAL→commit cycle,
times the round-trip, records ρ in PrismMetrics, and returns a health card.
No internet or filesystem access required — pure internal.

Triggered by the horizon planner once every 24 h in production.
Can also be invoked directly: "run canary check" / "check system health".
"""
ORGAN_META = {
    "intent":      "canary_check",
    "description": "internal pipeline health probe — measures write/commit latency and performance drift",
    "version":     "1.0",
    "capabilities": [],
}

import time  # noqa: E402
import uuid  # noqa: E402

ORGAN_POLICY = {
    "risk_level":        "low",
    "requires_approval": False,
    "irreversible":      False,
    "max_per_session":   None,
}

_CANARY_NODE_TYPE = "_canary"
_CANARY_VALUE     = {"synthetic": True, "purpose": "pipeline_health_probe"}


def execute(intent: str, message: str, ctx: dict):
    from prism_responses import text_card

    try:
        from prism_memory_graph import GraphNode, PrismMemoryGraph
        from prism_metrics import metrics
    except ImportError as exc:
        return text_card(f"Canary: import error — {exc}", "canary_check")

    # Use a shared graph if injected via ctx, else open a short-lived one
    graph: PrismMemoryGraph | None = ctx.get("memory_graph")
    owned = False
    if graph is None:
        try:
            graph = PrismMemoryGraph()
            owned = True
        except Exception as exc:
            return text_card(f"Canary: could not open graph — {exc}", "canary_check")

    canary_id = f"_canary_{uuid.uuid4().hex[:8]}"
    node = GraphNode(
        node_id   = canary_id,
        node_type = _CANARY_NODE_TYPE,
        value     = _CANARY_VALUE,
        ts        = time.time(),
    )

    t0 = time.monotonic()
    success = False
    try:
        graph.write_node(node)           # hot buffer + WAL
        graph.commit_pending()           # cold layer
        # Verify round-trip
        found = graph._cold.get_node(canary_id)
        success = found is not None
    except Exception as exc:
        return text_card(f"Canary: pipeline error — {exc}", "canary_check")
    finally:
        if owned:
            try:
                graph.close()
            except Exception:
                pass

    duration_ms = (time.monotonic() - t0) * 1000
    metrics.record_canary(duration_ms, success=success)
    metrics.inc("canary_runs")

    # Build report
    stats = metrics.canary_stats(last_n=30)
    rho   = metrics.performance_rho(last_n=30)
    alert = metrics.critical_alert()

    status_line = "OK" if success else "FAIL"
    rho_note = ""
    if rho is not None:
        rho_note = f"ρ = {rho:+.4f} ms/run"
        if rho > 5.0:
            rho_note += " ⚠ degradation detected"

    lines = [
        f"Canary: {status_line}  ({duration_ms:.1f} ms)",
        f"Mean over last {stats['n']} runs: {stats['mean_ms']} ms  |  max: {stats['max_ms']} ms",
        f"Success rate: {stats['success_rate'] * 100:.1f}%" if stats["success_rate"] is not None else "",
        rho_note,
    ]
    if alert:
        lines.append("CRITICAL: drift magnitude growing while latency is high — self-healing may be failing")

    body = "\n".join(line for line in lines if line)
    return text_card(body, "canary_check")
