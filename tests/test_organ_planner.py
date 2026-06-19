"""Tests for prism_organ_planner — wire-diagram + DAG executor."""
from prism_organ_planner import compose, execute_plan, has_cycle, topological_order


class _StubLoader:
    """Minimal loader stand-in: declare each intent's schema + execute fn."""

    def __init__(self, schemas: dict):
        self._s = schemas

    def get(self, intent):
        return self._s.get(intent, {}).get("fn")

    def get_organ_schema(self, intent):
        e = self._s.get(intent, {})
        return {"inputs": e.get("inputs", {}), "outputs": e.get("outputs", {})}


def _identity_fn(value):
    def fn(intent, message, ctx):
        return {"value": value, "upstream": dict(ctx.get("_upstream", {}))}
    return fn


def test_compose_draws_arrow_on_matching_key_and_type():
    loader = _StubLoader({
        "producer": {"outputs": {"value": "float"}, "fn": _identity_fn(1.0)},
        "consumer": {"inputs":  {"value": "float"}, "fn": _identity_fn(2.0)},
    })
    plan = compose(loader, ["producer", "consumer"])
    assert len(plan["arrows"]) == 1
    arr = plan["arrows"][0]
    assert arr["from"] == "producer" and arr["to"] == "consumer"
    assert arr["matched_types"] == ["value"]


def test_compose_skips_spurious_type_only_matches():
    """Producer.out_str and consumer.in_str both 'str' but key names differ —
    no arrow should be drawn (this was the pre-fix cycle bug)."""
    loader = _StubLoader({
        "p": {"outputs": {"city": "str"}, "fn": _identity_fn(0)},
        "c": {"inputs":  {"country": "str"}, "fn": _identity_fn(0)},
    })
    plan = compose(loader, ["p", "c"])
    assert plan["arrows"] == []
    assert sorted(plan["orphans"]) == ["c", "p"]


def test_topological_order_and_executor_propagate_upstream():
    loader = _StubLoader({
        "src":  {"outputs": {"x": "float"},        "fn": lambda i, m, c: {"x": 3.14}},
        "mid":  {"inputs":  {"x": "float"},
                 "outputs": {"y": "float"},
                 "fn": lambda i, m, c: {"y": c["_upstream"]["src"]["x"] * 2}},
        "leaf": {"inputs":  {"y": "float"},
                 "fn": lambda i, m, c: {"final": c["_upstream"]["mid"]["y"] + 1}},
    })
    plan = compose(loader, ["src", "mid", "leaf"])
    assert not has_cycle(plan)
    assert topological_order(plan) == ["src", "mid", "leaf"]
    result = execute_plan(loader, plan)
    assert result["executed"] == 3
    assert result["outputs"]["leaf"]["final"] == 3.14 * 2 + 1


def test_has_cycle_detects_loop():
    loader = _StubLoader({
        "a": {"inputs": {"v": "int"}, "outputs": {"v": "int"}, "fn": _identity_fn(0)},
        "b": {"inputs": {"v": "int"}, "outputs": {"v": "int"}, "fn": _identity_fn(0)},
    })
    plan = compose(loader, ["a", "b"])
    assert has_cycle(plan)
    assert topological_order(plan) == []
