"""Tests for prism_organ_planner — wire-diagram + DAG executor + auto-pick."""
from prism_organ_planner import (
    auto_select_organs,
    compose,
    execute_plan,
    has_cycle,
    topological_order,
)


class _StubRouter:
    def __init__(self, response):
        self.response = response
        self.last_prompt = None

    def call(self, prompt, **kw):
        self.last_prompt = prompt
        return (self.response, {})


class _LiveLoaderStub:
    def __init__(self, intents):
        self._intents = intents

    def known_intents(self):
        return dict(self._intents)


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


def test_compose_dedupes_repeated_intents():
    loader = _StubLoader({"a": {"fn": _identity_fn(1)}})
    plan = compose(loader, ["a", "a", "a"])
    assert plan["nodes"] == ["a"]


def test_execute_plan_distinguishes_empty_from_cycle():
    loader = _StubLoader({})
    plan = compose(loader, [])
    result = execute_plan(loader, plan)
    assert "empty plan" in result["errors"]["_plan"]


def test_has_cycle_detects_loop():
    loader = _StubLoader({
        "a": {"inputs": {"v": "int"}, "outputs": {"v": "int"}, "fn": _identity_fn(0)},
        "b": {"inputs": {"v": "int"}, "outputs": {"v": "int"}, "fn": _identity_fn(0)},
    })
    plan = compose(loader, ["a", "b"])
    assert has_cycle(plan)
    assert topological_order(plan) == []


# ── auto_select_organs ───────────────────────────────────────────────────────

def test_auto_select_returns_picked_intents():
    loader = _LiveLoaderStub({
        "weather_check":    "fetches current weather",
        "currency_convert": "converts currencies",
    })
    router = _StubRouter('{"intents": ["weather_check"]}')
    picked = auto_select_organs(loader, "weather in Lagos", router)
    assert picked == ["weather_check"]
    assert "weather_check" in router.last_prompt
    assert "currency_convert" in router.last_prompt


def test_auto_select_filters_unknown_intents():
    loader = _LiveLoaderStub({"weather_check": "weather"})
    router = _StubRouter('{"intents": ["weather_check", "fictional_organ"]}')
    assert auto_select_organs(loader, "weather", router) == ["weather_check"]


def test_auto_select_strips_code_fences():
    loader = _LiveLoaderStub({"translate_text": "translate"})
    router = _StubRouter('```json\n{"intents":["translate_text"]}\n```')
    assert auto_select_organs(loader, "translate hello", router) == ["translate_text"]


def test_auto_select_returns_empty_on_garbage():
    loader = _LiveLoaderStub({"x": "x"})
    assert auto_select_organs(loader, "do", _StubRouter("not-json")) == []


def test_auto_select_returns_empty_without_router():
    loader = _LiveLoaderStub({"x": "x"})
    assert auto_select_organs(loader, "do", None) == []


def test_auto_select_respects_max_organs_cap():
    loader = _LiveLoaderStub({f"o{i}": f"organ {i}" for i in range(10)})
    router = _StubRouter('{"intents": ["o0","o1","o2","o3","o4","o5"]}')
    picked = auto_select_organs(loader, "do many things", router, max_organs=3)
    assert picked == ["o0", "o1", "o2"]


def test_auto_select_dedupes():
    loader = _LiveLoaderStub({"a": "a"})
    router = _StubRouter('{"intents": ["a","a","a"]}')
    assert auto_select_organs(loader, "x", router) == ["a"]
