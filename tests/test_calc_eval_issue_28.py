"""calc_eval intent + organ for issue #28 bug 26.

Live tests revealed PRISM has no built-in arithmetic. The agent
either tried to synthesize a brand-new organ for "calculate
1234 * 56" (wasteful) or routed "what is the square root of 144"
to wikipedia_lookup, which returned the article on "New Jerusalem".

Fix: a calc_eval intent above wikipedia_lookup, backed by an
AST-based safe arithmetic evaluator. No eval(), no name lookups,
no function calls — only BinOp, UnaryOp, and numeric Constants
with a whitelist of operators (+ - * / // % **).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

from prism_intents import INTENTS
from prism_routing import route_intent

_spec = importlib.util.spec_from_file_location(
    "_calc_eval_organ",
    Path(__file__).resolve().parent.parent / "organs" / "calc_eval.py",
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
execute = _mod.execute
_extract_expression = _mod._extract_expression
_safe_eval = _mod._safe_eval


def _route(msg: str) -> str:
    return route_intent(msg, INTENTS, lambda _m: None)


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

class TestRouting:
    def test_calculate_with_star(self):
        assert _route("calculate 1234 * 56") == "calc_eval"

    def test_what_is_with_plus(self):
        assert _route("what is 2 + 2") == "calc_eval"

    def test_square_root(self):
        assert _route("what is the square root of 144") == "calc_eval"

    def test_word_plus(self):
        assert _route("what is 5 plus 3") == "calc_eval"

    def test_word_divided_by(self):
        assert _route("what is 10 divided by 4") == "calc_eval"

    def test_compute_expression(self):
        assert _route("compute 3 * (4 + 5)") == "calc_eval"


class TestRoutingDoesNotOverreach:
    def test_what_is_paris_still_wikipedia(self):
        # No numeric operator — should not hit calc_eval.
        assert _route("what is paris") != "calc_eval"

    def test_word_with_number_in_prose_not_calc(self):
        # "I have 5 apples" — no operator.
        assert _route("I have 5 apples in my fridge") != "calc_eval"

    def test_remind_me_in_5_minutes_not_calc(self):
        # "remind me in 5 minutes" — not arithmetic.
        assert _route("remind me in 5 minutes to call john") != "calc_eval"


# ---------------------------------------------------------------------------
# Expression extraction
# ---------------------------------------------------------------------------

class TestExtraction:
    def test_lead_in_stripped(self):
        assert _extract_expression("calculate 1234 * 56") == "1234 * 56"

    def test_what_is_stripped(self):
        assert _extract_expression("what is 2 + 2") == "2 + 2"

    def test_question_mark_stripped(self):
        assert _extract_expression("what is 2 + 2?") == "2 + 2"

    def test_word_operators_converted(self):
        assert _extract_expression("5 plus 3") == "5 + 3"
        assert _extract_expression("10 divided by 4") == "10 / 4"
        assert _extract_expression("6 times 7") == "6 * 7"

    def test_square_root_converted(self):
        assert "**0.5" in _extract_expression("square root of 144")

    def test_caret_to_pow(self):
        assert _extract_expression("2 ^ 10") == "2 ** 10"


# ---------------------------------------------------------------------------
# Safe evaluator
# ---------------------------------------------------------------------------

class TestSafeEval:
    def test_addition(self):
        assert _safe_eval("2 + 2") == 4

    def test_multiplication(self):
        assert _safe_eval("1234 * 56") == 69104

    def test_parens(self):
        assert _safe_eval("3 * (4 + 5)") == 27

    def test_power(self):
        assert _safe_eval("2 ** 10") == 1024

    def test_floor_div(self):
        assert _safe_eval("17 // 5") == 3

    def test_modulo(self):
        assert _safe_eval("17 % 5") == 2

    def test_unary_minus(self):
        assert _safe_eval("-5 + 3") == -2

    def test_square_root_form(self):
        assert _safe_eval("(144)**0.5") == 12.0

    def test_rejects_name_lookup(self):
        # No name access — would otherwise call __import__ etc.
        import pytest
        with pytest.raises((ValueError, NameError, SyntaxError)):
            _safe_eval("__import__('os').system('ls')")

    def test_rejects_function_call(self):
        import pytest
        with pytest.raises((ValueError, SyntaxError)):
            _safe_eval("abs(-5)")

    def test_rejects_huge_exponent(self):
        import pytest
        with pytest.raises(ValueError):
            _safe_eval("2 ** 99999")


# ---------------------------------------------------------------------------
# End-to-end execute()
# ---------------------------------------------------------------------------

class TestExecute:
    def test_calculate_returns_result(self):
        card = execute("calc_eval", "calculate 1234 * 56", {})
        assert card.title == "Calculator"
        assert "69104" in card.body

    def test_square_root_144(self):
        card = execute("calc_eval", "what is the square root of 144", {})
        assert "12" in card.body
        # Must NOT have returned a Wikipedia article.
        assert "Wikipedia" not in card.title

    def test_division_by_zero(self):
        card = execute("calc_eval", "what is 5 / 0", {})
        assert "zero" in card.body.lower()

    def test_no_expression_explains(self):
        card = execute("calc_eval", "calculate", {})
        assert "couldn't" in card.body.lower() or "could not" in card.body.lower()
