"""calc_eval log/factorial fix for issue #28 bug 39.

Live test surfaced five forms that fell through to the LLM Chat fallback
(burning budget for pure math):

* ``log of 1000`` — should be 3
* ``ln of 10`` — should be ln(10) ≈ 2.302585
* ``log base 2 of 8`` — should be 3
* ``factorial of 5`` — should be 120
* ``5!`` — should be 120

The calc_eval AST walker is intentionally function-call-free, so the
fix preprocesses these forms into literal numeric values (math.log10,
math.log, math.factorial) before the AST sees the expression.
"""
from __future__ import annotations

import importlib.util
import math
import re
from pathlib import Path

from prism_intents import INTENTS

_spec = importlib.util.spec_from_file_location(
    "_calc_organ",
    Path(__file__).resolve().parent.parent / "organs" / "calc_eval.py",
)
_organ = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_organ)


def _eval(message: str) -> float:
    expr = _organ._extract_expression(message)
    return _organ._safe_eval(expr)


def _route(message: str) -> str:
    for pattern, intent in INTENTS:
        if re.search(pattern, message, re.IGNORECASE):
            return intent
    return "_NO_MATCH_"


class TestLogRouting:
    def test_log_of_routes(self):
        assert _route("log of 1000") == "calc_eval"

    def test_ln_of_routes(self):
        assert _route("ln of 10") == "calc_eval"

    def test_log_base_routes(self):
        assert _route("log base 2 of 8") == "calc_eval"

    def test_factorial_of_routes(self):
        assert _route("factorial of 5") == "calc_eval"

    def test_bang_routes(self):
        assert _route("5!") == "calc_eval"

    def test_what_is_log_routes(self):
        assert _route("what is log of 100") == "calc_eval"


class TestLogEval:
    def test_log_of_1000(self):
        assert _eval("log of 1000") == 3

    def test_log_of_100(self):
        assert _eval("log of 100") == 2

    def test_ln_of_e(self):
        assert abs(_eval("ln of 10") - math.log(10)) < 1e-9

    def test_log_base_2_of_8(self):
        assert _eval("log base 2 of 8") == 3

    def test_log_base_10_of_1000(self):
        assert _eval("log base 10 of 1000") == 3


class TestFactorialEval:
    def test_factorial_of_5(self):
        assert _eval("factorial of 5") == 120

    def test_factorial_of_0(self):
        assert _eval("factorial of 0") == 1

    def test_factorial_of_10(self):
        assert _eval("factorial of 10") == 3628800

    def test_bang_5(self):
        assert _eval("5!") == 120

    def test_bang_6(self):
        assert _eval("6!") == 720

    def test_what_is_5_bang(self):
        assert _eval("what is 5!") == 120


class TestRegression:
    """Original forms must keep working."""

    def test_exponent(self):
        assert _eval("2 ** 10") == 1024

    def test_squared(self):
        assert _eval("5 squared") == 25

    def test_percent_of(self):
        assert _eval("10% of 200") == 20

    def test_square_root(self):
        assert _eval("square root of 144") == 12

    def test_word_plus(self):
        assert _eval("5 plus 7") == 12

    def test_routing_plus(self):
        assert _route("12 + 8") == "calc_eval"
