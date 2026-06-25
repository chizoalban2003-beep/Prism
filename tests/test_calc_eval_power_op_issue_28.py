"""calc_eval intent fix for issue #28 bug 37.

Live test: ``2 ** 10`` returned an empty body — the calc_eval intent
regex character class ``[+\\-*/×÷%^]`` only matches single-character
operators, so ``**`` (Python exponent) fell through. The calc_eval
organ itself supports ``**`` evaluation; the gap was purely in
routing. ``100 // 3`` (floor-division) had the same shape.

Fix: replace the bare ``*`` / ``/`` in the char class with a
disjunction ``(?:\\*\\*|//|[+\\-*/×÷%^])`` so the two-char operators
match.
"""
from __future__ import annotations

import re

from prism_intents import INTENTS


def _route(message: str) -> str:
    for pattern, intent in INTENTS:
        if re.search(pattern, message, re.IGNORECASE):
            return intent
    return "_NO_MATCH_"


class TestTwoCharOperators:
    def test_exponent_routes(self):
        # The reported bug.
        assert _route("2 ** 10") == "calc_eval"

    def test_floor_division_routes(self):
        assert _route("100 // 3") == "calc_eval"

    def test_exponent_with_prelude(self):
        assert _route("what is 2 ** 8") == "calc_eval"


class TestSingleCharStillRoute:
    """Don't regress the original single-char operators."""

    def test_plus(self):
        assert _route("12 + 8") == "calc_eval"

    def test_minus(self):
        assert _route("7 - 2") == "calc_eval"

    def test_star(self):
        assert _route("3 * 4") == "calc_eval"

    def test_divide(self):
        assert _route("100 / 4") == "calc_eval"

    def test_modulo(self):
        assert _route("17 % 5") == "calc_eval"

    def test_caret(self):
        # Caret is commonly used as exponent — keep it routing.
        assert _route("2 ^ 8") == "calc_eval"

    def test_unicode_multiply(self):
        assert _route("3 × 7") == "calc_eval"

    def test_unicode_divide(self):
        assert _route("100 ÷ 4") == "calc_eval"

    def test_decimal_operand(self):
        assert _route("3.14 * 2") == "calc_eval"

    def test_word_form(self):
        assert _route("5 plus 7") == "calc_eval"

    def test_square_root(self):
        assert _route("square root of 144") == "calc_eval"
