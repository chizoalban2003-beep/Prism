"""calc_eval word-form fix for issue #28 bug 38.

Live test surfaced three forms the intent regex claimed but the
organ couldn't evaluate:

* ``5 squared`` → "invalid syntax"
* ``5 cubed`` → fell through to LLM (which returned "115" — wrong)
* ``10% of 200`` / ``100 percent of 50`` → "invalid syntax"

The intent regex was already grabbing these (``%`` is in the char
class, and ``\\d squared/cubed`` would match if extended), but the
organ's word-op map only knew about ``plus/minus/times/divided by/
modulo/to the power of`` and didn't normalise ``squared``, ``cubed``,
or ``X% of Y``.

Fix: add three preprocessors to ``_extract_expression``:

* ``_SQUARED_RE`` rewrites ``N squared`` to ``((N)**2)``
* ``_CUBED_RE`` rewrites ``N cubed`` to ``((N)**3)``
* ``_PERCENT_OF_RE`` rewrites ``X% of Y`` / ``X percent of Y`` to
  ``((X/100)*Y)``

All three stay infix arithmetic, so the AST walker still only sees
operators it already allows.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "_calc_organ",
    Path(__file__).resolve().parent.parent / "organs" / "calc_eval.py",
)
_organ = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_organ)


def _eval(message: str) -> float:
    expr = _organ._extract_expression(message)
    return _organ._safe_eval(expr)


class TestSquaredCubed:
    def test_five_squared(self):
        assert _eval("5 squared") == 25

    def test_five_cubed(self):
        assert _eval("5 cubed") == 125

    def test_decimal_squared(self):
        assert _eval("3.14 squared") == 3.14 ** 2

    def test_what_is_seven_squared(self):
        assert _eval("what is 7 squared") == 49

    def test_calculate_two_cubed(self):
        assert _eval("calculate 2 cubed") == 8


class TestPercentOf:
    def test_ten_percent_of_two_hundred(self):
        # The reported bug.
        assert _eval("10% of 200") == 20

    def test_one_hundred_percent_of_fifty(self):
        assert _eval("100 percent of 50") == 50

    def test_twenty_five_percent_of_eighty(self):
        assert _eval("25% of 80") == 20

    def test_decimal_percent(self):
        assert _eval("12.5% of 200") == 25

    def test_with_what_is_prelude(self):
        assert _eval("what is 30% of 60") == 18


class TestRegression:
    """Original forms must keep working."""

    def test_exponent(self):
        assert _eval("2 ** 10") == 1024

    def test_floor_div(self):
        assert _eval("100 // 3") == 33

    def test_mixed(self):
        assert _eval("12 + 8 * 3") == 36

    def test_square_root(self):
        assert _eval("square root of 144") == 12

    def test_word_plus(self):
        assert _eval("5 plus 7") == 12
