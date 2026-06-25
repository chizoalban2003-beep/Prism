"""calc_eval mixed-script extraction fix for issue #28 bug 47.

Live test: messages that contain an arithmetic expression embedded in
non-ASCII text routed correctly to calc_eval (the intent regex matched
``2 + 3``) but then blew up inside ``_safe_eval`` with::

    I couldn't evaluate that as arithmetic:
    invalid syntax (<unknown>, line 1).

Examples that surfaced the bug:

* ``"日本語テスト 計算して 2 + 3 を"``
* ``"hola 🌮 cuanto es 7 + 8"``

The reason: ``_extract_expression`` only substituted ASCII English
words like "calculate"/"plus" — it had no step that drops leftover
non-arithmetic characters. So ``ast.parse`` saw kana or an emoji code
point and raised ``invalid character (U+1F32E)`` or ``invalid syntax``.

Fix: after all language→symbol substitutions, replace any character
outside ``[0-9 . + - * / % ( ) whitespace]`` with a space. The cleaned
expression goes into the AST walker safely; the user gets the answer
instead of a parse error.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "_calc_eval_organ",
    Path(__file__).resolve().parent.parent / "organs" / "calc_eval.py",
)
_organ = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_organ)


class TestExtractStripsForeignScript:
    def test_japanese_math_extracts_to_digits_only(self):
        # The reported repro.
        expr = _organ._extract_expression("日本語テスト 計算して 2 + 3 を")
        # Whatever the extractor returns, it must be a parseable arithmetic
        # expression and contain only the relevant operators / digits.
        assert expr.replace(" ", "") == "2+3", expr

    def test_emoji_math_extracts_cleanly(self):
        expr = _organ._extract_expression("hola 🌮 cuanto es 7 + 8")
        assert expr.replace(" ", "") == "7+8", expr

    def test_cyrillic_math(self):
        expr = _organ._extract_expression("привет посчитай 12 * 9 пожалуйста")
        assert expr.replace(" ", "") == "12*9", expr

    def test_arabic_math(self):
        expr = _organ._extract_expression("حساب 50 + 25 من فضلك")
        assert expr.replace(" ", "") == "50+25", expr


class TestExtractPreservesAscii:
    """Sanity — the strip step must not break existing English flows."""

    def test_calculate_then_expression(self):
        assert _organ._extract_expression("calculate 12 * 7") == "12 * 7"

    def test_what_is_word_form(self):
        # "what is" stripped by _LEAD_RE; "plus" → "+"
        assert _organ._extract_expression("what is 5 plus 3").replace(" ", "") == "5+3"

    def test_square_root_word(self):
        # sqrt of 144 → ((144)**0.5)
        out = _organ._extract_expression("what is the square root of 144")
        assert "0.5" in out and "144" in out

    def test_factorial_bang(self):
        # 5! → 120 (handled by _BANG_RE before whitelist)
        assert _organ._extract_expression("compute 5!") == "120"

    def test_percent_of(self):
        # 20% of 50 → ((20/100)*50)
        out = _organ._extract_expression("20 percent of 50")
        assert "100" in out and "50" in out and "20" in out


class TestExecuteEndToEnd:
    """End-to-end: execute() must return the answer card, not the
    calculator-parse-error card."""

    def test_japanese_round_trip_returns_answer(self):
        card = _organ.execute("calc_eval", "日本語テスト 計算して 2 + 3 を", {})
        # The reported bug: body used to be "I couldn't evaluate that as
        # arithmetic: invalid syntax". Must now end with " = 5".
        assert card.body.endswith("= 5"), card.body
        assert "couldn't evaluate" not in card.body

    def test_emoji_round_trip_returns_answer(self):
        card = _organ.execute("calc_eval", "hola 🌮 cuanto es 7 + 8", {})
        assert card.body.endswith("= 15"), card.body
        assert "invalid character" not in card.body
