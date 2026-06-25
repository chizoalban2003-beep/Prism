"""currency_convert display fix for issue #28 bug 42.

Live test: ``50 GBP to JPY`` rendered as ``1.065e+04 JPY``. The
original ``:,.4g`` format collapses to scientific notation for any
value >= 1e4, which is awful UX for normal yen / won / rupiah amounts.

Fix: ``_fmt_amount`` uses fixed-point with comma separator for
abs(value) >= 1, falling back to 6-sig-digit ``g`` only for sub-unit
sized conversions.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "_cc_organ",
    Path(__file__).resolve().parent.parent / "organs" / "currency_convert.py",
)
_organ = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_organ)


class TestFormatLargeAmount:
    def test_jpy_amount_not_scientific(self):
        # The reported bug — 50 GBP * 212.94 ≈ 10647 must not collapse.
        out = _organ._fmt_amount(10647.05)
        assert "e" not in out.lower()
        assert out == "10,647.05"

    def test_million_amount_not_scientific(self):
        out = _organ._fmt_amount(1_234_567.89)
        assert "e" not in out.lower()
        assert out == "1,234,567.89"

    def test_thousand_uses_comma(self):
        out = _organ._fmt_amount(1500.5)
        assert out == "1,500.50"


class TestFormatNormalAmount:
    def test_eur_amount(self):
        out = _organ._fmt_amount(88.08)
        assert out == "88.08"

    def test_one_unit(self):
        out = _organ._fmt_amount(1.0)
        assert out == "1.00"


class TestFormatSubUnit:
    def test_tiny_amount_keeps_precision(self):
        out = _organ._fmt_amount(0.000813)
        # Below 1 we want significant digits, not "0.00".
        assert "0.000813" in out or out == "0.000813"

    def test_half_unit(self):
        out = _organ._fmt_amount(0.5)
        assert "0.5" in out
