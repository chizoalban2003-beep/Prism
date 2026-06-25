"""unit_convert routing fix for issue #28 bug 35.

Live test: ``convert json to yaml`` was routed to the unit converter
which then returned "Could not parse conversion request" with an
examples list of physical units (km / fahrenheit / kg). The user
wanted a data-format conversion, which should fall through to organ
synthesis.

Root cause: the broad fallback ``(?:convert|how many|how much) .*
(?:to|in|into)`` had no domain qualifier and claimed any
``convert X to Y`` shape.

Fix: add a negative lookahead listing common data-format and
programming-language tokens. Those messages now fall through to the
synthesis pipeline.
"""
from __future__ import annotations

import re

from prism_intents import INTENTS


def _route(message: str) -> str:
    for pattern, intent in INTENTS:
        if re.search(pattern, message, re.IGNORECASE):
            return intent
    return "_NO_MATCH_"


class TestDataFormatsFallThrough:
    """Format conversions must not be claimed by unit_convert."""

    def test_json_to_yaml(self):
        # The reported bug.
        assert _route("convert json to yaml") != "unit_convert"

    def test_yaml_to_json(self):
        assert _route("convert yaml to json") != "unit_convert"

    def test_xml_to_json(self):
        assert _route("convert xml to json") != "unit_convert"

    def test_csv_to_tsv(self):
        assert _route("convert csv to tsv") != "unit_convert"

    def test_html_to_markdown(self):
        assert _route("convert html to markdown") != "unit_convert"

    def test_python_to_javascript(self):
        assert _route("convert python to javascript") != "unit_convert"

    def test_encode_to_base64(self):
        assert _route("encode hello to base64") != "unit_convert"


class TestUnitsStillRoute:
    """Real unit conversions must keep working."""

    def test_km_to_miles(self):
        assert _route("100 km to miles") == "unit_convert"

    def test_how_many_feet_in_a_mile(self):
        assert _route("how many feet in a mile") == "unit_convert"

    def test_convert_kg_to_pounds(self):
        assert _route("convert 5 kg to pounds") == "unit_convert"

    def test_convert_fahrenheit_to_celsius(self):
        assert _route("convert 72 fahrenheit to celsius") == "unit_convert"

    def test_how_much_cm_in_inches(self):
        assert _route("how much is 10 cm in inches") == "unit_convert"
