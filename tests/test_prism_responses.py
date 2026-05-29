from __future__ import annotations

from prism_responses import CardType, text_card


def test_text_card_type():
    assert text_card("x").card_type == CardType.TEXT


def test_to_json_has_type():
    assert "type" in text_card("x").to_json()
