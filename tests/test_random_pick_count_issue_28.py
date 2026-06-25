"""Coin/dice count fix for issue #28 bug 33.

Live test: ``flip 5 coins`` was routed to calc_eval and returned
"invalid syntax" because the random_pick intent regex required
``(?:flip|toss) (?:a |me )?coin`` — bare or "a"/"me" only, never a
count or plural. ``roll 3 dice`` had the same shape.

Fix lives in two places:

* ``prism_intents.INTENTS`` — widen the prefix optional-group to also
  accept ``\\d+`` or number-word (two..ten) and accept ``coin`` or
  ``coins``.
* ``organs/random_pick.execute`` — parse the count and produce N
  flips / N rolls instead of a single sample.
"""
from __future__ import annotations

import importlib.util
import re
from pathlib import Path

from prism_intents import INTENTS

_spec = importlib.util.spec_from_file_location(
    "_random_pick_organ",
    Path(__file__).resolve().parent.parent / "organs" / "random_pick.py",
)
_organ = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_organ)


def _route(message: str) -> str:
    for pattern, intent in INTENTS:
        if re.search(pattern, message, re.IGNORECASE):
            return intent
    return "_NO_MATCH_"


class TestIntentRouting:
    def test_flip_5_coins_routes(self):
        # The reported bug — was hitting calc_eval (invalid syntax).
        assert _route("flip 5 coins") == "random_pick"

    def test_flip_two_coins_routes(self):
        assert _route("flip two coins") == "random_pick"

    def test_toss_3_coins_routes(self):
        assert _route("toss 3 coins") == "random_pick"

    def test_flip_a_coin_still_routes(self):
        # Backwards compat.
        assert _route("flip a coin") == "random_pick"

    def test_roll_3_dice_routes(self):
        assert _route("roll 3 dice") == "random_pick"

    def test_roll_two_dice_routes(self):
        assert _route("roll two dice") == "random_pick"

    def test_roll_a_die_still_routes(self):
        assert _route("roll a die") == "random_pick"


class TestOrganCounting:
    def _run(self, message: str) -> dict:
        return _organ.execute("random_pick", message, {}).card_data

    def test_flip_5_coins_produces_five_results(self):
        data = self._run("flip 5 coins")
        assert data["kind"] == "coin"
        assert len(data["flips"]) == 5
        assert data["heads"] + data["tails"] == 5

    def test_flip_two_coins_produces_two_results(self):
        data = self._run("flip two coins")
        assert len(data["flips"]) == 2

    def test_flip_a_coin_single_result(self):
        # Single-coin path keeps the legacy shape (no flips list).
        data = self._run("flip a coin")
        assert data["kind"] == "coin"
        assert data["result"] in ("Heads", "Tails")

    def test_roll_3_dice_produces_three_rolls(self):
        data = self._run("roll 3 dice")
        assert data["kind"] == "dice"
        assert len(data["rolls"]) == 3
        assert all(1 <= r <= 6 for r in data["rolls"])

    def test_roll_two_dice_produces_two_rolls(self):
        data = self._run("roll two dice")
        assert len(data["rolls"]) == 2

    def test_roll_3d20_dice_notation_still_works(self):
        # Old d-notation path must not regress.
        data = self._run("roll 3d20")
        assert len(data["rolls"]) == 3
        assert all(1 <= r <= 20 for r in data["rolls"])

    def test_count_capped_at_100(self):
        # Don't let a user ask for a million coin flips.
        data = self._run("flip 1000 coins")
        assert len(data["flips"]) == 100
