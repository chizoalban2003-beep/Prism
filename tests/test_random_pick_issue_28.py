"""random_pick intent + organ for issue #28 bug 20.

Live test: ``flip a coin`` and ``roll a die`` both returned "Build new
organ?" approval cards. Adds a random_pick intent and a stdlib-only
organ that handles coin flips, dice rolls, random numbers, and
"choose between X or Y" picks.
"""
from __future__ import annotations

import importlib.util

from prism_intents import INTENTS
from prism_routing import route_intent


def _route(message: str) -> str:
    return route_intent(message, INTENTS, lambda _m: None)


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, f"organs/{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestRoutingRandomPick:
    def test_flip_a_coin(self):
        assert _route("flip a coin") == "random_pick"

    def test_toss_a_coin(self):
        assert _route("toss a coin") == "random_pick"

    def test_roll_a_die(self):
        assert _route("roll a die") == "random_pick"

    def test_roll_dice(self):
        assert _route("roll dice") == "random_pick"

    def test_roll_2d6(self):
        assert _route("roll 2d6") == "random_pick"

    def test_random_number_between(self):
        assert _route("random number between 1 and 100") == "random_pick"

    def test_pick_a_random_number(self):
        assert _route("pick a random number") == "random_pick"


class TestNoOverreach:
    def test_play_music_unaffected(self):
        assert _route("play some music") != "random_pick"

    def test_roll_back_changes(self):
        # "roll back" is unrelated to dice — must not match.
        assert _route("roll back the last change") != "random_pick"


class TestRandomPickOrgan:
    organ = _load("random_pick")

    def test_coin_returns_heads_or_tails(self):
        card = self.organ.execute("random_pick", "flip a coin", {})
        assert card.card_data["kind"] == "coin"
        assert card.card_data["result"] in ("Heads", "Tails")

    def test_die_returns_1_to_6(self):
        card = self.organ.execute("random_pick", "roll a die", {})
        assert card.card_data["kind"] == "dice"
        value = int(card.card_data["result"])
        assert 1 <= value <= 6

    def test_d20_returns_1_to_20(self):
        card = self.organ.execute("random_pick", "roll a d20", {})
        assert card.card_data["kind"] == "dice"
        value = int(card.card_data["result"])
        assert 1 <= value <= 20

    def test_2d6_sums_two_rolls(self):
        card = self.organ.execute("random_pick", "roll 2d6", {})
        assert card.card_data["kind"] == "dice"
        assert len(card.card_data["rolls"]) == 2
        total = int(card.card_data["result"])
        assert 2 <= total <= 12

    def test_random_number_in_range(self):
        card = self.organ.execute(
            "random_pick", "give me a random number between 50 and 60", {},
        )
        assert card.card_data["kind"] == "number"
        value = int(card.card_data["result"])
        assert 50 <= value <= 60

    def test_default_number_range(self):
        card = self.organ.execute("random_pick", "pick a random number", {})
        value = int(card.card_data["result"])
        assert 1 <= value <= 100
