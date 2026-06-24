"""Bundled organ: random_pick — coin flip, dice roll, random number/choice."""
import re

ORGAN_META = {
    "intent":      "random_pick",
    "description": "flip a coin, roll dice, pick a random number, or choose between options",
    "version":     "1.0",
    "capabilities": [],
    "inputs":  {},
    "outputs": {"kind": "str", "result": "str"},
}

ORGAN_POLICY = {
    "risk_level":        "low",
    "requires_approval": False,
    "irreversible":      False,
    "max_per_session":   None,
}


def _parse_dice(msg: str) -> tuple[int, int] | None:
    m = re.search(r"\b(\d*)d(\d+)\b", msg, re.IGNORECASE)
    if m:
        n = int(m.group(1)) if m.group(1) else 1
        sides = int(m.group(2))
        if 1 <= n <= 100 and 2 <= sides <= 1000:
            return n, sides
    if re.search(r"\b(?:roll|throw)\s+(?:a\s+|me\s+)?(?:die|dice)\b", msg, re.IGNORECASE):
        return 1, 6
    return None


def _parse_range(msg: str) -> tuple[int, int] | None:
    m = re.search(
        r"(?:between|from)\s+(-?\d+)\s+(?:and|to)\s+(-?\d+)",
        msg, re.IGNORECASE,
    )
    if m:
        lo, hi = int(m.group(1)), int(m.group(2))
        if lo > hi:
            lo, hi = hi, lo
        return lo, hi
    return None


def _parse_choice(msg: str) -> list[str] | None:
    m = re.search(
        r"(?:between|from|among)\s+(.+?)(?:\s+for me)?\s*[.?!]?\s*$",
        msg, re.IGNORECASE,
    )
    if not m:
        return None
    raw = m.group(1)
    parts = re.split(r"\s*,\s*|\s+or\s+|\s+and\s+", raw)
    parts = [p.strip(" .?!\"'") for p in parts if p.strip(" .?!\"'")]
    return parts if len(parts) >= 2 else None


def execute(intent: str, message: str, ctx: dict):
    import random

    from prism_responses import text_card

    msg = (message or "").strip()
    rng = random.SystemRandom()

    if re.search(r"\b(?:flip|toss)\b.*\bcoin\b|\bcoin\s+(?:flip|toss)\b", msg, re.IGNORECASE):
        result = rng.choice(["Heads", "Tails"])
        card = text_card(f"{result}.", "Coin flip")
        card.card_data.update({"kind": "coin", "result": result})
        return card

    dice = _parse_dice(msg)
    if dice:
        n, sides = dice
        rolls = [rng.randint(1, sides) for _ in range(n)]
        total = sum(rolls)
        if n == 1:
            body = f"d{sides}: {rolls[0]}"
        else:
            body = f"{n}d{sides}: {' + '.join(str(r) for r in rolls)} = {total}"
        card = text_card(body, "Dice roll")
        card.card_data.update({"kind": "dice", "result": str(total), "rolls": rolls})
        return card

    rng_range = _parse_range(msg)
    if rng_range and re.search(r"\bnumber\b|\binteger\b", msg, re.IGNORECASE):
        lo, hi = rng_range
        value = rng.randint(lo, hi)
        card = text_card(f"{value} (from {lo}–{hi})", "Random number")
        card.card_data.update({"kind": "number", "result": str(value)})
        return card

    choice = _parse_choice(msg)
    if choice and re.search(r"\bchoice\b|\bpick\b|\bchoose\b", msg, re.IGNORECASE):
        picked = rng.choice(choice)
        card = text_card(f"{picked}.", "Random pick")
        card.card_data.update({"kind": "choice", "result": picked, "options": choice})
        return card

    value = rng.randint(1, 100)
    card = text_card(f"{value} (default 1–100)", "Random number")
    card.card_data.update({"kind": "number", "result": str(value)})
    return card
