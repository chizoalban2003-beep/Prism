"""
prism_tool_loop.py
==================
RFC step 3 (docs/rfc-agentic-loop.md): the bounded LLM→policy→organ tool
loop that runs where the old fallback single-shot-classified into one
intent label. Shadow rollout: PrismAgent invokes it only when routing
lands on ``general_chat`` — the "shrug" outcome — so every turn the
regex table already handles stays instant, free, and offline-capable.

Policy comes from two places, deliberately:

* **The user** — the ``[tool_loop]`` config section: ``enabled``,
  ``max_hops``, ``max_risk`` (tool-belt ceiling), ``deny`` (organs the
  loop may never call), ``allow_only`` (restrict the belt to a set).
* **The Prism's own self-preservation** — mechanical rules that don't
  ask the model's opinion: every proposed call still passes
  dispatch_organ's L1 constitution / L2 approval+rate / L3 bud gates
  unchanged; the default belt excludes critical-risk organs; provider
  budget ceilings apply per hop via the router; and after any tool
  returns third-party content (web, email, documents, clipboard) the
  remaining hops get a low-risk belt with every outbound/exfil-capable
  organ removed — the classic injection→exfiltration chain is cut
  structurally, not judged.

The loop only ever *proposes*; the gate disposes.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Callable, Optional

from prism_responses import CardType, PrismCard, text_card

logger = logging.getLogger(__name__)

# Organs whose output is third-party content an attacker could author.
UNTRUSTED_SOURCE_INTENTS = frozenset({
    "web_search", "web_scrape", "wikipedia_lookup", "news_headlines",
    "document_read", "dropbox_fetch", "gdrive_search", "notion_query",
    "file_read", "clipboard_read", "meeting_brief", "finance_summary",
})

# Organs that can carry data out of the machine or actuate it — denied
# for the rest of the loop once tainted (note web_search/web_scrape
# appear here too: a query string is an exfiltration channel).
OUTBOUND_INTENTS = frozenset({
    "email_send", "phone_call", "telegram_send", "discord_send",
    "github_issue", "shell_run", "file_write", "calendar_write",
    "web_search", "web_scrape", "smart_home_control", "system_power",
    "system_lock", "mesh_orchestrate", "mesh_register", "policy_update",
    "veax_control",
})

_SYSTEM_PROMPT = (
    "You are PRISM, a local-first personal assistant. You may call the "
    "provided tools when they help answer the user; pass each tool one "
    "clear natural-language request in its 'message' argument. Prefer "
    "the cheapest safe tool, and never call a tool you don't need. If "
    "no tool is needed, just answer directly — conversationally and "
    "concisely, one or two short sentences for casual chat."
)


class ToolLoop:
    def __init__(
        self,
        router: Any,
        organ_loader: Any,
        dispatch_fn: Callable[..., Optional[PrismCard]],
        config: dict | None = None,
    ) -> None:
        self._router = router
        self._loader = organ_loader
        self._dispatch = dispatch_fn
        self._cfg = dict(config or {})

    # ── Tool belt (user policy + taint rule) ─────────────────────────────

    def _belt(self, tainted: bool) -> list[dict]:
        max_risk = "low" if tainted else str(self._cfg.get("max_risk", "high"))
        tools = self._loader.organ_tool_schemas(max_risk=max_risk)
        deny = set(self._cfg.get("deny", []) or [])
        if tainted:
            deny |= OUTBOUND_INTENTS
        allow_only = set(self._cfg.get("allow_only", []) or [])
        out = []
        for t in tools:
            name = t["function"]["name"]
            if name in deny:
                continue
            if allow_only and name not in allow_only:
                continue
            out.append(t)
        return out

    # ── The loop ─────────────────────────────────────────────────────────

    def run(self, agent: Any, message: str, ctx: dict,
            max_hops: int | None = None) -> Optional[PrismCard]:
        """Run the loop; None means "use the old path" (disabled, no
        tools, or no LLM backend reachable). ``max_hops`` overrides the
        configured budget — folded chain/composer triggers (RFC step 5)
        pass the larger multistep allowance."""
        if not self._cfg.get("enabled", True):
            return None
        if self._router is None or self._loader is None:
            return None
        if max_hops is None:
            max_hops = int(self._cfg.get("max_hops", 3))
        max_tokens = int(self._cfg.get("max_tokens", 700))
        tainted = False
        belt = self._belt(tainted)
        if not belt:
            return None

        messages: list[dict] = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": message},
        ]

        for _hop in range(max_hops):
            res = self._router.call_tools(
                messages, belt, max_tokens=max_tokens, source="tool_loop")
            if res.get("model") == "none":
                # No backend reachable — offline behaviour is unchanged
                # by construction: hand back to the old path.
                return None
            calls = res.get("tool_calls") or []
            if not calls:
                text = (res.get("content") or "").strip()
                return text_card(text, "PRISM") if text else None

            # Echo the assistant turn (OpenAI shape) so the next hop has it.
            messages.append({
                "role": "assistant",
                "content": res.get("content") or None,
                "tool_calls": [{
                    "id": c["id"], "type": "function",
                    "function": {"name": c["name"],
                                 "arguments": json.dumps(c["arguments"])},
                } for c in calls],
            })

            belt_names = {t["function"]["name"] for t in belt}
            for call in calls:
                name = call["name"]
                organ_msg = str((call.get("arguments") or {}).get("message")
                                or message)
                if name not in belt_names:
                    # Model proposed outside its belt (or belt shrank
                    # after taint) — refuse mechanically.
                    logger.info("[tool_loop] denied off-belt call: %s", name)
                    messages.append({
                        "role": "tool", "tool_call_id": call["id"],
                        "content": f"'{name}' is not permitted by policy "
                                   "in this context.",
                    })
                    continue
                card = self._dispatch(agent, name, organ_msg, dict(ctx))
                if card is None:
                    messages.append({
                        "role": "tool", "tool_call_id": call["id"],
                        "content": f"No organ named '{name}' is loaded.",
                    })
                    continue
                if getattr(card, "card_type", None) == CardType.APPROVAL:
                    # requires_approval organ — surface the approval card
                    # as the turn's outcome; the existing approve flow
                    # runs the organ if the user consents.
                    return card
                result_text = f"{card.title}\n{card.body or ''}"[:1200]
                messages.append({
                    "role": "tool", "tool_call_id": call["id"],
                    "content": result_text,
                })
                if name in UNTRUSTED_SOURCE_INTENTS and not tainted:
                    tainted = True
                    belt = self._belt(tainted=True)
                    logger.info(
                        "[tool_loop] tainted by %s — belt reduced to "
                        "%d low-risk tools, outbound denied", name, len(belt))

        # Hop cap reached — force a final answer, no more tools.
        messages.append({
            "role": "user",
            "content": "Answer now using what you have — no more tool calls.",
        })
        res = self._router.call_tools(
            messages, None, max_tokens=max_tokens, source="tool_loop")
        text = (res.get("content") or "").strip()
        return text_card(text, "PRISM") if text else None
