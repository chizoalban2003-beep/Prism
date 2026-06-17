"""Standing instruction store — persistent rules taught once, applied to every relevant request."""
from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class Instruction:
    instr_id:  str
    text:      str          # plain language rule
    trigger:   str          # when this applies: "email"|"calendar"|"always"|"planning"
    active:    bool = True
    created_at:float = field(default_factory=time.time)
    use_count: int = 0

class PrismInstructions:
    """
    Persistent standing instructions from the user.

    Examples the user can set in plain language:
      "always check my calendar before planning my day"
      "whenever I get an email from my manager, flag it urgent"
      "never schedule meetings before 9am"
      "when writing emails, keep them under 3 sentences"
      "always ask for approval before sending any message"

    Instructions are retrieved by keyword/trigger match before each request
    and injected into the LLM prompt as explicit constraints.
    """

    # Keywords that map user phrases to trigger categories
    TRIGGER_MAP = {
        "email":      ["email","inbox","message","reply","send","mail"],
        "calendar":   ["calendar","schedule","meeting","appointment","event"],
        "planning":   ["plan","day","morning","briefing","task","todo"],
        "writing":    ["write","draft","compose","document","report"],
        "finance":    ["spend","buy","purchase","pay","budget","order"],
        "device":     ["file","folder","download","move","delete","run"],
        "always":     ["always","every time","whenever","all requests"],
    }

    def __init__(self, db_path: str = "~/.prism/instructions.db"):
        self._db = Path(db_path).expanduser()
        self._db.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def add(self, text: str, trigger: str = "always") -> Instruction:
        """Store a new standing instruction."""
        import hashlib
        instr_id = hashlib.sha256(text.encode()).hexdigest()[:10]
        instr = Instruction(instr_id=instr_id, text=text,
                             trigger=trigger)
        with sqlite3.connect(self._db, timeout=30.0) as c:
            c.execute("INSERT OR REPLACE INTO instructions VALUES(?,?,?,?,?,?)",
                      (instr_id, text, trigger, 1,
                       time.time(), 0))
        return instr

    def remove(self, instr_id: str) -> bool:
        with sqlite3.connect(self._db, timeout=30.0) as c:
            n = c.execute("DELETE FROM instructions WHERE id=?",
                          (instr_id,)).rowcount
        return n > 0

    def all_active(self) -> list[Instruction]:
        with sqlite3.connect(self._db, timeout=30.0) as c:
            rows = c.execute(
                "SELECT id,text,trigger,active,created_at,use_count "
                "FROM instructions WHERE active=1 "
                "ORDER BY use_count DESC").fetchall()
        return [Instruction(*r) for r in rows]

    def relevant_for(self, request: str) -> list[Instruction]:
        """
        Return instructions relevant to this specific request.
        Matches by trigger category keyword overlap, or by trigger task-slug
        words appearing in the request (so denial notes stored with
        trigger=task_slug like "send_email" still surface).
        Always returns "always" trigger instructions.
        """
        req_lower  = request.lower()
        all_instrs = self.all_active()
        relevant   = []
        for instr in all_instrs:
            if instr.trigger == "always":
                relevant.append(instr)
                continue
            keywords = self.TRIGGER_MAP.get(instr.trigger, [])
            if keywords and any(kw in req_lower for kw in keywords):
                relevant.append(instr)
                continue
            # Fallback: trigger is a task slug (e.g. "send_email") —
            # match if any non-trivial slug token appears in the request.
            slug_tokens = [t for t in instr.trigger.replace("-", "_").split("_") if len(t) > 2]
            if slug_tokens and any(t in req_lower for t in slug_tokens):
                relevant.append(instr)
        return relevant

    # Phrases that turn a one-shot "no" into a standing rule.
    _STANDING_MARKERS = (
        "never", "always", "stop ", "don't ever", "do not ever", "no more ",
        "from now on", "whenever", "every time", "any time",
    )

    @classmethod
    def classify_denial(
        cls, task: str, reason: str
    ) -> tuple[Optional[str], Optional[str]]:
        """Decide whether a denial reason is a standing rule.

        Returns ``(standing_text, trigger_category)`` when the reason reads
        like a permanent rule the user wants applied to all similar requests
        — otherwise ``(None, None)`` for a one-shot decision.

        Trigger category is inferred from TRIGGER_MAP keyword overlap on the
        reason first, then the task slug as a fallback; falls back to
        ``"always"`` only when nothing matches.
        """
        if not reason or not reason.strip():
            return None, None
        text   = reason.strip()
        lower  = text.lower()
        if not any(m in lower for m in cls._STANDING_MARKERS):
            return None, None
        # Pick category from the reason text first (user wrote what it's about).
        for cat, keywords in cls.TRIGGER_MAP.items():
            if cat == "always":
                continue
            if any(kw in lower for kw in keywords):
                return text[:300], cat
        # Fall back to scanning the originating task slug.
        task_lower = (task or "").lower()
        for cat, keywords in cls.TRIGGER_MAP.items():
            if cat == "always":
                continue
            if any(kw in task_lower for kw in keywords):
                return text[:300], cat
        return text[:300], "always"

    def prior_denials_for(self, task: str) -> list[Instruction]:
        """
        Return all active denial-derived instructions tagged with this task
        slug. Used by the approval-card builders to surface a "you denied
        this before" banner so the user (and the LLM classifier) doesn't
        re-ask without context.
        """
        if not task:
            return []
        slug = task[:80]
        with sqlite3.connect(self._db, timeout=30.0) as c:
            rows = c.execute(
                "SELECT id,text,trigger,active,created_at,use_count "
                "FROM instructions WHERE active=1 AND trigger=? "
                "ORDER BY created_at DESC", (slug,)).fetchall()
        return [Instruction(*r) for r in rows]

    def to_context_string(self, request: str) -> str:
        """
        Return relevant instructions formatted for LLM injection.
        Returns empty string if no relevant instructions.
        """
        relevant = self.relevant_for(request)
        if not relevant:
            return ""
        rules = "\n".join(f"- {i.text}" for i in relevant)
        # Increment use count
        ids = [i.instr_id for i in relevant]
        with sqlite3.connect(self._db, timeout=30.0) as c:
            for iid in ids:
                c.execute("UPDATE instructions SET use_count=use_count+1 "
                          "WHERE id=?", (iid,))
        return f"Standing instructions from the user (follow these):\n{rules}"

    def parse_from_chat(self, message: str) -> Optional[Instruction]:
        """
        Detect and store instructions from natural language.
        Triggers on: "remember:", "always:", "from now on:", "whenever", "never"
        Returns the stored instruction or None if not an instruction.
        """
        msg   = message.strip()
        lower = msg.lower()

        # Detection patterns
        prefixes = ["remember:", "remember that", "always ", "never ",
                    "from now on", "whenever ", "every time ", "make sure",
                    "don't forget", "note:", "rule:"]
        is_instruction = any(lower.startswith(p) or lower.startswith("please "+p)
                             for p in prefixes)
        if not is_instruction:
            return None

        # Infer trigger from content
        trigger = "always"
        for cat, keywords in self.TRIGGER_MAP.items():
            if any(kw in lower for kw in keywords):
                trigger = cat
                break

        return self.add(msg, trigger)

    def _init_db(self) -> None:
        with sqlite3.connect(self._db, timeout=30.0) as c:
            c.execute("""CREATE TABLE IF NOT EXISTS instructions(
                id TEXT PRIMARY KEY, text TEXT, trigger TEXT,
                active INTEGER, created_at REAL, use_count INTEGER)""")
