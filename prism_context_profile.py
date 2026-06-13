"""
prism_context_profile.py
========================
Work / personal / focus context switching for PRISM.

Each ContextProfile bundles:
  - soul_lens_ids       which PrismSoul lenses to activate
  - policy_overrides    dict of action → bool (True=allowed, False=denied)
  - organ_priorities    dict of intent → priority int (higher = preferred)
  - description         human-readable label

Usage
-----
    from prism_context_profile import ContextManager

    manager = ContextManager()
    manager.create("work",     description="Work mode — full access")
    manager.create("personal", description="Personal mode — no email_send")
    manager.switch("work")

    profile = manager.active()
    profile.policy_overrides   # {"email_send": True, ...}
    profile.soul_lens_ids      # ["lens_work_focus", ...]

The active context_id flows through:
  - PrismChain._context_id        injected at chain run-time
  - OutcomeTracker records        for per-context learning stats
  - PrismSoul lens filtering      via manager.apply_to_soul(soul)
  - PolicyEngine overrides        via manager.apply_to_policy(policy)
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_DB_PATH = "~/.prism/contexts.json"

BUILTIN_CONTEXTS = {
    "default": {
        "description": "Default context — all capabilities active",
        "soul_lens_ids": [],
        "policy_overrides": {},
        "organ_priorities": {},
    },
    "work": {
        "description": "Work mode — professional focus, email and calendar active",
        "soul_lens_ids": [],
        "policy_overrides": {"email_send": True, "calendar_write": True, "browser_task": True},
        "organ_priorities": {"meeting_brief": 10, "document_read": 8, "task_reminder": 9},
    },
    "personal": {
        "description": "Personal mode — health and finance tracking, email off by default",
        "soul_lens_ids": [],
        "policy_overrides": {"email_send": False, "calendar_write": True},
        "organ_priorities": {"health_summary": 10, "finance_summary": 8, "task_reminder": 9},
    },
    "focus": {
        "description": "Focus mode — minimal interruptions, no proactive triggers",
        "soul_lens_ids": [],
        "policy_overrides": {"email_send": False, "send_push": False, "autonomous": False},
        "organ_priorities": {},
    },
}


@dataclass
class ContextProfile:
    context_id:       str
    description:      str = ""
    soul_lens_ids:    list[str] = field(default_factory=list)
    policy_overrides: dict[str, bool] = field(default_factory=dict)
    organ_priorities: dict[str, int] = field(default_factory=dict)
    created_at:       float = field(default_factory=time.time)
    updated_at:       float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "context_id":       self.context_id,
            "description":      self.description,
            "soul_lens_ids":    self.soul_lens_ids,
            "policy_overrides": self.policy_overrides,
            "organ_priorities": self.organ_priorities,
            "created_at":       self.created_at,
            "updated_at":       self.updated_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ContextProfile:
        return cls(
            context_id       = d["context_id"],
            description      = d.get("description", ""),
            soul_lens_ids    = d.get("soul_lens_ids", []),
            policy_overrides = d.get("policy_overrides", {}),
            organ_priorities = d.get("organ_priorities", {}),
            created_at       = d.get("created_at", time.time()),
            updated_at       = d.get("updated_at", time.time()),
        )


class ContextManager:
    """
    Manages named ContextProfiles and tracks the active one.

    Persisted as JSON at ~/.prism/contexts.json.
    """

    def __init__(self, db_path: str = _DB_PATH):
        self._path   = Path(db_path).expanduser()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._profiles: dict[str, ContextProfile] = {}
        self._active_id: str = "default"
        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create(
        self,
        context_id: str,
        description: str = "",
        soul_lens_ids: Optional[list[str]] = None,
        policy_overrides: Optional[dict[str, bool]] = None,
        organ_priorities: Optional[dict[str, int]] = None,
    ) -> ContextProfile:
        """Create or overwrite a named context profile."""
        profile = ContextProfile(
            context_id       = context_id,
            description      = description,
            soul_lens_ids    = soul_lens_ids or [],
            policy_overrides = policy_overrides or {},
            organ_priorities = organ_priorities or {},
        )
        self._profiles[context_id] = profile
        self._save()
        logger.info("[context] created profile '%s'", context_id)
        return profile

    def switch(self, context_id: str) -> ContextProfile:
        """Switch the active context. Creates from builtin defaults if unknown."""
        if context_id not in self._profiles:
            if context_id in BUILTIN_CONTEXTS:
                d = BUILTIN_CONTEXTS[context_id]
                self.create(
                    context_id,
                    description      = d["description"],
                    soul_lens_ids    = d["soul_lens_ids"],
                    policy_overrides = d["policy_overrides"],
                    organ_priorities = d["organ_priorities"],
                )
            else:
                raise ValueError(f"Unknown context '{context_id}'. Create it first.")
        self._active_id = context_id
        self._save()
        logger.info("[context] switched to '%s'", context_id)
        return self._profiles[context_id]

    def active(self) -> ContextProfile:
        """Return the currently active ContextProfile."""
        if self._active_id not in self._profiles:
            self._ensure_default()
        return self._profiles[self._active_id]

    @property
    def active_id(self) -> str:
        return self._active_id

    def get(self, context_id: str) -> Optional[ContextProfile]:
        return self._profiles.get(context_id)

    def list_profiles(self) -> list[ContextProfile]:
        return list(self._profiles.values())

    def delete(self, context_id: str) -> bool:
        if context_id == "default":
            logger.warning("[context] cannot delete 'default' context")
            return False
        removed = self._profiles.pop(context_id, None)
        if removed and self._active_id == context_id:
            self._active_id = "default"
        self._save()
        return removed is not None

    # ------------------------------------------------------------------
    # Integration helpers
    # ------------------------------------------------------------------

    def apply_to_soul(self, soul) -> None:
        """
        Filter soul lenses to only those in the active context's soul_lens_ids.
        If soul_lens_ids is empty, all lenses are active (no filtering).
        """
        profile = self.active()
        if not profile.soul_lens_ids:
            return
        try:
            all_lenses = soul.list_lenses()
            for ln in all_lenses:
                # PrismSoul doesn't have a disable_lens yet — we add a note
                # to each inactive lens so compress_for_llm can skip it
                active = ln.lens_id in profile.soul_lens_ids
                if hasattr(soul, "set_lens_active"):
                    soul.set_lens_active(ln.lens_id, active)
        except Exception as exc:
            logger.debug("[context] apply_to_soul failed: %s", exc)

    def apply_to_policy(self, policy) -> None:
        """
        Push policy_overrides from the active context into the PolicyEngine.
        Uses set_allowance() if available, falls back gracefully.
        """
        profile = self.active()
        if not profile.policy_overrides:
            return
        for action, allowed in profile.policy_overrides.items():
            try:
                if hasattr(policy, "set_allowance"):
                    policy.set_allowance(action, allowed)
                elif hasattr(policy, "allow") and allowed:
                    policy.allow(action)
                elif hasattr(policy, "deny") and not allowed:
                    policy.deny(action)
            except Exception as exc:
                logger.debug("[context] apply_to_policy failed for %s: %s", action, exc)

    def inject_into_chain(self, chain) -> None:
        """Set chain._context_id to the active context."""
        chain._context_id = self._active_id

    def inject_into_chain_ctx(self, ctx: dict) -> dict:
        """Add context_id and organ_priorities to a chain base_ctx dict."""
        profile = self.active()
        ctx["context_id"]       = self._active_id
        ctx["organ_priorities"] = profile.organ_priorities
        return ctx

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        self._ensure_default()
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text())
            for d in data.get("profiles", []):
                p = ContextProfile.from_dict(d)
                self._profiles[p.context_id] = p
            self._active_id = data.get("active_id", "default")
        except Exception as exc:
            logger.warning("[context] load failed: %s", exc)

    def _save(self) -> None:
        try:
            data = {
                "active_id": self._active_id,
                "profiles":  [p.to_dict() for p in self._profiles.values()],
            }
            self._path.write_text(json.dumps(data, indent=2))
        except Exception as exc:
            logger.warning("[context] save failed: %s", exc)

    def _ensure_default(self) -> None:
        if "default" not in self._profiles:
            d = BUILTIN_CONTEXTS["default"]
            self._profiles["default"] = ContextProfile(
                context_id       = "default",
                description      = d["description"],
                soul_lens_ids    = d["soul_lens_ids"],
                policy_overrides = d["policy_overrides"],
                organ_priorities = d["organ_priorities"],
            )
