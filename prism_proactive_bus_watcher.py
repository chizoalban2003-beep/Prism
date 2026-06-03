"""
prism_proactive_bus_watcher.py
================================
Bridges OrganBus signals into PrismProactive triggers.

When a health, finance, calendar, or goal organ emits an anomaly signal,
this watcher translates it into a proactive notification — without waiting
for the user to ask.

Signal types watched
--------------------
  health_alert      — e.g. HRV drop, low step count
  finance_alert     — e.g. budget overrun, unusual transaction
  calendar_conflict — overlapping or missing prep time
  goal_triggered    — HorizonPlanner goal became active
  task_completed    — background task finished
  organ_error       — organ execution failed (warn the user)

Each signal is converted to a ProactiveEvent via proactive.schedule()
with a short delay (0–30s) so signals don't interrupt a live chain.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from prism_organ_bus import OrganBus
    from prism_proactive import PrismProactive

logger = logging.getLogger(__name__)

# Signal types this watcher subscribes to
WATCHED_SIGNALS = {
    "health_alert",
    "finance_alert",
    "calendar_conflict",
    "goal_triggered",
    "task_completed",
    "organ_error",
}

# Cooldown per signal type (seconds) — prevents spam
_COOLDOWNS: dict[str, float] = {}
_COOLDOWN_SECONDS = 300  # 5 min per signal type


class ProactiveBusWatcher:
    """
    Subscribes to OrganBus and fires proactive notifications via PrismProactive.

    Usage (inside PrismAgent.__init__):
        self._bus_watcher = ProactiveBusWatcher(
            proactive  = self._proactive,
            organ_bus  = self._organ_bus,
        )
        self._bus_watcher.register()
    """

    def __init__(self, proactive: "PrismProactive", organ_bus: "OrganBus"):
        self._proactive = proactive
        self._bus       = organ_bus

    def register(self) -> None:
        """Subscribe to all watched signal types on the OrganBus."""
        self._bus.register(
            organ_name   = "proactive_watcher",
            signal_types = list(WATCHED_SIGNALS),
            handler      = self._handle,
            vocabulary   = (
                "health_alert finance_alert calendar_conflict goal_triggered "
                "task_completed organ_error anomaly notification alert warning"
            ),
        )
        logger.info("[bus_watcher] subscribed to %d signal types", len(WATCHED_SIGNALS))

    def _handle(self, payload: dict) -> None:
        """Called by OrganBus for each matching signal."""
        import time
        signal_type = payload.get("signal_type", "unknown")
        source      = payload.get("source", "unknown")
        message     = payload.get("message") or payload.get("text") or str(payload)[:200]

        # Cooldown check
        last = _COOLDOWNS.get(signal_type, 0.0)
        now  = time.time()
        if now - last < _COOLDOWN_SECONDS:
            logger.debug("[bus_watcher] cooldown active for %s, skipping", signal_type)
            return
        _COOLDOWNS[signal_type] = now

        notification = self._format_notification(signal_type, source, message, payload)
        if notification:
            try:
                self._proactive.schedule_in(notification, seconds=5)
                logger.info("[bus_watcher] scheduled notification for %s from %s", signal_type, source)
            except Exception as exc:
                logger.debug("[bus_watcher] schedule_in failed: %s", exc)

    def _format_notification(
        self, signal_type: str, source: str, message: str, payload: dict
    ) -> str:
        """Build a human-readable notification string from a signal."""
        templates = {
            "health_alert": (
                "Health check: {message}"
            ),
            "finance_alert": (
                "Finance alert from {source}: {message}"
            ),
            "calendar_conflict": (
                "Calendar conflict detected: {message}"
            ),
            "goal_triggered": (
                "Horizon goal activated: {message}"
            ),
            "task_completed": (
                "Background task complete: {message}"
            ),
            "organ_error": (
                "Organ error in {source}: {message}"
            ),
        }
        template = templates.get(signal_type, "Alert from {source}: {message}")
        try:
            return template.format(source=source, message=message[:200])
        except Exception:
            return f"{signal_type} from {source}: {message[:200]}"
