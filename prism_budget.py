"""
prism_budget.py
===============
CEO-style budget governance for PRISM.

The user is the CEO: they set a daily USD ceiling for LLM spending and a
soft warning band. PRISM (the manager) checks itself before every LLM call
and surfaces a budget card when ceilings are approached or breached.

This is intentionally a thin layer over the existing prism_llm_ledger —
the ledger is the source of truth for what was spent. This module is
only the *policy*: how much is allowed, who's warned, when to block.

Configuration (prism_config.toml):

    [budget]
    daily_usd        = 5.00     # hard ceiling per UTC day
    warn_at_fraction = 0.8      # warn the user at 80% of daily
    block_at_ceiling = true     # if false, only warn, never block
    monthly_usd      = 50.00    # optional monthly hard ceiling
    free_provider_bypass = true # local Ollama / stdlib never count
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

from prism_llm_ledger import get_ledger

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BudgetDecision:
    allowed:      bool
    reason:       str
    spent_today:  float
    daily_limit:  float
    warn:         bool
    fraction:     float


class BudgetPolicy:
    """
    CEO-level budget gate. Reads spend from llm_ledger, checks against the
    config-defined ceiling, returns a BudgetDecision the caller honors.
    """

    def __init__(
        self,
        daily_usd:            float = 5.00,
        warn_at_fraction:     float = 0.8,
        block_at_ceiling:     bool  = True,
        monthly_usd:          Optional[float] = None,
        free_provider_bypass: bool  = True,
    ) -> None:
        self.daily_usd            = max(0.0, float(daily_usd))
        self.warn_at_fraction     = max(0.0, min(1.0, float(warn_at_fraction)))
        self.block_at_ceiling     = bool(block_at_ceiling)
        self.monthly_usd          = float(monthly_usd) if monthly_usd else None
        self.free_provider_bypass = bool(free_provider_bypass)

    # ------------------------------------------------------------------
    # Spend snapshots — backed by prism_llm_ledger
    # ------------------------------------------------------------------

    def _day_window_start(self) -> float:
        # UTC midnight today
        t = time.gmtime()
        return time.mktime(time.struct_time((
            t.tm_year, t.tm_mon, t.tm_mday, 0, 0, 0, 0, 0, 0,
        ))) - time.timezone

    def spent_today_usd(self) -> float:
        try:
            return float(get_ledger().summary(since_ts=self._day_window_start()).get("total_cost_usd", 0.0))
        except Exception as exc:
            logger.debug("[budget] ledger read failed: %s", exc)
            return 0.0

    def spent_this_month_usd(self) -> float:
        if self.monthly_usd is None:
            return 0.0
        try:
            t = time.gmtime()
            month_start = time.mktime(time.struct_time((
                t.tm_year, t.tm_mon, 1, 0, 0, 0, 0, 0, 0,
            ))) - time.timezone
            return float(get_ledger().summary(since_ts=month_start).get("total_cost_usd", 0.0))
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # Decision
    # ------------------------------------------------------------------

    def check(self, provider: str = "") -> BudgetDecision:
        """
        Return a BudgetDecision for a *pending* LLM call. Caller honors:
          - decision.allowed=False → refuse the call and surface decision.reason
          - decision.warn=True     → surface a banner but proceed
        """
        # Local providers don't burn the CEO's budget.
        if self.free_provider_bypass and provider.lower() in ("ollama", "stdlib", "local"):
            return BudgetDecision(
                allowed=True, reason="local provider — free",
                spent_today=self.spent_today_usd(),
                daily_limit=self.daily_usd, warn=False, fraction=0.0,
            )

        spent = self.spent_today_usd()
        fraction = (spent / self.daily_usd) if self.daily_usd > 0 else 0.0

        # Monthly hard stop
        if self.monthly_usd is not None and self.spent_this_month_usd() >= self.monthly_usd:
            return BudgetDecision(
                allowed=not self.block_at_ceiling,
                reason=f"monthly budget ${self.monthly_usd:.2f} reached",
                spent_today=spent, daily_limit=self.daily_usd,
                warn=True, fraction=fraction,
            )

        # Daily hard stop
        if spent >= self.daily_usd:
            return BudgetDecision(
                allowed=not self.block_at_ceiling,
                reason=f"daily budget ${self.daily_usd:.2f} reached "
                       f"(spent ${spent:.4f})",
                spent_today=spent, daily_limit=self.daily_usd,
                warn=True, fraction=fraction,
            )

        # Warning band
        warn = fraction >= self.warn_at_fraction
        reason = "ok" if not warn else (
            f"approaching daily budget "
            f"({int(fraction*100)}% of ${self.daily_usd:.2f})"
        )
        return BudgetDecision(
            allowed=True, reason=reason,
            spent_today=spent, daily_limit=self.daily_usd,
            warn=warn, fraction=fraction,
        )

    def snapshot(self) -> dict:
        """Dashboard view — what the user (CEO) sees in a chat card."""
        spent_d = self.spent_today_usd()
        spent_m = self.spent_this_month_usd() if self.monthly_usd else None
        return {
            "daily_limit_usd":  round(self.daily_usd, 4),
            "spent_today_usd":  round(spent_d, 6),
            "remaining_usd":    round(max(0.0, self.daily_usd - spent_d), 6),
            "fraction_used":    round((spent_d / self.daily_usd) if self.daily_usd > 0 else 0.0, 3),
            "warn_at_fraction": self.warn_at_fraction,
            "block_at_ceiling": self.block_at_ceiling,
            "monthly_limit_usd": self.monthly_usd,
            "spent_this_month_usd": (round(spent_m, 6) if spent_m is not None else None),
            "free_provider_bypass": self.free_provider_bypass,
        }


# ---------------------------------------------------------------------------
# Loader from config
# ---------------------------------------------------------------------------

def from_config(cfg: Optional[dict]) -> BudgetPolicy:
    section = (cfg or {}).get("budget", {}) if isinstance(cfg, dict) else {}
    return BudgetPolicy(
        daily_usd            = section.get("daily_usd", 5.00),
        warn_at_fraction     = section.get("warn_at_fraction", 0.8),
        block_at_ceiling     = section.get("block_at_ceiling", True),
        monthly_usd          = section.get("monthly_usd"),
        free_provider_bypass = section.get("free_provider_bypass", True),
    )
