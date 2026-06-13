"""
PRISM Speculative Decoding Pipeline

Architecture-level speculative decoding: draft model generates, target model
verifies. Maps to token-level speculative decoding semantics:
  - draft model = small fast capability-1 LLM
  - target model = large precise capability-3 LLM
  - verification = semantic rejection sampling (vs token-level in classic specd)

When budget is tight (capability_ceil ≤ 1): bypass verification entirely.
Draft IS the response. Target never invoked.

When CRYSTAL budget: always verify. Target may correct or accept draft.
Cost model: verification only when marginal accuracy gain > latency cost.
The threshold is the speculative window gamma (default 5 conceptual "steps").

Reference: Leviathan et al. (2022) "Fast Inference from Transformers via
Speculative Decoding". Token-level algo mapped to message-level here.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from prism_silicon_policy import ExecutionBudget

_log = logging.getLogger(__name__)


@dataclass
class SpeculativeResult:
    response: str
    verified: bool  # was draft reviewed by target model?
    draft_model: str
    verify_model: str  # empty string if verification was bypassed
    latency_ms: float
    corrected: bool  # did target model modify the draft?


class SpeculativeDecodingPipeline:
    """
    Coordinates draft + optional verify stages using the LLMRouter.

    Bypass conditions (draft IS final response, no verification):
    - budget.capability_ceil <= 1
    - budget.speculative is True AND budget.capability_ceil <= 2
    - draft response meets length threshold AND budget is VISCOUS/LIQUID

    Verification prompt uses a minimal system instruction so the target
    model acts purely as a quality gate, not a rewriter.
    """

    _VERIFY_SYSTEM = (
        "You are verifying a draft response for factual accuracy and completeness. "
        "If the draft is correct and complete, reproduce it EXACTLY. "
        "If there are errors or critical omissions, return a corrected version of "
        "similar length. Do not add unsolicited information."
    )
    _MIN_VERIFY_TOKENS = 30  # drafts under this word count skipped for verification

    def __init__(self, router: Any) -> None:
        self._router = router
        self._stats: dict[str, int] = {
            "drafts": 0,
            "verifications": 0,
            "corrections": 0,
            "bypasses": 0,
        }

    def call(
        self,
        prompt: str,
        budget: ExecutionBudget,
        system: str = "",
        conversation_history: list[dict] | None = None,
    ) -> SpeculativeResult:
        """
        Execute speculative pipeline for the given prompt.
        Always generates a draft. Verifies only when budget allows.
        """
        t0 = time.monotonic()
        history = conversation_history or []

        # Stage 1: Draft — always fastest available
        draft_resp, draft_model = self._router.call(
            prompt,
            min_capability=1,
            max_tokens=budget.max_tokens,
            system=system,
            conversation_history=history,
            phase_hint="fast",
        )
        self._stats["drafts"] += 1

        # Bypass decision
        if self._should_bypass(draft_resp, budget):
            self._stats["bypasses"] += 1
            return SpeculativeResult(
                response=draft_resp,
                verified=False,
                draft_model=draft_model,
                verify_model="",
                latency_ms=(time.monotonic() - t0) * 1000,
                corrected=False,
            )

        # Stage 2: Target verification
        verify_prompt = (
            f"Original question: {prompt}\n\n"
            f"Draft response to verify:\n---\n{draft_resp}\n---"
        )
        verify_resp, verify_model = self._router.call(
            verify_prompt,
            min_capability=min(budget.capability_ceil, 3),
            max_tokens=budget.max_tokens,
            system=self._VERIFY_SYSTEM,
            conversation_history=[],  # verification is stateless
        )
        self._stats["verifications"] += 1

        corrected = bool(verify_resp) and verify_resp.strip() != draft_resp.strip()
        if corrected:
            self._stats["corrections"] += 1

        return SpeculativeResult(
            response=verify_resp if corrected else draft_resp,
            verified=True,
            draft_model=draft_model,
            verify_model=verify_model,
            latency_ms=(time.monotonic() - t0) * 1000,
            corrected=corrected,
        )

    def _should_bypass(self, draft_resp: str, budget: ExecutionBudget) -> bool:
        """True when verification should be skipped."""
        # Hard bypass: budget ceiling too low
        if budget.capability_ceil <= 1:
            return True
        # Bypass when speculative flag set and budget is tight
        if budget.speculative and budget.capability_ceil <= 2:
            return True
        # Bypass very short drafts (not worth a second call)
        if len(draft_resp.split()) < self._MIN_VERIFY_TOKENS:
            return True
        return False

    @property
    def stats(self) -> dict[str, int]:
        return dict(self._stats)

    @property
    def correction_rate(self) -> float:
        """Fraction of verified responses that were corrected by target model."""
        v = self._stats["verifications"]
        return self._stats["corrections"] / v if v > 0 else 0.0

    @property
    def bypass_rate(self) -> float:
        """Fraction of calls where verification was bypassed."""
        d = self._stats["drafts"]
        return self._stats["bypasses"] / d if d > 0 else 0.0


# Module-level singleton factory (router injected at first use)
_pipeline: SpeculativeDecodingPipeline | None = None


def get_pipeline(router: Any | None = None) -> SpeculativeDecodingPipeline | None:
    global _pipeline
    if _pipeline is None and router is not None:
        _pipeline = SpeculativeDecodingPipeline(router)
    return _pipeline
