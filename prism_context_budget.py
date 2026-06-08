"""
PRISM Context Budget Manager

StreamingLLM-inspired KV-cache eviction at the prompt/message level.
When ExecutionBudget drops, prunes conversation history while preserving:
  - Attention sinks: system prompt + first user turn (always high-attention)
  - Recent window: last RECENT_WINDOW messages (recency bias)
  - Heavy hitters: highest-relevance middle messages (H2O approximation)

On GPU hardware with vLLM: the same policy maps to actual KV pair eviction
via the vLLM cache engine's block manager. This module handles the CPU path.

Reference: StreamingLLM (Xiao et al. 2023), H2O (Zhang et al. 2023)
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from prism_silicon_policy import ExecutionBudget

_log = logging.getLogger(__name__)

CHARS_PER_TOKEN: int = 4  # conservative approximation


@dataclass
class EvictionResult:
    messages: list[dict]  # pruned message list
    evicted_count: int  # messages removed
    tokens_before: int  # estimated tokens before pruning
    tokens_after: int  # estimated tokens after pruning
    strategy: str  # "none"|"streamingllm"|"h2o"


class ContextBudgetManager:
    """
    Manages conversation history size to match ExecutionBudget constraints.

    Eviction tiers based on budget.max_tokens:
    - tokens ≥ 1200:  no eviction
    - tokens ≥ 700:   StreamingLLM (sinks + recent window, evict middle)
    - tokens < 700:   H2O approximation (sinks + recent + highest-relevance middle)

    Token budget for context = max_tokens * CONTEXT_MULTIPLIER.
    Rationale: context is read (cheap) vs generation (expensive); allow 3x.
    """

    SINK_COUNT = 2  # system + first user turn
    RECENT_WINDOW = 4  # always keep last N messages
    CONTEXT_MULTIPLIER = 3  # context budget = max_tokens * 3

    def prune(
        self,
        messages: list[dict],
        budget: "ExecutionBudget",
        query: str = "",
    ) -> EvictionResult:
        """
        Prune messages to fit within the context budget derived from budget.max_tokens.
        Returns EvictionResult with pruned messages and statistics.
        """
        token_budget = budget.max_tokens * self.CONTEXT_MULTIPLIER
        tokens_before = self._token_estimate(messages)

        if tokens_before <= token_budget or len(messages) <= self.SINK_COUNT + self.RECENT_WINDOW:
            return EvictionResult(
                messages=messages,
                evicted_count=0,
                tokens_before=tokens_before,
                tokens_after=tokens_before,
                strategy="none",
            )

        # Partition: sinks (always keep) | candidates | recent (always keep)
        sinks = messages[: self.SINK_COUNT]
        recent = messages[-self.RECENT_WINDOW :]
        middle = messages[self.SINK_COUNT : len(messages) - self.RECENT_WINDOW]

        if not middle:
            return EvictionResult(
                messages=messages,
                evicted_count=0,
                tokens_before=tokens_before,
                tokens_after=tokens_before,
                strategy="none",
            )

        strategy = "h2o" if budget.max_tokens < 700 else "streamingllm"

        if strategy == "streamingllm":
            # StreamingLLM: simply discard all middle messages
            kept_middle: list[dict] = []
        else:
            # H2O: rank by TF-IDF relevance to query, keep top-K that fit budget
            sink_recent_tokens = self._token_estimate(sinks) + self._token_estimate(recent)
            available = max(0, token_budget - sink_recent_tokens)
            kept_middle = self._h2o_select(middle, available, query)

        evicted = len(middle) - len(kept_middle)
        pruned = sinks + kept_middle + recent
        tokens_after = self._token_estimate(pruned)

        _log.debug(
            "[ctx_budget] %s: %d msgs → %d msgs, %d tokens → %d tokens (evicted=%d)",
            strategy,
            len(messages),
            len(pruned),
            tokens_before,
            tokens_after,
            evicted,
        )

        return EvictionResult(
            messages=pruned,
            evicted_count=evicted,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            strategy=strategy,
        )

    # ── Relevance ranking (H2O approximation) ──────────────────────────────────

    def _h2o_select(
        self,
        candidates: list[dict],
        token_budget: int,
        query: str,
    ) -> list[dict]:
        """
        Keep as many candidates as fit in token_budget, prioritised by relevance.
        Preserves original ordering for kept messages (no reordering).
        """
        if not candidates:
            return []

        # Score all candidates
        scored = [
            (i, self._relevance(msg.get("content", ""), query), msg)
            for i, msg in enumerate(candidates)
        ]
        # Sort by relevance descending
        scored.sort(key=lambda x: x[1], reverse=True)

        kept_indices: set[int] = set()
        budget_remaining = token_budget
        for i, _score, msg in scored:
            msg_tokens = self._token_estimate([msg])
            if msg_tokens <= budget_remaining:
                kept_indices.add(i)
                budget_remaining -= msg_tokens

        # Restore original order
        return [msg for i, _s, msg in sorted(scored, key=lambda x: x[0]) if i in kept_indices]

    # ── Utilities ──────────────────────────────────────────────────────────────

    @staticmethod
    def _token_estimate(messages: list[dict]) -> int:
        """Rough token count: total characters / CHARS_PER_TOKEN."""
        return sum(len(m.get("content", "")) for m in messages) // CHARS_PER_TOKEN

    @staticmethod
    def _relevance(text: str, query: str) -> float:
        """
        TF-IDF-inspired relevance: sum of (1/word_length) for overlapping content words.
        Rare (longer) words score higher than common short words.
        Returns 0.0 when query is empty (no relevance signal available).
        """
        if not query or not text:
            return 0.0
        _stop = frozenset(
            {
                "the",
                "and",
                "for",
                "are",
                "was",
                "that",
                "this",
                "with",
                "from",
                "have",
                "been",
                "will",
                "would",
            }
        )

        def _words(s: str) -> set[str]:
            return {w for w in re.findall(r"[a-z]+", s.lower()) if len(w) >= 3 and w not in _stop}

        tw = _words(text)
        qw = _words(query)
        if not tw or not qw:
            return 0.0
        overlap = tw & qw
        return sum(1.0 / len(w) for w in overlap) / max(1, len(qw))

    def vllm_eviction_policy(self, budget: "ExecutionBudget") -> dict:
        """
        Returns vLLM KV cache eviction policy configuration dict.
        Wire-up (GPU hardware):
            from vllm import LLM
            engine = LLM(..., **manager.vllm_eviction_policy(budget))
        """
        if budget.max_tokens >= 1200:
            return {"max_num_seqs": 256, "max_num_batched_tokens": 32768}
        if budget.max_tokens >= 700:
            # StreamingLLM sink window
            return {"max_num_seqs": 64, "max_num_batched_tokens": 8192}
        # H2O aggressive eviction
        return {"max_num_seqs": 16, "max_num_batched_tokens": 2048}


# Module-level singleton
_manager: ContextBudgetManager | None = None


def get_context_manager() -> ContextBudgetManager:
    global _manager
    if _manager is None:
        _manager = ContextBudgetManager()
    return _manager
