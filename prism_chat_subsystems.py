"""
prism_chat_subsystems.py
========================
Factory for the must-succeed chat-path components.

PrismAgent.__init__ needs five tightly-coupled subsystems built in a
specific order:

* PrismAutonomous   — execution gateway
* PrismComposer     — multi-step composition
* PrismChain        — primary adaptive chain
* OrganLoader       — synthesised-organ registry, back-patched onto chain
* PrismChainExpert  — research-heavy chain (shares autonomous + memory)

All five take overlapping construction args (router/policy/push/queue)
and they participate together in every chat turn. Building them here
keeps __init__ short and groups the cluster as one named concept.

These are not fail-soft: any constructor exception propagates so the
agent fails loudly at startup rather than producing a half-built chat
path.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from prism_autonomous import PrismAutonomous
from prism_chain import PrismChain
from prism_chain_expert import PrismChainExpert
from prism_chain_theory import InterceptorPolicy
from prism_composer import PrismComposer
from prism_organ_loader import OrganLoader


@dataclass
class ChatSubsystems:
    autonomous:   PrismAutonomous
    composer:     PrismComposer
    chain:        PrismChain
    organ_loader: OrganLoader
    chain_expert: PrismChainExpert


def build_chat_subsystems(
    *,
    router:       Any,
    policy:       Any,
    push:         Any,
    task_queue:   Any,
    device_agent: Any,
    memory:       Any,
) -> ChatSubsystems:
    """Build the chat-path subsystem cluster. Raises on any failure —
    these components are core to the agent and not safe to skip."""
    autonomous = PrismAutonomous(
        llm_router    = router,
        device_agent  = device_agent,
        policy_engine = policy,
        push          = push,
        task_queue    = task_queue,
    )
    composer = PrismComposer(
        llm_router    = router,
        policy_engine = policy,
        push          = push,
        task_queue    = task_queue,
    )
    chain = PrismChain(
        llm_router         = router,
        policy_engine      = policy,
        push               = push,
        autonomous         = autonomous,
        memory             = memory,
        interceptor_policy = InterceptorPolicy(),
        # soul omitted — PrismSoul is built later and back-patched via
        # PrismAgent._wire_backpatches(); passing it here is a no-op.
    )
    organ_loader = OrganLoader(llm_router=router)
    chain._organ_loader = organ_loader
    chain_expert = PrismChainExpert(
        llm_router    = router,
        policy_engine = policy,
        push          = push,
        autonomous    = autonomous,
        memory        = memory,
    )
    return ChatSubsystems(
        autonomous   = autonomous,
        composer     = composer,
        chain        = chain,
        organ_loader = organ_loader,
        chain_expert = chain_expert,
    )
