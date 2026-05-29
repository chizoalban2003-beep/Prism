from __future__ import annotations

from prism_collaborator import PrismCollaborator
from prism_tool_finder import ToolFinder


def test_always_has_manual():
    finder = ToolFinder()

    result = finder.find(task="order dinner", provider_name="Pizza Palace")

    assert any(option.execution_type == "manual" for option in result.options)


def test_always_has_app():
    finder = ToolFinder()

    result = finder.find(task="order dinner", provider_name="Pizza Palace")

    assert any(option.execution_type == "app_install" for option in result.options)


def test_urgent_prefers_fast():
    finder = ToolFinder(collaborator=PrismCollaborator())

    result = finder.find(
        task="order dinner",
        provider_name="Pizza Palace",
        urgency=0.9,
        cost_tolerance=0.6,
        prefers_auto=0.7,
        budget_left=1.0,
    )

    assert result.recommended.execution_type in {"aggregator", "phone"}


def test_no_budget_prefers_free():
    finder = ToolFinder(collaborator=PrismCollaborator())

    result = finder.find(
        task="order dinner",
        provider_name="Pizza Palace",
        budget_left=0.0,
        cost_tolerance=0.0,
        prefers_auto=0.2,
    )

    assert result.recommended.execution_type == "manual"


def test_discover_returns_list():
    finder = ToolFinder(collaborator=PrismCollaborator())

    options = finder._discover_options("order dinner", "Pizza Palace")

    assert options
