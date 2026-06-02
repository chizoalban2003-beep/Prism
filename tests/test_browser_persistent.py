"""Tests for persistent browser sessions — Gap Prompt 13c."""
from unittest.mock import MagicMock, patch

from prism_browser_agent import PersistentBrowserContext, PrismBrowserAgent


def test_session_path_returns_string():
    """session_path("google.com") returns a string."""
    ctx = PersistentBrowserContext()
    result = ctx.session_path("google.com")
    assert isinstance(result, str)
    assert "google_com" in result


def test_has_session_false_initially():
    """has_session("new-site.com") returns False for a domain with no saved session."""
    ctx = PersistentBrowserContext()
    assert ctx.has_session("new-site-xyzzy-notreal.com") is False


def test_available_property_still_works():
    """PrismBrowserAgent().available returns a bool (unchanged by new additions)."""
    agent = PrismBrowserAgent()
    assert isinstance(agent.available, bool)


def test_execute_with_session_unavailable():
    """When playwright is missing, execute_with_session() returns success=False."""
    agent = PrismBrowserAgent()
    with patch.object(type(agent), "available",
                      new_callable=lambda: property(lambda self: False)):
        result = agent.execute_with_session("find something")
    assert result.success is False
    assert result.error != ""


def test_login_indicators_detection():
    """_execute_action returns login_required when page contains login wall text."""
    agent = PrismBrowserAgent()

    # Build a mock page whose text contains enough login indicators
    mock_page = MagicMock()
    mock_page.url = "https://example.com/login"
    mock_page.evaluate.return_value = (
        "sign in to your account log in create account "
        "password email address username"
    )

    action_json = {
        "action": "navigate",
        "target": "https://example.com/login",
        "description": "Go to login page",
    }

    # Patch goto so it doesn't actually navigate
    mock_page.goto = MagicMock()

    step = agent._execute_action(mock_page, action_json)
    assert step.action == "login_required"
    assert step.success is False
    assert "example.com" in step.target
