from __future__ import annotations

import pytest

from prism_service_discovery import DiscoveredService, PrismServiceDiscovery


@pytest.fixture
def disc(tmp_path):
    return PrismServiceDiscovery(
        collaborator=None,
        tool_registry=None,
        db_path=str(tmp_path / "services.db"),
    )


def test_not_known_initially(disc):
    assert disc.is_known("NewApp") is False


def test_discover_returns_tuple(disc):
    result = disc.discover("Telegram", "send reminders")
    assert isinstance(result, tuple)
    assert len(result) == 2
    service, questions = result
    assert isinstance(service, DiscoveredService)
    assert isinstance(questions, list)


def test_discover_stores(disc):
    disc.discover("Notion", "manage notes")
    assert disc.is_known("Notion") is True


def test_list_all_is_list(disc):
    disc.discover("Mastodon", "post updates")
    result = disc.list_all()
    assert isinstance(result, list)
    assert len(result) >= 1


def test_api_preferred(disc):
    # Simulate a profile that has an API — no_browser should not block API selection
    profile = {"name": "TestApp", "has_api": True, "has_webhook": False,
               "has_cli_app": False, "api_url": "", "needs_auth": True,
               "auth_type": "api_key"}
    method = disc._choose_method(profile, {})
    assert method == "official_api"

    # no_browser constraint should not block API (APIs are browser-independent)
    method_no_browser = disc._choose_method(profile, {"no_browser": True})
    assert method_no_browser == "official_api"


def test_browser_fallback(disc):
    # No API, no webhook, no CLI → browser or manual_steps
    profile = {"name": "TestApp", "has_api": False, "has_webhook": False,
               "has_cli_app": False, "api_url": "", "needs_auth": False,
               "auth_type": "none"}
    method = disc._choose_method(profile, {})
    assert method in ("browser", "manual_steps")
