"""Tests for prism_search.py — Gap Prompt 14a."""
from unittest.mock import patch

from prism_search import PrismSearch, SearchResult


def test_always_configured():
    """PrismSearch().configured is always True (DuckDuckGo requires no key)."""
    assert PrismSearch().configured is True


def test_ddg_provider_when_no_keys():
    """With no API keys, provider resolves to 'ddg'."""
    assert PrismSearch()._resolve_provider() == "ddg"


def test_brave_provider_when_key_set():
    """With a Brave key, provider resolves to 'brave'."""
    assert PrismSearch(brave_key="x")._resolve_provider() == "brave"


def test_serp_provider_when_key_set():
    """With a SerpAPI key and no Brave key, provider resolves to 'serp'."""
    assert PrismSearch(serp_key="x")._resolve_provider() == "serp"


def test_auto_prefers_brave_over_serp():
    """Auto mode picks brave first when both keys are provided."""
    s = PrismSearch(provider="auto", brave_key="b", serp_key="s")
    assert s._resolve_provider() == "brave"


def test_quick_answer_returns_str():
    """quick_answer always returns a str (may be empty on network failure)."""
    with patch("urllib.request.urlopen") as mock_open:
        mock_open.side_effect = OSError("no network")
        result = PrismSearch().quick_answer("capital of France")
    assert isinstance(result, str)


def test_status_summary_has_provider():
    """status_summary() contains a 'provider' key."""
    s = PrismSearch().status_summary()
    assert "provider" in s
    assert "configured" in s
    assert "free_tier" in s


def test_from_config_uses_search_section():
    """from_config reads the [search] section."""
    cfg = {"search": {"provider": "serp", "serp_api_key": "abc"}}
    s = PrismSearch.from_config(cfg)
    assert s._resolve_provider() == "serp"


def test_search_returns_list_on_network_failure():
    """search() returns a list (possibly empty) when network is unavailable."""
    with patch("urllib.request.urlopen") as mock_open:
        mock_open.side_effect = OSError("no network")
        results = PrismSearch().search("test query")
    assert isinstance(results, list)
