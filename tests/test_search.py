from prism_search import PrismSearch

def test_always_configured():
    assert PrismSearch().configured

def test_ddg_provider_when_no_keys():
    assert PrismSearch()._resolve_provider() == "ddg"

def test_brave_provider_when_key_set():
    assert PrismSearch(brave_key="x")._resolve_provider() == "brave"

def test_quick_answer_returns_str():
    result = PrismSearch().quick_answer("capital of France")
    assert isinstance(result, str)

def test_status_summary_has_provider():
    s = PrismSearch().status_summary()
    assert "provider" in s
