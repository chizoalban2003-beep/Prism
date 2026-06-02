"""Tests for prism_calls.py — Gap Prompt 13b."""
from prism_calls import PrismCalls, CallResult


def test_not_configured_when_empty():
    """PrismCalls() with no args should not be configured on non-macOS."""
    import sys
    calls = PrismCalls()
    if sys.platform != "darwin":
        assert calls.configured is False


def test_twilio_configured_when_set():
    """PrismCalls with account_sid and auth_token should report configured."""
    assert PrismCalls(account_sid="x", auth_token="y").configured is True


def test_macos_detection():
    """_is_macos() should return a bool."""
    result = PrismCalls._is_macos()
    assert isinstance(result, bool)


def test_resolve_provider_auto_with_twilio_creds():
    """auto provider with Twilio creds should resolve to 'twilio'."""
    calls = PrismCalls(provider="auto", account_sid="ACxxx", auth_token="tok")
    assert calls._resolve_provider() == "twilio"


def test_resolve_provider_none():
    """No creds and not macOS should resolve to 'none'."""
    import sys
    calls = PrismCalls(provider="auto")
    if sys.platform != "darwin":
        assert calls._resolve_provider() == "none"


def test_call_returns_call_result_when_not_configured():
    """call() returns CallResult(success=False) when no provider is configured."""
    import sys
    if sys.platform == "darwin":
        return  # macOS Continuity would be available — skip
    result = PrismCalls().call("+15005550006")
    assert isinstance(result, CallResult)
    assert result.success is False


def test_from_config_empty():
    """from_config({}) should produce a PrismCalls with default values."""
    calls = PrismCalls.from_config({})
    assert calls._provider == "auto"
    assert calls._sid == ""


def test_status_summary_keys():
    """status_summary() should contain 'configured', 'provider', and 'from_number'."""
    summary = PrismCalls().status_summary()
    assert "configured" in summary
    assert "provider" in summary
    assert "from_number" in summary
