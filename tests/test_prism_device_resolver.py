"""Tests for prism_device_resolver.py"""
from prism_device_resolver import ToolResolver


def test_import():
    """Module imports without error."""
    pass  # import above is the test


def test_instantiation():
    """ToolResolver instantiates without error."""
    obj = ToolResolver()
    assert obj is not None


def test_resolve_returns_result():
    """resolve() returns a ToolResolution for a known task type."""
    from prism_device_resolver import CapabilityMap
    import sys
    resolver = ToolResolver()
    caps = CapabilityMap(
        cli_tools=set(),
        py_packages=set(),
        platform=sys.platform,
        has_browser=False,
    )
    result = resolver.resolve("open_file", "open a text file", caps)
    # Should return a ToolResolution regardless of whether a tool was found
    assert result is not None
    assert hasattr(result, "resolved")
