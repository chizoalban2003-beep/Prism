from prism_push import PrismPush


def test_not_configured_when_no_topic():
    assert not PrismPush().configured

def test_configured_when_topic_set():
    assert PrismPush(topic="test").configured

def test_send_returns_false_unconfigured():
    assert not PrismPush().send("t", "b")

def test_priority_map_has_urgent():
    assert PrismPush.PRIORITY_MAP["urgent"] == 5

def test_status_summary():
    s = PrismPush(topic="test-topic").status_summary()
    assert "topic" in s and "server" in s
