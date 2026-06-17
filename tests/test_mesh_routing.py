"""M12a — capability-aware mesh auto-routing + hop-limit enforcement.

Covers:
  * score_peer_for_task (pure scorer)
  * PrismMesh.find_capable_peer (unique, ambiguous, no-signal cases)
  * forward_task / forward_chat hop-limit refusal at MAX_HOPS
  * mesh_orchestrate organ auto-routes when peer omitted and one peer wins
  * mesh_orchestrate organ asks for disambiguation when tied
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from prism_mesh import (
    MAX_HOPS,
    PrismMesh,
    score_peer_for_task,
)

# ---------------------------------------------------------------------------
# score_peer_for_task — pure scorer
# ---------------------------------------------------------------------------

def test_score_browser_task_matches_browser_peer():
    caps = {"has_browser": True, "categories": {}}
    assert score_peer_for_task(caps, "screenshot_url", {"url": "x"}) >= 1


def test_score_browser_task_misses_headless_peer():
    caps = {"has_browser": False, "categories": {}}
    assert score_peer_for_task(caps, "screenshot_url", {"url": "x"}) == 0


def test_score_ffmpeg_task_matches_video_peer():
    caps = {"has_browser": False, "categories": {"video": ["ffmpeg"]}}
    assert score_peer_for_task(caps, "ffmpeg_compress", {}) >= 1


def test_score_unrelated_task_returns_zero():
    caps = {"has_browser": True, "categories": {"git": ["git"]}}
    assert score_peer_for_task(caps, "fetch_random_thing", {}) == 0


def test_score_empty_task_returns_zero():
    assert score_peer_for_task({"has_browser": True}, "", None) == 0


# ---------------------------------------------------------------------------
# find_capable_peer — registry-level routing
# ---------------------------------------------------------------------------

@pytest.fixture
def mesh_with_peers(tmp_path):
    m = PrismMesh(store_path=str(tmp_path / "mesh.json"))
    # Register peers without triggering network capability refresh.
    with patch.object(PrismMesh, "refresh_capabilities", lambda self, pid: {}):
        p1 = m.register_peer("desk",   "10.0.0.1", 8742, "tok1")
        p2 = m.register_peer("laptop", "10.0.0.2", 8742, "tok2")
    # Hand-set capabilities the way /device/capabilities would shape them.
    p1.capabilities = {"has_browser": True,  "categories": {"git": ["git"]}}
    p2.capabilities = {"has_browser": False, "categories": {"video": ["ffmpeg"]}}
    return m


def test_find_capable_picks_unique_browser_peer(mesh_with_peers):
    best, candidates = mesh_with_peers.find_capable_peer("screenshot_url", {"url": "x"})
    assert best is not None
    assert best.name == "desk"
    assert len(candidates) == 1


def test_find_capable_picks_unique_ffmpeg_peer(mesh_with_peers):
    best, candidates = mesh_with_peers.find_capable_peer("ffmpeg_resize", {})
    assert best is not None
    assert best.name == "laptop"


def test_find_capable_ambiguous_returns_none_and_tied_candidates(mesh_with_peers):
    # Both peers have ZERO score for an unrelated task → ambiguity (all peers).
    best, candidates = mesh_with_peers.find_capable_peer("unknown_thing", {})
    assert best is None
    assert {p.name for p in candidates} == {"desk", "laptop"}


def test_find_capable_empty_mesh(tmp_path):
    m = PrismMesh(store_path=str(tmp_path / "mesh.json"))
    best, candidates = m.find_capable_peer("anything", {})
    assert best is None
    assert candidates == []


# ---------------------------------------------------------------------------
# Hop-limit enforcement
# ---------------------------------------------------------------------------

def test_forward_task_refuses_at_max_hops(mesh_with_peers):
    desk = mesh_with_peers.find_peer_by_name("desk")
    result = mesh_with_peers.forward_task(
        desk.peer_id, "anything", params={"_hop": MAX_HOPS}
    )
    assert result["success"] is False
    assert "hop limit" in result["error"].lower()


def test_forward_task_increments_hop(mesh_with_peers):
    desk = mesh_with_peers.find_peer_by_name("desk")
    captured = {}

    def fake_http(self, method, peer, path, body=None):
        captured["body"] = body
        return {"success": True, "output": "ok"}

    with patch.object(PrismMesh, "_http_json", fake_http):
        mesh_with_peers.forward_task(desk.peer_id, "noop", params={"_hop": 1})
    assert captured["body"]["params"]["_hop"] == 2


def test_forward_chat_refuses_at_max_hops(mesh_with_peers):
    desk = mesh_with_peers.find_peer_by_name("desk")
    result = mesh_with_peers.forward_chat(desk.peer_id, "hi", hop=MAX_HOPS)
    assert "error" in result
    assert "hop limit" in result["error"].lower()


# ---------------------------------------------------------------------------
# mesh_orchestrate organ — auto-routing UX
# ---------------------------------------------------------------------------

def test_mesh_organ_auto_routes_when_peer_omitted(mesh_with_peers):
    import organs.mesh_orchestrate as organ

    # Empty title on the forwarded reply so the organ's auto-routed
    # "From <peer> (auto-routed…)" title takes effect.
    with patch("prism_mesh.get_mesh", return_value=mesh_with_peers), \
         patch.object(PrismMesh, "forward_chat",
                      return_value={"body": "remote response"}):
        card = organ.execute(
            intent="mesh_orchestrate",
            message="screenshot github.com",
            ctx={"params": {}},
        )
    assert "remote response" in card.body
    assert "auto-routed" in card.title
    # Browser task → desk wins
    assert "desk" in card.title


def test_mesh_organ_asks_to_pick_when_ambiguous(mesh_with_peers):
    import organs.mesh_orchestrate as organ

    # Make both peers tie at score 0 — generic message picks no category.
    with patch("prism_mesh.get_mesh", return_value=mesh_with_peers):
        card = organ.execute(
            intent="mesh_orchestrate",
            message="say hello",
            ctx={"params": {}},
        )
    assert "Pick a peer" in card.title or "Which device" in card.body


def test_mesh_organ_no_peers_returns_no_peers_card(tmp_path):
    import organs.mesh_orchestrate as organ

    empty = PrismMesh(store_path=str(tmp_path / "mesh.json"))
    with patch("prism_mesh.get_mesh", return_value=empty):
        card = organ.execute(
            intent="mesh_orchestrate",
            message="run a thing",
            ctx={"params": {}},
        )
    assert "No peers" in card.title or "No mesh peers" in card.body
