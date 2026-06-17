"""
prism_mesh.py
=============
Device mesh for multi-device PRISM orchestration. Each device runs its own
daemon; the mesh registry tracks peers (host:port + bearer token + last-seen
capabilities) so tasks can be forwarded to whichever device best matches the
intent.

The registry is JSON-backed at ~/.prism/mesh.json. Discovery is manual: the
user registers a peer with `register_peer()` (typically from chat) and the
mesh refreshes capabilities by polling each peer's /device/capabilities on
demand. No automatic broadcast — keeps the surface small and local-first.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

_DEFAULT_TIMEOUT = 8.0

# Hop limit for chained forwards. Each forward bumps `_hop` in params/body;
# at MAX_HOPS the local mesh refuses to forward again. Prevents A→B→C→…
# loops from a misbehaving organ or a compromised peer. Two hops is enough
# for legitimate "ask laptop, laptop asks desk" patterns and stops longer
# chains cold.
MAX_HOPS = 2

# Capability hints: keyword → predicate(peer_capabilities_dict) -> bool.
# The router scores each peer by counting satisfied predicates whose
# keyword appears in the task description / params. Predicates are tiny
# and read against the dict returned by /device/capabilities, so the
# shape is exactly what `Peer.capabilities` already stores.
def _has_browser(caps: dict) -> bool:
    return bool(caps.get("has_browser"))

def _has_cat(category: str):
    def pred(caps: dict) -> bool:
        cats = caps.get("categories") or caps.get("cli_tools") or {}
        return bool(cats.get(category))
    return pred

def _has_pkg(pkg: str):
    def pred(caps: dict) -> bool:
        return pkg in (caps.get("py_packages") or [])
    return pred

CAPABILITY_HINTS: dict[str, list] = {
    "browser":      [_has_browser],
    "url":          [_has_browser],
    "screenshot":   [_has_browser],
    "scrape":       [_has_browser],
    "git":          [_has_cat("git")],
    "search":       [_has_cat("search")],
    "find":         [_has_cat("find_file")],
    "image":        [_has_cat("image_resize"), _has_cat("image_compress"), _has_pkg("PIL")],
    "video":        [_has_cat("video")],
    "ffmpeg":       [_has_cat("video")],
    "compress":     [_has_cat("compress_zip"), _has_cat("compress_tar")],
    "zip":          [_has_cat("compress_zip")],
    "tar":          [_has_cat("compress_tar")],
    "install":      [_has_cat("package_manager")],
    "package":      [_has_cat("package_manager")],
}


def _task_keywords(task: str, params: Optional[dict]) -> list[str]:
    """Lowercase keywords drawn from the task slug + str-valued params."""
    bag: list[str] = []
    if task:
        bag.append(task.lower())
    for v in (params or {}).values():
        if isinstance(v, str):
            bag.append(v.lower())
    return bag


def score_peer_for_task(peer_caps: dict, task: str, params: Optional[dict] = None) -> int:
    """How well does this peer match the task? Higher = better.

    Score = number of satisfied capability predicates whose keyword
    appears in the task or string-valued params. Returns 0 when nothing
    matches — the caller decides whether to fall back to "any peer" or
    refuse to auto-route.
    """
    bag = " ".join(_task_keywords(task, params))
    if not bag:
        return 0
    score = 0
    for keyword, preds in CAPABILITY_HINTS.items():
        if keyword not in bag:
            continue
        if any(p(peer_caps) for p in preds):
            score += 1
    return score


@dataclass
class Peer:
    peer_id:      str                       # short hash, unique
    name:         str                       # human label ("desk", "laptop")
    host:         str                       # 127.0.0.1 or LAN IP
    port:         int                       # daemon port (default 8742)
    token:        str                       # bearer token for that peer's daemon
    capabilities: dict      = field(default_factory=dict)
    last_seen:    float     = 0.0
    added_at:     float     = field(default_factory=time.time)

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


class PrismMesh:
    def __init__(self, store_path: str = "~/.prism/mesh.json"):
        self._path = Path(store_path).expanduser()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._peers: dict[str, Peer] = {}
        self._load()

    # ------------------------------------------------------------------ #
    # Registry
    # ------------------------------------------------------------------ #

    def register_peer(self, name: str, host: str, port: int, token: str) -> Peer:
        import hashlib
        key = f"{host}:{port}".encode()
        peer_id = hashlib.sha256(key).hexdigest()[:10]
        peer = Peer(peer_id=peer_id, name=name or host, host=host,
                    port=int(port), token=token or "")
        self._peers[peer_id] = peer
        # Best-effort capability refresh — failure is fine, peer is still
        # registered and a later forward can retry.
        try:
            self.refresh_capabilities(peer_id)
        except Exception:
            pass
        self._save()
        return self._peers[peer_id]

    def remove_peer(self, peer_id: str) -> bool:
        if peer_id in self._peers:
            del self._peers[peer_id]
            self._save()
            return True
        return False

    def list_peers(self) -> list[Peer]:
        return list(self._peers.values())

    def get_peer(self, peer_id: str) -> Optional[Peer]:
        return self._peers.get(peer_id)

    def find_peer_by_name(self, name: str) -> Optional[Peer]:
        low = (name or "").strip().lower()
        if not low:
            return None
        for p in self._peers.values():
            if p.name.lower() == low:
                return p
        for p in self._peers.values():
            if low in p.name.lower():
                return p
        return None

    def find_capable_peer(
        self, task: str, params: Optional[dict] = None
    ) -> tuple[Optional[Peer], list[Peer]]:
        """Auto-route by capability match.

        Scores every registered peer with ``score_peer_for_task`` and
        returns ``(best, candidates)`` where:

        * ``best`` is the single highest-scoring peer when it strictly beats
          every other peer (no ties at the top, score > 0).
        * ``candidates`` is the list of peers tied at the maximum score so
          the caller can show "pick one" UX when auto-routing is ambiguous.

        Returns ``(None, [])`` when no peers are registered. Returns
        ``(None, candidates)`` when the best is ambiguous or the score is
        zero — caller chooses fallback policy.
        """
        peers = list(self._peers.values())
        if not peers:
            return None, []
        scored = [(score_peer_for_task(p.capabilities or {}, task, params), p) for p in peers]
        scored.sort(key=lambda t: t[0], reverse=True)
        top_score = scored[0][0]
        if top_score <= 0:
            return None, peers  # no capability signal — caller picks
        top = [p for s, p in scored if s == top_score]
        if len(top) == 1:
            return top[0], top
        return None, top

    # ------------------------------------------------------------------ #
    # Outbound calls
    # ------------------------------------------------------------------ #

    def refresh_capabilities(self, peer_id: str) -> dict:
        peer = self._peers.get(peer_id)
        if peer is None:
            return {}
        data = self._http_json("GET", peer, "/device/capabilities")
        if isinstance(data, dict):
            peer.capabilities = data
            peer.last_seen = time.time()
            self._save()
        return peer.capabilities

    def forward_task(self, peer_id: str, task: str, params: Optional[dict] = None,
                     dry_run: bool = False) -> dict:
        peer = self._peers.get(peer_id)
        if peer is None:
            return {"success": False, "error": f"Unknown peer: {peer_id}"}
        out_params = dict(params or {})
        hop = int(out_params.get("_hop") or 0)
        if hop >= MAX_HOPS:
            return {
                "success": False,
                "error": f"Hop limit reached ({hop}/{MAX_HOPS}); refusing to forward further.",
            }
        out_params["_hop"] = hop + 1
        body = {"task": task, "params": out_params, "dry_run": dry_run}
        data = self._http_json("POST", peer, "/device/execute", body)
        if isinstance(data, dict):
            peer.last_seen = time.time()
            self._save()
            return data
        return {"success": False, "error": "Peer returned no JSON"}

    def forward_chat(self, peer_id: str, message: str, hop: int = 0) -> dict:
        peer = self._peers.get(peer_id)
        if peer is None:
            return {"error": f"Unknown peer: {peer_id}"}
        if hop >= MAX_HOPS:
            return {
                "error": f"Hop limit reached ({hop}/{MAX_HOPS}); refusing to forward further.",
            }
        data = self._http_json("POST", peer, "/chat",
                               {"message": message, "_hop": hop + 1})
        if isinstance(data, dict):
            peer.last_seen = time.time()
            self._save()
            return data
        return {"error": "Peer returned no JSON"}

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _http_json(self, method: str, peer: Peer, path: str,
                   body: Optional[dict] = None) -> Optional[dict]:
        url = peer.base_url + path
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("Content-Type", "application/json")
        if peer.token:
            req.add_header("Authorization", f"Bearer {peer.token}")
        try:
            with urllib.request.urlopen(req, timeout=_DEFAULT_TIMEOUT) as resp:
                raw = resp.read().decode("utf-8", "replace")
                try:
                    return json.loads(raw)
                except json.JSONDecodeError:
                    return {"_raw": raw[:1000]}
        except urllib.error.HTTPError as e:
            try:
                msg = e.read().decode("utf-8", "replace")[:500]
            except Exception:
                msg = str(e)
            return {"success": False, "error": f"HTTP {e.code}: {msg}"}
        except Exception as exc:
            return {"success": False, "error": f"{type(exc).__name__}: {exc}"}

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text())
        except Exception:
            return
        for pid, pd in (raw.get("peers") or {}).items():
            try:
                self._peers[pid] = Peer(
                    peer_id      = pd["peer_id"],
                    name         = pd.get("name", pid),
                    host         = pd["host"],
                    port         = int(pd.get("port", 8742)),
                    token        = pd.get("token", ""),
                    capabilities = pd.get("capabilities", {}) or {},
                    last_seen    = float(pd.get("last_seen", 0) or 0),
                    added_at     = float(pd.get("added_at", time.time())),
                )
            except Exception:
                continue

    def _save(self) -> None:
        payload = {
            "peers": {pid: asdict(p) for pid, p in self._peers.items()},
            "saved_at": time.time(),
        }
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        tmp.replace(self._path)


_MESH: Optional[PrismMesh] = None


def get_mesh() -> PrismMesh:
    global _MESH
    if _MESH is None:
        _MESH = PrismMesh()
    return _MESH
