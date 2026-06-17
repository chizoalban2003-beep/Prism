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
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


_DEFAULT_TIMEOUT = 8.0


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
        body = {"task": task, "params": params or {}, "dry_run": dry_run}
        data = self._http_json("POST", peer, "/device/execute", body)
        if isinstance(data, dict):
            peer.last_seen = time.time()
            self._save()
            return data
        return {"success": False, "error": "Peer returned no JSON"}

    def forward_chat(self, peer_id: str, message: str) -> dict:
        peer = self._peers.get(peer_id)
        if peer is None:
            return {"error": f"Unknown peer: {peer_id}"}
        data = self._http_json("POST", peer, "/chat", {"message": message})
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
