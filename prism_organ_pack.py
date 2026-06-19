"""
prism_organ_pack.py
===================
Portable Organ-Pack share format — PRISM's analog of agentskills.io.

An *organ pack* bundles one or more organs into a single, hash-verified JSON
document that can be exported from one PRISM instance and imported into
another. It is the unit of capability sharing: publish a pack, hand someone a
file or URL, and they gain the same organs you built — without copying loose
``.py`` files around.

Security model
--------------
A pack carries code, so importing is treated exactly like installing a
third-party bundle:

  * every organ's ``code`` is integrity-checked against its recorded
    ``sha256`` (and the whole pack against a canonical ``sha256`` digest), and
  * each organ is installed through :meth:`OrganLoader.install_bundle`, which
    runs the **strict AST safety scan**, the capability auditor, and the
    critical-capability block before the code ever touches disk.

So a malformed or tampered pack fails the hash check, and a malicious-but-
well-formed pack still cannot smuggle ``eval``/``subprocess``/file-writes past
the loader's static analysis.

Format (``prism.organ-pack/v1``)
--------------------------------
::

    {
      "format": "prism.organ-pack/v1",
      "name": "research-tools",
      "version": "1.0",
      "description": "Web + reference lookup organs",
      "author": "alice",
      "created_at": 1750000000.0,
      "sha256": "<digest over the organ set>",
      "organs": [
        {
          "intent": "hacker_news",
          "description": "...",
          "version": "1.0",
          "capabilities": ["internet_read"],
          "risk_level": "low",
          "requires_approval": false,
          "code": "ORGAN_META = {...}\\ndef execute(...): ...",
          "sha256": "<digest over code>"
        },
        ...
      ]
    }

Usage
-----
    from prism_organ_pack import build_pack, import_pack, dumps, loads

    pack = build_pack(loader, ["hacker_news", "currency_convert"],
                      name="research-tools", author="alice")
    open("research-tools.organpack.json", "w").write(dumps(pack))

    # on another machine
    pack = loads(open("research-tools.organpack.json").read())
    report = import_pack(loader, pack)
    # report == {"installed": [...], "skipped": [...], "failed": [...], "ok": True}
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

PACK_FORMAT = "prism.organ-pack/v1"


# ── Hashing helpers ──────────────────────────────────────────────────────────

def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _pack_digest(organs: list[dict]) -> str:
    """Canonical digest over the (intent, code-hash) set, order-independent.

    Sorting by intent makes the digest stable regardless of organ ordering, so
    two packs with the same organs always hash identically.
    """
    pairs = sorted(
        (str(o.get("intent", "")), _sha256(str(o.get("code", "")))) for o in organs
    )
    canonical = json.dumps(pairs, separators=(",", ":"), sort_keys=True)
    return _sha256(canonical)


# ── Build / export ───────────────────────────────────────────────────────────

def build_pack(
    loader: Any,
    intents: list[str],
    *,
    name: str,
    version: str = "1.0",
    description: str = "",
    author: str = "",
) -> dict:
    """Build an organ pack from organs currently loaded by *loader*.

    Raises ``ValueError`` if *intents* is empty or no source can be resolved
    for any requested organ.
    """
    if not name or not str(name).strip():
        raise ValueError("pack 'name' is required")
    if not intents:
        raise ValueError("at least one organ intent is required")

    organs: list[dict] = []
    missing: list[str] = []
    seen: set[str] = set()
    for intent in intents:
        intent = str(intent).strip()
        if not intent or intent in seen:
            continue
        seen.add(intent)
        code = loader.organ_source(intent)
        if not code:
            missing.append(intent)
            continue
        details: dict = {}
        try:
            details = loader.organ_details(intent) or {}
        except Exception:
            details = {}
        organs.append({
            "intent":            intent,
            "description":       details.get("description", intent),
            "version":           details.get("version", "1.0"),
            "capabilities":      list(details.get("capabilities", []) or []),
            "risk_level":        details.get("risk_level", "unknown"),
            "requires_approval": bool(details.get("requires_approval", False)),
            "code":              code,
            "sha256":            _sha256(code),
        })

    if missing:
        raise ValueError(
            "no source available for organ(s): " + ", ".join(sorted(missing))
        )
    if not organs:
        raise ValueError("no exportable organs resolved")

    pack = {
        "format":      PACK_FORMAT,
        "name":        str(name).strip(),
        "version":     str(version),
        "description": str(description),
        "author":      str(author),
        "created_at":  time.time(),
        "organs":      organs,
    }
    pack["sha256"] = _pack_digest(organs)
    return pack


def dumps(pack: dict, *, indent: int = 2) -> str:
    """Serialise a pack to a JSON string."""
    return json.dumps(pack, indent=indent, sort_keys=False)


def loads(text: str) -> dict:
    """Parse a pack from a JSON string. Raises ``ValueError`` on bad JSON."""
    try:
        data = json.loads(text)
    except Exception as exc:
        raise ValueError(f"invalid pack JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("pack must be a JSON object")
    return data


def write_pack(pack: dict, path: str | Path) -> Path:
    """Write a pack to *path* and return the resolved path."""
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(dumps(pack))
    return p


def read_pack(path: str | Path) -> dict:
    """Read and parse a pack file."""
    return loads(Path(path).expanduser().read_text())


# ── Verify ───────────────────────────────────────────────────────────────────

def verify_pack(pack: dict) -> tuple[bool, str]:
    """Return ``(True, "")`` if the pack is well-formed and every hash matches.

    Checks: format tag, organ list shape, per-organ ``sha256`` over its code,
    and the canonical pack ``sha256`` digest (when present).
    """
    if not isinstance(pack, dict):
        return False, "pack must be an object"
    fmt = pack.get("format")
    if fmt != PACK_FORMAT:
        return False, f"unsupported pack format: {fmt!r} (expected {PACK_FORMAT!r})"
    organs = pack.get("organs")
    if not isinstance(organs, list) or not organs:
        return False, "pack has no organs"

    for o in organs:
        if not isinstance(o, dict):
            return False, "organ entry must be an object"
        intent = str(o.get("intent", "")).strip()
        code = o.get("code", "")
        if not intent:
            return False, "organ entry missing 'intent'"
        if not isinstance(code, str) or not code:
            return False, f"organ {intent!r} missing 'code'"
        declared = str(o.get("sha256", "")).strip().lower()
        if not declared:
            return False, f"organ {intent!r} missing 'sha256'"
        actual = _sha256(code)
        if actual != declared:
            return False, f"organ {intent!r} sha256 mismatch (tampered code)"

    declared_pack = str(pack.get("sha256", "")).strip().lower()
    if declared_pack:
        actual_pack = _pack_digest(organs)
        if actual_pack != declared_pack:
            return False, "pack sha256 mismatch (tampered pack)"

    return True, ""


# ── Import ───────────────────────────────────────────────────────────────────

def import_pack(loader: Any, pack: dict, *, overwrite: bool = False) -> dict:
    """Install every organ in *pack* through the loader's safe install path.

    Each organ is verified (sha256) and then handed to
    :meth:`OrganLoader.install_bundle`, which enforces the strict AST scan and
    capability audit. Existing organs are skipped unless ``overwrite=True``.

    Returns a report::

        {"ok": bool, "name": str,
         "installed": [intent, ...],
         "skipped":   [{"intent": ..., "reason": ...}, ...],
         "failed":    [{"intent": ..., "reason": ...}, ...]}
    """
    ok, reason = verify_pack(pack)
    if not ok:
        return {
            "ok": False, "name": pack.get("name", ""),
            "error": reason,
            "installed": [], "skipped": [], "failed": [],
        }

    try:
        existing = set(loader.list_organs())
    except Exception:
        existing = set()

    installed: list[str] = []
    skipped: list[dict] = []
    failed: list[dict] = []

    for o in pack["organs"]:
        intent = str(o["intent"]).strip()
        code = o["code"]
        if intent in existing and not overwrite:
            skipped.append({"intent": intent, "reason": "already installed"})
            continue
        try:
            ok_install = loader.install_bundle(intent, code)
        except Exception as exc:
            failed.append({"intent": intent, "reason": f"install error: {exc}"})
            continue
        if ok_install:
            installed.append(intent)
        else:
            failed.append({
                "intent": intent,
                "reason": "rejected by loader (safety/interface/capability)",
            })

    return {
        "ok": len(failed) == 0,
        "name": pack.get("name", ""),
        "installed": installed,
        "skipped": skipped,
        "failed": failed,
    }


def pack_summary(pack: dict) -> dict:
    """Return a lightweight, code-free summary of a pack for previews/UI."""
    organs = pack.get("organs", []) if isinstance(pack, dict) else []
    return {
        "format":      pack.get("format", ""),
        "name":        pack.get("name", ""),
        "version":     pack.get("version", ""),
        "description": pack.get("description", ""),
        "author":      pack.get("author", ""),
        "created_at":  pack.get("created_at", 0),
        "organ_count": len(organs),
        "organs": [
            {
                "intent":       str(o.get("intent", "")),
                "description":  str(o.get("description", "")),
                "version":      str(o.get("version", "1.0")),
                "capabilities": list(o.get("capabilities", []) or []),
                "risk_level":   str(o.get("risk_level", "unknown")),
            }
            for o in organs if isinstance(o, dict)
        ],
    }
