"""
prism_organ_loader.py
=====================
Dynamic organ discovery, loading, and synthesis for PRISM.

Organs are thin Python modules — each handles one intent and returns a
PrismCard. They live in two directories:

  ./organs/           bundled organs shipped with PRISM (version-controlled)
  ~/.prism/organs/    synthesized organs, persisted across sessions

Organ interface
---------------
    ORGAN_META = {
        "intent":      "unique_intent_name",
        "description": "one-line description shown to the LLM router",
        "version":     "1.0",
        # Optional — declares I/O schema so other organs can compose with this
        # one (PowerBI-style arrows). Each field is {name: type_string}. Types
        # are advisory strings, not enforced — they document the contract.
        "inputs":      {"message": "str", "ctx": "dict"},
        "outputs":     {"card": "PrismCard"},
        # Optional — list of capability strings (file_write, network, etc.)
        # used by the policy node and the capability auditor.
        "capabilities": [],
    }

    # Optional — declares risk level so the policy node is self-extending.
    # Organs that omit this fall back to the legacy HIGH_RISK hardcoded set.
    ORGAN_POLICY = {
        "risk_level":        "low",      # "low" | "medium" | "high" | "critical"
        "requires_approval": False,      # True → policy node flags before repeat
        "irreversible":      False,      # True → extra warning in chain context
        "max_per_session":   None,       # int → hard cap per session; None = unlimited
    }

    def execute(intent: str, message: str, ctx: dict):
        from prism_responses import text_card
        ...
        return text_card(result_string, intent)

On load, OrganLoader validates each file with the same AST safety visitor
used in prism_autonomous. Unsafe files are logged and skipped; they never
reach exec().

On synthesize(), the LLM generates a complete organ file, the safety
visitor runs before any code executes, and the file is saved to
~/.prism/organs/ for reuse in future sessions.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import logging
import py_compile
import time
from pathlib import Path
from typing import Any, Callable, Optional  # noqa: F401

logger = logging.getLogger(__name__)

BUNDLED_DIR = Path(__file__).parent / "organs"
USER_DIR    = Path("~/.prism/organs").expanduser()

_INDEX_FILE    = "index.json"
_INDEX_VERSION = 1

# ── AST safety ───────────────────────────────────────────────────────────────

_BLOCKED_IMPORTS = {
    "os", "subprocess", "shutil", "socket", "ctypes",
    "multiprocessing", "importlib", "builtins", "pty",
}
_BLOCKED_CALLS = {"eval", "exec", "compile", "__import__", "breakpoint", "open"}
_BLOCKED_ATTRS = {
    "system", "popen", "remove", "unlink", "rmtree", "rmdir", "chmod", "chown",
    "rename", "symlink", "fork", "spawn", "execv", "execve", "kill",
    # Sandbox-escape vectors via the type/MRO chain or func.__globals__.
    "__mro__", "__subclasses__", "__bases__", "__globals__", "__class__",
    # "replace" intentionally omitted — str.replace() is safe and commonly used.
    # Bundled organs (./organs/) are version-controlled and may legitimately
    # call Path.write_text / write_bytes, so those are NOT in this set.
}
# Stricter set applied at synthesize() time: an LLM-generated organ must not
# write arbitrary files to disk, even into the user's home directory.
_BLOCKED_ATTRS_STRICT = _BLOCKED_ATTRS | {"write_text", "write_bytes", "write"}
# Bare-name Load references to these are flagged so that `e = eval; e('1')`
# and `getattr(__builtins__, 'exec')(...)` can't bypass visit_Call.
_BLOCKED_NAME_LOADS = _BLOCKED_CALLS | {"globals", "vars", "__builtins__"}


class _SafetyVisitor(ast.NodeVisitor):
    def __init__(self, blocked_attrs: set[str] = _BLOCKED_ATTRS):
        self.violations: list[str] = []
        self._blocked_attrs = blocked_attrs

    def visit_Import(self, node):
        for alias in node.names:
            if alias.name.split(".")[0] in _BLOCKED_IMPORTS:
                self.violations.append(f"blocked import: {alias.name}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if (node.module or "").split(".")[0] in _BLOCKED_IMPORTS:
            self.violations.append(f"blocked from-import: {node.module}")
        self.generic_visit(node)

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id in _BLOCKED_CALLS:
            self.violations.append(f"blocked call: {node.func.id}()")
        elif isinstance(node.func, ast.Attribute) and node.func.attr in self._blocked_attrs:
            self.violations.append(f"blocked attr call: .{node.func.attr}()")
        self.generic_visit(node)

    def visit_Attribute(self, node):
        if node.attr in self._blocked_attrs:
            self.violations.append(f"blocked attribute: .{node.attr}")
        self.generic_visit(node)

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load) and node.id in _BLOCKED_NAME_LOADS:
            self.violations.append(f"blocked name reference: {node.id}")
        self.generic_visit(node)


def _is_safe(code: str, strict: bool = False) -> tuple[bool, str]:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return False, f"SyntaxError: {exc}"
    v = _SafetyVisitor(_BLOCKED_ATTRS_STRICT if strict else _BLOCKED_ATTRS)
    v.visit(tree)
    return (False, "; ".join(v.violations)) if v.violations else (True, "")


# ── Capability audit for synthesized organs ───────────────────────────────────
# Map constitutionally sensitive capabilities to code-level signals.
# If a signal appears in synthesized code but the capability isn't declared in
# ORGAN_META["capabilities"], we flag it — and block critical ones.

_CAPABILITY_SIGNALS: dict[str, list[str]] = {
    "shell_execution": ["shell_run", "subprocess", "os.system", "popen", "pty"],
    "file_write":      ["write_text", "write_bytes", ".write(", "open(", "fh.write"],
    "network":         ["urllib.request", "urlopen", "http.client", "requests."],
    "telephony":       ["twilio", "phone_call", "make_call", "plivo"],
    "email":           ["smtplib", "email_send", "sendmail"],
    # PrismCard.body is rendered unescaped in the UI; raw HTML/JS strings
    # injected into the body field can mutate the running frontend. Any organ
    # that emits one must declare this capability so the policy node can gate it.
    "frontend_mutate": ["<script", "<iframe", "onclick=", "onerror=", "javascript:", "innerHTML"],
}

_CRITICAL_CAPABILITIES = frozenset({"shell_execution", "telephony", "frontend_mutate"})


def _audit_capability_gap(code: str, declared: set[str]) -> list[str]:
    """
    Return list of capability names that appear to be used in *code* but are
    absent from *declared*.  Does a simple substring scan — fast and
    conservative (may have false positives, but never false negatives for the
    signals listed).
    """
    warnings = []
    for cap, signals in _CAPABILITY_SIGNALS.items():
        if cap not in declared:
            for sig in signals:
                if sig in code:
                    warnings.append(cap)
                    break
    return warnings


# ── Synthesis prompt ──────────────────────────────────────────────────────────

_SYNTHESIS_PROMPT = """\
You are writing a PRISM organ module. An organ handles exactly one intent.

Intent to implement: "{intent}"
Example user request: "{message}"

Return ONLY valid JSON with this shape:
{{
  "intent":      "{intent}",
  "description": "one sentence what this organ does",
  "code": "complete Python source as a single string"
}}

The code string MUST contain exactly this structure:

ORGAN_META = {{
    "intent":      "{intent}",
    "description": "...",
    "version":     "1.0",
    # Optional I/O schema — enables composition with other organs.
    # Default contract is message+ctx → card; override only if you produce
    # structured data another organ would consume.
    "inputs":      {{"message": "str", "ctx": "dict"}},
    "outputs":     {{"card": "PrismCard"}},
}}

def execute(intent: str, message: str, ctx: dict):
    from prism_responses import text_card
    # implementation
    return text_card(result_string, intent)

Constraints:
- Allowed: json, re, datetime, pathlib, urllib.request, urllib.parse,
  urllib.error, base64, hashlib, math, random, time, collections, html
- FORBIDDEN: os, subprocess, shutil, socket, ctypes, eval, exec, any file
  write — open(), Path.write_text, Path.write_bytes, fh.write — all banned
- Never raise — catch all exceptions and return text_card with error message
- API keys: read from ctx.get("{intent}_key", "") or ctx.get("api_key", "")
- Keep code under 80 lines
"""


# ── OrganLoader ───────────────────────────────────────────────────────────────


class OrganLoader:
    """
    Discovers, loads, synthesizes, and registers PRISM logic organs.

    Usage
    -----
        loader = OrganLoader(llm_router=router)
        fn = loader.get("weather_check")
        if fn:
            card = fn("weather_check", "London weather", {})

        # Synthesize and register a new organ for an unknown intent:
        success = loader.synthesize("stock_price", "what is AAPL stock price?")
    """

    def __init__(
        self,
        bundled_dir: Optional[Path] = None,
        user_dir: Optional[Path]    = None,
        llm_router: Optional[Any]             = None,
    ):
        self._bundled = Path(bundled_dir) if bundled_dir else BUNDLED_DIR
        self._user    = Path(user_dir).expanduser() if user_dir else USER_DIR
        # Quarantine: synthesised organs land here for human review before
        # promote_quarantined() moves them into self._user and registers them.
        self._quarantine = self._user.parent / "organs.quarantine"
        self._router  = llm_router
        # {intent: (execute_fn, organ_meta_dict)}
        self._organs:  dict[str, tuple[Callable, dict]] = {}
        self._disabled: set[str] = set()
        self._organ_sources: dict[str, str] = {}  # intent → "bundled" | "user"
        # {intent: {path, hash, version, description, compiled, safe, created_at}}
        self._index: dict[str, dict] = {}
        # {filename: reason} — organ files present on disk that did NOT
        # register in the last load pass (unsafe AST, import error, no
        # execute()). Basis for health_report().
        self._skipped: dict[str, str] = {}
        # {filename: intent} — which intent each successfully loaded file
        # registered under (intent may differ from the file stem).
        self._file_intents: dict[str, str] = {}
        self._user.mkdir(parents=True, exist_ok=True)
        self._load_all()

    # ── Public API ────────────────────────────────────────────────────────────

    def get(self, intent: str) -> Optional[Callable]:
        """Return the execute function for intent, or None if not loaded."""
        if intent in self._disabled:
            return None
        entry = self._organs.get(intent)
        return entry[0] if entry else None

    def get_organ_policy(self, intent: str) -> dict:
        """Return the ORGAN_POLICY dict for intent, or {} if not declared."""
        entry = self._organs.get(intent)
        if entry is None:
            return {}
        fn = entry[0]
        return getattr(fn, "_organ_policy", {})

    def get_organ_capabilities(self, intent: str) -> list:
        """Return the capabilities list declared in ORGAN_META, or [] if absent."""
        entry = self._organs.get(intent)
        if entry is None:
            return []
        fn = entry[0]
        return list(getattr(fn, "_organ_meta", {}).get("capabilities", []))

    def get_organ_schema(self, intent: str) -> dict:
        """
        Return the I/O schema declared in ORGAN_META as
        ``{"inputs": {...}, "outputs": {...}}``.

        Missing fields default to the standard organ contract:
        inputs={"message": "str", "ctx": "dict"}, outputs={"card": "PrismCard"}.
        Used by the chain planner to wire organs together (PowerBI-style arrows).
        """
        entry = self._organs.get(intent)
        if entry is None:
            return {}
        meta = getattr(entry[0], "_organ_meta", {})
        return {
            "inputs":  dict(meta.get("inputs",  {"message": "str", "ctx": "dict"})),
            "outputs": dict(meta.get("outputs", {"card": "PrismCard"})),
        }

    def composable_with(self, producer: str, consumer: str) -> bool:
        """
        Return True if at least one *output* of producer matches an *input*
        type of consumer — i.e. an arrow can be drawn from producer → consumer.
        Type comparison is exact string match; advisory only.
        """
        p = self.get_organ_schema(producer).get("outputs", {})
        c = self.get_organ_schema(consumer).get("inputs",  {})
        if not p or not c:
            return False
        out_types = set(p.values())
        in_types  = set(c.values())
        return bool(out_types & in_types)

    def known_intents(self) -> dict[str, str]:
        """Return {intent: description} for every loaded organ."""
        return {k: v[1].get("description", k) for k, v in self._organs.items()}

    def organ_source(self, intent: str) -> Optional[str]:
        """Return the raw Python source for an organ, or None if not on disk.

        Resolution order: indexed user-organ path → ``<user>/<intent>.py`` →
        ``<bundled>/<intent>.py``. Used by the Organ-Pack exporter so an organ
        can be shared with another PRISM instance.
        """
        candidates = []
        entry = self._index.get(intent)
        if entry and entry.get("path"):
            candidates.append(self._user / str(entry["path"]))
        candidates.append(self._user / f"{intent}.py")
        candidates.append(self._bundled / f"{intent}.py")
        for path in candidates:
            try:
                if path.exists():
                    return path.read_text()
            except Exception:
                continue
        return None

    def list_organs(self) -> list[str]:
        """Return sorted list of loaded organ intent names."""
        return sorted(self._organs.keys())

    def enable(self, intent: str) -> bool:
        """Enable a previously disabled organ. Returns True if it was disabled."""
        if intent in self._disabled:
            self._disabled.discard(intent)
            return True
        return False

    def disable(self, intent: str) -> bool:
        """Disable an organ without unloading it. Returns True if found and disabled."""
        if intent in self._organs:
            self._disabled.add(intent)
            return True
        return False

    def is_enabled(self, intent: str) -> bool:
        return intent in self._organs and intent not in self._disabled

    def organ_details(self, intent: str) -> dict | None:
        """Return full metadata dict for intent, or None if not found."""
        entry = self._organs.get(intent)
        if entry is None:
            return None
        fn, meta = entry
        policy = getattr(fn, "_organ_policy", {})
        return {
            "intent":      intent,
            "description": meta.get("description", intent),
            "version":     meta.get("version", "1.0"),
            "source":      self._organ_sources.get(intent, "bundled"),
            "enabled":     intent not in self._disabled,
            "risk_level":  policy.get("risk_level", "unknown"),
            "requires_approval": policy.get("requires_approval", False),
            "irreversible":      policy.get("irreversible", False),
            "max_per_session":   policy.get("max_per_session", None),
            "capabilities": meta.get("capabilities", []),
            "inputs":       dict(meta.get("inputs",  {"message": "str", "ctx": "dict"})),
            "outputs":      dict(meta.get("outputs", {"card": "PrismCard"})),
        }

    def list_organ_details(self) -> list[dict]:
        """Return organ_details for all known intents, sorted by intent name."""
        return [d for i in sorted(self._organs.keys()) if (d := self.organ_details(i)) is not None]

    def index_status(self) -> dict:
        """
        Return the current in-memory index for user organs.

        Keys: ``version``, ``entry_count``, ``compiled_count``, ``entries``
        (a copy of the index dict so callers cannot mutate internal state).
        """
        compiled = sum(1 for e in self._index.values() if e.get("compiled"))
        return {
            "version":        _INDEX_VERSION,
            "entry_count":    len(self._index),
            "compiled_count": compiled,
            "entries":        {k: dict(v) for k, v in self._index.items()},
        }

    def health_report(self) -> dict:
        """
        Self-check: does every organ file on disk correspond to a registered
        organ?  Catches the whole class of "file ships, tests pass, daemon
        silently skips it" failures (unsafe AST, import error, missing
        execute()) — not just the AST case the CI gate covers.
        """
        report: dict = {"directories": {}, "disabled": sorted(self._disabled)}
        missing_total = 0
        for directory, source in ((self._bundled, "bundled"), (self._user, "user")):
            if not directory.exists():
                report["directories"][source] = {
                    "path": str(directory), "files": 0, "registered": 0,
                    "missing": [],
                }
                continue
            files = sorted(
                p.name for p in directory.glob("*.py")
                if not p.name.startswith("_")
            )
            missing = [
                {"file": f, "reason": self._skipped.get(f, "not registered")}
                for f in files
                if self._file_intents.get(f) not in self._organs
            ]
            missing_total += len(missing)
            report["directories"][source] = {
                "path":       str(directory),
                "files":      len(files),
                "registered": len(files) - len(missing),
                "missing":    missing,
            }
        report["total_registered"] = len(self._organs)
        report["ok"] = missing_total == 0
        return report

    def reload(self) -> int:
        """Re-scan bundled and user dirs. Returns number of organs now loaded."""
        self._organs.clear()
        self._organ_sources.clear()
        self._index.clear()
        # Keep disabled set intact so user's choices persist across reload
        self._load_all()
        return len(self._organs)

    def delete_user_organ(self, intent: str) -> bool:
        """
        Delete a user-synthesized organ from disk and unregister it.
        Bundled organs cannot be deleted (returns False).
        Returns True on success.
        """
        if self._organ_sources.get(intent) != "user":
            return False
        path = self._user / f"{intent}.py"
        # Remove compiled bytecode if present
        try:
            pyc = Path(importlib.util.cache_from_source(str(path)))
            pyc.unlink(missing_ok=True)
        except Exception:
            pass
        path.unlink(missing_ok=True)
        self._organs.pop(intent, None)
        self._organ_sources.pop(intent, None)
        self._disabled.discard(intent)
        # Remove from index and persist
        if intent in self._index:
            del self._index[intent]
            self._save_index()
        # Also remove from LOGIC_REGISTRY if present
        try:
            from prism_composer import LOGIC_REGISTRY
            LOGIC_REGISTRY.pop(intent, None)
        except ImportError:
            pass
        return True

    def execute_parallel(
        self,
        intents: list[str],
        message: str,
        ctx: dict,
        timeout: float = 30.0,
    ) -> dict[str, Any]:
        """
        Run multiple organs concurrently. Only organs whose ORGAN_POLICY marks
        irreversible=False and requires_approval=False are eligible; others are
        silently skipped (call execute() for those individually).

        Returns {intent: card_or_error_dict}.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from concurrent.futures import TimeoutError as _FTimeout

        safe = [
            i for i in intents
            if not self.get_organ_policy(i).get("irreversible", False)
            and not self.get_organ_policy(i).get("requires_approval", False)
            and self.get(i) is not None
        ]
        if not safe:
            return {}

        results: dict[str, Any] = {}
        with ThreadPoolExecutor(max_workers=min(len(safe), 4)) as pool:
            futures = {pool.submit(fn, i, message, ctx): i for i in safe if (fn := self.get(i)) is not None}
            try:
                for future in as_completed(futures, timeout=timeout):
                    intent = futures[future]
                    try:
                        results[intent] = future.result(timeout=5.0)
                    except _FTimeout:
                        results[intent] = {"error": "timeout", "output": ""}
                    except Exception as exc:
                        results[intent] = {"error": str(exc), "output": ""}
            except _FTimeout:
                for _future, intent in futures.items():
                    if intent not in results:
                        results[intent] = {"error": "overall timeout", "output": ""}
        return results

    # Names the router classifier uses as sentinels — never let synthesis
    # poison these with a cached organ, or every unmapped request will hit it.
    _RESERVED_INTENTS = frozenset({"novel_capability", "general_chat", "chat"})

    def synthesize(self, intent: str, message: str, *,
                   quarantine: bool = False) -> bool:
        """
        Ask the LLM to write a new organ for this intent, safety-check it,
        save to ~/.prism/organs/<intent>.py, and register it immediately.

        When ``quarantine=True``, the synthesised code lands in
        ~/.prism/organs.quarantine/<intent>.py and is NOT registered. Use
        list_quarantined() to inspect, promote_quarantined() to install, or
        discard_quarantined() to drop. Promote re-runs the safety pipeline
        so on-disk tampering between synthesis and review is caught.

        Returns True if the organ was successfully synthesised (quarantined
        or registered, depending on the flag).
        """
        if intent in self._RESERVED_INTENTS:
            logger.warning(
                "[organ_loader] Refusing to synthesize reserved intent %r — "
                "this name is a router sentinel, caching it would mis-route "
                "every unmapped request.", intent)
            return False
        if not self._router:
            logger.warning("[organ_loader] No router — cannot synthesize %s", intent)
            return False

        # L1 constitution: block synthesis of forbidden capabilities
        # and intent-name patterns (catches LLM-coined aliases like
        # "run_shell" that the capability table wouldn't know).
        try:
            from prism_constitution import get_guard
            guard = get_guard()
            allowed, pattern = guard.may_synthesize_intent(intent)
            if not allowed:
                logger.warning(
                    "[organ_loader] Synthesis blocked for %s — intent name "
                    "matches forbidden pattern %r", intent, pattern)
                return False
            required = guard.required_capabilities(intent)
            blocked = [c for c in required if not guard.may_synthesize(c)]
            if blocked:
                logger.warning(
                    "[organ_loader] Synthesis blocked for %s — capabilities %s "
                    "are forbidden by constitution", intent, blocked)
                return False
        except Exception:
            pass

        prompt = _SYNTHESIS_PROMPT.format(intent=intent, message=message[:300])
        try:
            raw, _ = self._router.call(
                prompt, min_capability=2, max_tokens=1400, json_mode=True)
        except Exception as exc:
            logger.warning("[organ_loader] LLM call failed for %s: %s", intent, exc)
            return False

        data = self._parse_json(raw)
        if data is None:
            return False

        code = data.get("code", "")
        if not code or "def execute" not in code or "ORGAN_META" not in code:
            logger.warning("[organ_loader] Synthesized code missing interface for %s", intent)
            return False

        safe, reason = _is_safe(code, strict=True)
        if not safe:
            logger.warning("[organ_loader] Unsafe organ blocked (%s): %s", intent, reason)
            return False

        # Cross-check declared capabilities vs. actual code signals.
        # Critical undeclared capabilities (shell, telephony) are hard-blocked.
        _declared_caps: set[str] = set(data.get("capabilities", []) or [])
        _cap_gaps = _audit_capability_gap(code, _declared_caps)
        if _cap_gaps:
            logger.warning("[organ_loader] Undeclared capabilities in %s: %s", intent, _cap_gaps)
            _critical = [c for c in _cap_gaps if c in _CRITICAL_CAPABILITIES]
            if _critical:
                logger.warning(
                    "[organ_loader] Synthesis blocked — critical undeclared capabilities %s in %s",
                    _critical, intent,
                )
                return False

        if quarantine:
            self._quarantine.mkdir(parents=True, exist_ok=True)
            out_path = self._quarantine / f"{intent}.py"
            out_path.write_text(code)
            meta_sidecar = {
                "intent":                intent,
                "description":           data.get("description", intent),
                "declared_capabilities": sorted(_declared_caps),
                "audit_gaps":            sorted(_cap_gaps),
                "synthesised_at":        time.time(),
                "source_message":        message[:300],
            }
            (self._quarantine / f"{intent}.meta.json").write_text(
                json.dumps(meta_sidecar, indent=2))
            logger.info(
                "[organ_loader] Synthesised organ QUARANTINED at %s — "
                "review and call promote_quarantined(%r) to install",
                out_path, intent)
            return True

        out_path = self._user / f"{intent}.py"
        out_path.write_text(code)
        logger.info("[organ_loader] Synthesized organ saved: %s", out_path)

        compiled = self._compile_organ(out_path)

        fn = self._load_file(out_path, trusted=True)
        if fn is None:
            out_path.unlink(missing_ok=True)
            self._index.pop(intent, None)
            self._save_index()
            return False

        meta = {
            "intent":      intent,
            "description": data.get("description", intent),
            "version":     "1.0",
        }
        self._register(intent, fn, meta, source="user")
        # Index entry already written by _register; update compiled flag
        if intent in self._index:
            self._index[intent]["compiled"] = compiled
            self._save_index()
        return True

    # ── Quarantine API ────────────────────────────────────────────────────────

    def list_quarantined(self) -> list[dict]:
        """Return metadata for synthesised organs awaiting human review."""
        out: list[dict] = []
        if not self._quarantine.exists():
            return out
        for path in sorted(self._quarantine.glob("*.py")):
            meta_path = path.with_suffix(".meta.json")
            entry: dict = {"intent": path.stem, "path": str(path)}
            if meta_path.exists():
                try:
                    entry.update(json.loads(meta_path.read_text()))
                except Exception:
                    pass
            try:
                entry["code_preview"] = path.read_text()[:500]
            except Exception:
                entry["code_preview"] = ""
            out.append(entry)
        return out

    def promote_quarantined(self, intent: str) -> bool:
        """Move a quarantined organ into the user dir and register it.

        Re-runs the AST safety check on the on-disk content so any
        tampering between synthesis and review is rejected.
        """
        if intent in self._RESERVED_INTENTS:
            logger.warning(
                "[organ_loader] Refusing to promote reserved intent %r", intent)
            return False
        quar_path = self._quarantine / f"{intent}.py"
        meta_path = quar_path.with_suffix(".meta.json")
        if not quar_path.exists():
            return False
        try:
            code = quar_path.read_text()
        except Exception as exc:
            logger.warning(
                "[organ_loader] Cannot read quarantined %s: %s", intent, exc)
            return False
        safe, reason = _is_safe(code, strict=True)
        if not safe:
            logger.warning(
                "[organ_loader] Promote blocked — quarantined %s now unsafe: %s",
                intent, reason)
            return False
        out_path = self._user / f"{intent}.py"
        out_path.write_text(code)
        compiled = self._compile_organ(out_path)
        fn = self._load_file(out_path, trusted=True)
        if fn is None:
            out_path.unlink(missing_ok=True)
            return False
        description = intent
        if meta_path.exists():
            try:
                meta_data = json.loads(meta_path.read_text())
                description = meta_data.get("description", intent)
            except Exception:
                pass
        self._register(intent, fn,
                       {"intent": intent, "description": description, "version": "1.0"},
                       source="user")
        if intent in self._index:
            self._index[intent]["compiled"] = compiled
            self._save_index()
        quar_path.unlink(missing_ok=True)
        if meta_path.exists():
            meta_path.unlink(missing_ok=True)
        logger.info("[organ_loader] Promoted quarantined organ: %s", intent)
        return True

    def discard_quarantined(self, intent: str) -> bool:
        """Drop a quarantined organ without installing it."""
        quar_path = self._quarantine / f"{intent}.py"
        meta_path = quar_path.with_suffix(".meta.json")
        existed = quar_path.exists()
        quar_path.unlink(missing_ok=True)
        if meta_path.exists():
            meta_path.unlink(missing_ok=True)
        return existed

    def install_bundle(self, intent: str, code: str) -> bool:
        """
        Install a third-party organ bundle. Same safety guarantees as
        synthesize() — strict AST scan, capability auditor, critical-cap
        block — but the LLM is not involved; caller supplies the code.

        Returns True on successful install + hot-register. The caller is
        responsible for SHA256 verification before calling this method (the
        HTTP layer does that check).
        """
        if intent in self._RESERVED_INTENTS:
            logger.warning(
                "[organ_loader] Refusing to install bundle for reserved intent %r",
                intent,
            )
            return False
        if not isinstance(code, str) or "def execute" not in code or "ORGAN_META" not in code:
            logger.warning("[organ_loader] Install bundle missing interface for %s", intent)
            return False

        safe, reason = _is_safe(code, strict=True)
        if not safe:
            logger.warning("[organ_loader] Unsafe bundle blocked (%s): %s", intent, reason)
            return False

        try:
            tree = ast.parse(code)
            declared: set[str] = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id == "ORGAN_META" for t in node.targets
                ) and isinstance(node.value, ast.Dict):
                    for k, v in zip(node.value.keys, node.value.values):
                        if isinstance(k, ast.Constant) and k.value == "capabilities":
                            if isinstance(v, ast.List):
                                for elt in v.elts:
                                    if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                        declared.add(elt.value)
        except Exception:
            declared = set()

        gaps = _audit_capability_gap(code, declared)
        if gaps:
            critical = [c for c in gaps if c in _CRITICAL_CAPABILITIES]
            if critical:
                logger.warning(
                    "[organ_loader] Install blocked — critical undeclared capabilities %s in bundle %s",
                    critical, intent,
                )
                return False
            logger.warning(
                "[organ_loader] Install proceeding despite undeclared (non-critical) capabilities %s in %s",
                gaps, intent,
            )

        out_path = self._user / f"{intent}.py"
        out_path.write_text(code)
        logger.info("[organ_loader] Bundle installed: %s", out_path)

        compiled = self._compile_organ(out_path)

        fn = self._load_file(out_path, trusted=True)
        if fn is None:
            out_path.unlink(missing_ok=True)
            self._index.pop(intent, None)
            self._save_index()
            return False

        meta = getattr(fn, "_organ_meta", {}) or {}
        meta.setdefault("intent", intent)
        meta.setdefault("description", intent)
        meta.setdefault("version", "1.0")
        self._register(intent, fn, meta, source="user")
        if intent in self._index:
            self._index[intent]["compiled"] = compiled
            self._index[intent]["source"]   = "installed"
            self._save_index()
        return True

    # ── Loading ───────────────────────────────────────────────────────────────

    def _load_all(self):
        self._index = self._load_index()
        self._skipped.clear()
        self._file_intents.clear()
        index_dirty = False

        for directory, source in ((self._bundled, "bundled"), (self._user, "user")):
            if not directory.exists():
                continue
            for path in sorted(directory.glob("*.py")):
                if path.name.startswith("_"):
                    continue

                # For user organs: skip AST re-scan when hash matches cached entry
                trusted = False
                if source == "user":
                    file_hash = self._file_hash(path)
                    cached = self._index.get(path.stem, {})
                    trusted = (
                        cached.get("hash") == file_hash
                        and cached.get("safe", False)
                    )

                fn = self._load_file(path, trusted=trusted, strict=(source == "user"))
                if fn is None:
                    continue
                meta   = getattr(fn, "_organ_meta", {})
                intent = meta.get("intent") or path.stem
                self._file_intents[path.name] = intent
                self._register(intent, fn, meta, source=source)

                if source == "user":
                    fh = file_hash if trusted else self._file_hash(path)
                    existing = self._index.get(intent, {})
                    new_entry = {
                        "path":        path.name,
                        "version":     meta.get("version", "1.0"),
                        "description": meta.get("description", intent),
                        "hash":        fh,
                        "compiled":    self._pyc_is_current(path),
                        "safe":        True,
                        "source":      "user",
                        "created_at":  existing.get("created_at", time.time()),
                    }
                    if existing != new_entry:
                        self._index[intent] = new_entry
                        index_dirty = True

        if index_dirty:
            self._save_index()

        if self._skipped:
            logger.warning(
                "[organ_loader] HEALTH: %d organ file(s) on disk did NOT "
                "register: %s — see GET /organs/health for details.",
                len(self._skipped), ", ".join(sorted(self._skipped)),
            )

    def _load_file(self, path: Path, trusted: bool = False, strict: bool = False) -> Optional[Callable]:
        """Load an organ from *path*.  When *trusted* is True, skip AST safety
        scan. When *strict* is True (user organs), also reject file-write
        operations (write_text/write_bytes/write) in addition to the base set.
        """
        if not trusted:
            code = path.read_text()
            safe, reason = _is_safe(code, strict=strict)
            if not safe:
                logger.warning("[organ_loader] Skipping unsafe organ %s: %s",
                               path.name, reason)
                self._skipped[path.name] = f"unsafe: {reason}"
                return None
        try:
            spec   = importlib.util.spec_from_file_location(
                f"_organ_{path.stem}", path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Cannot load spec for {path}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        except Exception as exc:
            logger.warning("[organ_loader] Failed to load %s: %s", path.name, exc)
            self._skipped[path.name] = f"import error: {exc}"
            return None
        fn = getattr(module, "execute", None)
        if not callable(fn):
            logger.warning("[organ_loader] No execute() in %s", path.name)
            self._skipped[path.name] = "no execute() function"
            return None
        self._skipped.pop(path.name, None)
        fn._organ_meta   = getattr(module, "ORGAN_META", {})
        fn._organ_policy = getattr(module, "ORGAN_POLICY", {})
        return fn

    def _register(self, intent: str, fn: Callable, meta: dict, source: str = "bundled"):
        self._organs[intent] = (fn, meta)
        self._organ_sources[intent] = source
        # Dynamically extend LOGIC_REGISTRY so the chain router sees the new organ
        try:
            from prism_composer import LOGIC_REGISTRY
            if intent not in LOGIC_REGISTRY:
                LOGIC_REGISTRY[intent] = meta.get("description", intent)
                logger.debug("[organ_loader] Added %s to LOGIC_REGISTRY", intent)
        except ImportError:
            pass

    # ── Index ─────────────────────────────────────────────────────────────────

    def _load_index(self) -> dict:
        path = self._user / _INDEX_FILE
        try:
            data = json.loads(path.read_text())
            if isinstance(data, dict) and data.get("version") == _INDEX_VERSION:
                return data.get("entries", {})
        except Exception:
            pass
        return {}

    def _save_index(self) -> None:
        path = self._user / _INDEX_FILE
        try:
            path.write_text(json.dumps(
                {"version": _INDEX_VERSION, "entries": self._index},
                indent=2,
            ))
        except Exception as exc:
            logger.debug("[organ_loader] index write failed: %s", exc)

    @staticmethod
    def _file_hash(path: Path) -> str:
        try:
            return hashlib.sha256(path.read_bytes()).hexdigest()
        except Exception:
            return ""

    @staticmethod
    def _compile_organ(path: Path) -> bool:
        """Compile *path* to bytecode. Returns True on success."""
        try:
            py_compile.compile(str(path), doraise=True)
            return True
        except py_compile.PyCompileError as exc:
            logger.debug("[organ_loader] compile failed for %s: %s", path.name, exc)
            return False

    @staticmethod
    def _pyc_is_current(path: Path) -> bool:
        """Return True if a current .pyc bytecode file exists for *path*."""
        try:
            pyc = Path(importlib.util.cache_from_source(str(path)))
            return pyc.exists() and pyc.stat().st_mtime >= path.stat().st_mtime
        except Exception:
            return False

    @staticmethod
    def _parse_json(raw: str) -> Optional[dict]:
        try:
            clean = (
                raw.strip()
                   .lstrip("```json")
                   .lstrip("```")
                   .rstrip("```")
                   .strip()
            )
            return json.loads(clean)
        except Exception:
            return None
