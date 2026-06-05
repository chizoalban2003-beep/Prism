"""
PRISM LLM Setup Wizard
Run with:  python3 prism_setup_llm.py
Or via:    python3 prism_daemon.py --setup-llm
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path

_CONFIG_PATH = Path(__file__).parent / "prism_config.toml"

# ── colours ──────────────────────────────────────────────────────────────────
_T = sys.stdout.isatty()
def _c(code): return f"\033[{code}m" if _T else ""
G   = _c("92")
R   = _c("31")
Y   = _c("93")
C   = _c("96")
DIM = _c("2")
B   = _c("1")
Z   = _c("0")

# ── TOML helpers ─────────────────────────────────────────────────────────────

def _read_config() -> dict:
    try:
        import tomllib
        if _CONFIG_PATH.exists():
            return tomllib.loads(_CONFIG_PATH.read_text())
    except Exception:
        pass
    return {}


def _write_llm_section(updates: dict) -> None:
    """Rewrite only the [llm] section of prism_config.toml in-place."""
    text = _CONFIG_PATH.read_text() if _CONFIG_PATH.exists() else ""

    # Build new [llm] block
    lines = ["[llm]"]
    keys_order = ["preferred", "fallback", "claude_api_key",
                  "openai_api_key", "openai_host", "ollama_host", "ollama_model"]
    seen = set()
    for k in keys_order:
        if k in updates:
            v = updates[k]
            if isinstance(v, list):
                lines.append(f'{k} = {json.dumps(v)}')
            elif isinstance(v, str):
                lines.append(f'{k} = "{v}"')
            seen.add(k)
    for k, v in updates.items():
        if k not in seen:
            lines.append(f'{k} = "{v}"')
    new_block = "\n".join(lines) + "\n"

    # Replace existing [llm] section or append
    import re
    pattern = re.compile(r'^\[llm\][^\[]*', re.MULTILINE | re.DOTALL)
    m = pattern.search(text)
    if m:
        new_text = text[:m.start()] + new_block + text[m.end():]
    else:
        new_text = text.rstrip("\n") + "\n\n" + new_block

    _CONFIG_PATH.write_text(new_text)


# ── Provider testers ──────────────────────────────────────────────────────────

def _test_claude(api_key: str) -> tuple[bool, str]:
    try:
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=json.dumps({
                "model": "claude-haiku-4-5-20251001",
                "max_tokens": 10,
                "messages": [{"role": "user", "content": "ping"}]
            }).encode(),
            headers={
                "Content-Type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=8) as r:
            data = json.loads(r.read())
            model = data.get("model", "?")
            return True, f"OK — model={model}"
    except urllib.error.HTTPError as e:
        body = e.read().decode()[:120]
        return False, f"HTTP {e.code}: {body}"
    except Exception as e:
        return False, str(e)[:80]


def _test_ollama(host: str) -> tuple[bool, list[str]]:
    try:
        url = host.rstrip("/") + "/api/tags"
        with urllib.request.urlopen(url, timeout=4) as r:
            data = json.loads(r.read())
            models = [m["name"] for m in data.get("models", [])]
            return True, models
    except Exception as e:
        return False, [str(e)[:80]]


def _test_openai_compat(host: str, api_key: str) -> tuple[bool, str]:
    try:
        url = host.rstrip("/") + "/v1/models"
        req = urllib.request.Request(
            url,
            headers={"Authorization": f"Bearer {api_key}"},
        )
        with urllib.request.urlopen(req, timeout=8) as r:
            data = json.loads(r.read())
            ids = [m["id"] for m in data.get("data", [])[:5]]
            return True, f"models: {', '.join(ids) or 'none listed'}"
    except urllib.error.HTTPError as e:
        body = e.read().decode()[:120]
        return False, f"HTTP {e.code}: {body}"
    except Exception as e:
        return False, str(e)[:80]


# ── UI helpers ────────────────────────────────────────────────────────────────

def _hr(): print(f"{C}{'─'*64}{Z}")
def _ok(m): print(f"  {G}✓{Z}  {m}")
def _err(m): print(f"  {R}✗{Z}  {m}")
def _info(m): print(f"  {DIM}   {m}{Z}")
def _ask(prompt, default=""):
    d = f" [{DIM}{default}{Z}]" if default else ""
    sys.stdout.write(f"  {B}>{Z} {prompt}{d}: ")
    sys.stdout.flush()
    try:
        val = input().strip()
    except (EOFError, KeyboardInterrupt):
        print()
        sys.exit(0)
    return val or default


# ── Provider setup flows ───────────────────────────────────────────────────────

def _setup_claude(cfg: dict) -> dict | None:
    existing = cfg.get("llm", {}).get("claude_api_key", "")
    print(f"\n  {B}Claude API (Anthropic){Z}")
    print("  Get a key at: https://console.anthropic.com\n")
    key = _ask("API key (sk-ant-...)", existing)
    if not key:
        _err("No key entered — skipping Claude")
        return None
    print("  Testing... ", end="", flush=True)
    ok, msg = _test_claude(key)
    if ok:
        _ok(msg)
        return {"claude_api_key": key, "preferred": "claude"}
    else:
        _err(msg)
        ans = _ask("Save anyway? (y/N)", "n")
        if ans.lower() == "y":
            return {"claude_api_key": key}
        return None


def _setup_ollama(cfg: dict) -> dict | None:
    existing_host = cfg.get("llm", {}).get("ollama_host", "http://localhost:11434")
    print(f"\n  {B}Ollama (local LLM){Z}")
    print("  Install: https://ollama.ai  then  ollama pull mistral\n")
    host = _ask("Ollama host", existing_host)
    print("  Testing... ", end="", flush=True)
    ok, models = _test_ollama(host)
    if ok:
        _ok(f"running — {len(models)} model(s): {', '.join(models[:5]) or 'none pulled yet'}")
        if not models:
            print(f"\n  {Y}Tip:{Z} run  ollama pull mistral  to download a model")
            model = "mistral"
        else:
            print()
            for i, m in enumerate(models, 1):
                print(f"    [{i}] {m}")
            sel = _ask(f"Model to use [1-{len(models)}]", "1")
            try:
                model = models[int(sel) - 1]
            except Exception:
                model = models[0]
        return {"ollama_host": host, "ollama_model": model, "preferred": f"ollama/{model}"}
    else:
        _err(f"Ollama not reachable: {models[0]}")
        ans = _ask("Save host anyway? (y/N)", "n")
        if ans.lower() == "y":
            return {"ollama_host": host}
        return None


def _setup_openai(cfg: dict) -> dict | None:
    existing = cfg.get("llm", {}).get("openai_api_key", "")
    print(f"\n  {B}OpenAI{Z}")
    print("  Get a key at: https://platform.openai.com/api-keys\n")
    key = _ask("API key (sk-...)", existing)
    if not key:
        _err("No key entered — skipping OpenAI")
        return None
    print("  Testing... ", end="", flush=True)
    ok, msg = _test_openai_compat("https://api.openai.com", key)
    if ok:
        _ok(msg)
        return {"openai_api_key": key, "openai_host": "https://api.openai.com",
                "preferred": "openai"}
    else:
        _err(msg)
        ans = _ask("Save anyway? (y/N)", "n")
        if ans.lower() == "y":
            return {"openai_api_key": key, "openai_host": "https://api.openai.com"}
        return None


def _setup_compat(cfg: dict) -> dict | None:
    existing_host = cfg.get("llm", {}).get("openai_host", "")
    existing_key  = cfg.get("llm", {}).get("openai_api_key", "")
    print(f"\n  {B}OpenAI-compatible endpoint{Z}")
    print("  Supports: Groq, Together AI, LM Studio, llama.cpp, Gemini, Mistral AI\n")
    print("  Examples:")
    print(f"    {DIM}Groq     https://api.groq.com{Z}")
    print(f"    {DIM}Together https://api.together.xyz{Z}")
    print(f"    {DIM}LM Studio http://localhost:1234{Z}")
    print(f"    {DIM}Gemini   https://generativelanguage.googleapis.com/v1beta/openai{Z}\n")
    host = _ask("Endpoint URL", existing_host or "http://localhost:1234")
    key  = _ask("API key (or 'local' for keyless)", existing_key or "local")
    if key == "local":
        key = "local"
    print("  Testing... ", end="", flush=True)
    ok, msg = _test_openai_compat(host, key)
    if ok:
        _ok(msg)
        return {"openai_api_key": key, "openai_host": host, "preferred": "openai_compat"}
    else:
        _err(msg)
        ans = _ask("Save anyway? (y/N)", "n")
        if ans.lower() == "y":
            return {"openai_api_key": key, "openai_host": host}
        return None


# ── Main wizard ────────────────────────────────────────────────────────────────

def run_wizard() -> None:
    _hr()
    print(f"\n  {B}{C}PRISM — LLM Setup{Z}")
    print("  Connect an LLM to power PRISM's reasoning chains.\n")

    cfg = _read_config()
    llm = cfg.get("llm", {})

    # Auto-detect
    print(f"  {B}Detecting...{Z}")
    claude_ok = False
    ollama_ok = False
    oai_ok    = False

    api_key = llm.get("claude_api_key") or os.environ.get("ANTHROPIC_API_KEY", "")
    if api_key:
        ok, _ = _test_claude(api_key)
        claude_ok = ok

    ollama_host = llm.get("ollama_host", "http://localhost:11434")
    ok, models = _test_ollama(ollama_host)
    ollama_ok = ok

    oai_key = llm.get("openai_api_key") or os.environ.get("OPENAI_API_KEY", "")
    oai_host = llm.get("openai_host", "https://api.openai.com")
    if oai_key:
        ok, _ = _test_openai_compat(oai_host, oai_key)
        oai_ok = ok

    print()
    if ollama_ok:
        _ok(f"Ollama at {ollama_host}  —  running, {len(models)} model(s)")
    else:
        _err(f"Ollama at {ollama_host}  —  not running")
    if claude_ok:
        _ok("Claude API key found and valid")
    elif api_key:
        _info("Claude API key set (not tested)")
    else:
        _err("Claude API key not set")
    if oai_ok:
        _ok("OpenAI key found and valid")
    elif oai_key:
        _info("OpenAI key set (not tested)")
    else:
        _err("OpenAI key not set")

    print(f"\n  {B}Choose a provider:{Z}")
    print("    [1]  Ollama (local)              — free, private, needs install")
    print("    [2]  Claude API (Anthropic)       — best quality, needs API key")
    print("    [3]  OpenAI                       — GPT-4o, needs API key")
    print("    [4]  OpenAI-compatible            — Groq, Gemini, LM Studio, Together")
    print("    [5]  Fallback chain               — set preferred + ordered fallbacks")
    print("    [0]  Skip")
    print()

    choice = _ask("Choice", "0")
    updates = {}

    if choice == "1":
        r = _setup_ollama(cfg)
        if r:
            updates.update(r)
    elif choice == "2":
        r = _setup_claude(cfg)
        if r:
            updates.update(r)
    elif choice == "3":
        r = _setup_openai(cfg)
        if r:
            updates.update(r)
    elif choice == "4":
        r = _setup_compat(cfg)
        if r:
            updates.update(r)
    elif choice == "5":
        print(f"\n  {B}Fallback chain{Z}")
        print("  Enter providers in preferred order, comma-separated.")
        print("  Examples: claude, ollama/mistral, openai\n")
        chain = _ask("Chain (e.g. ollama/mistral,claude)", "")
        if chain:
            parts = [p.strip() for p in chain.split(",") if p.strip()]
            if parts:
                updates["preferred"] = parts[0]
                updates["fallback"]  = parts[1:]
    elif choice == "0":
        print(f"\n  {DIM}Skipping LLM setup. PRISM will run in stdlib-only mode.{Z}\n")
        return

    if not updates:
        print(f"\n  {Y}No changes made.{Z}\n")
        return

    # Merge with existing config
    merged = dict(llm)
    merged.update(updates)

    _write_llm_section(merged)

    print(f"\n  {G}{B}Config saved to {_CONFIG_PATH}{Z}")
    _info(f"Active provider: {merged.get('preferred', 'auto-detect')}")
    if merged.get("ollama_model"):
        _info(f"Ollama model:    {merged['ollama_model']}")
    print("\n  Restart PRISM for changes to take effect.")
    print("  Or run  python3 kde_cli.py server --port 8742\n")
    _hr()


if __name__ == "__main__":
    run_wizard()
