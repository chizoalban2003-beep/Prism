"""
PRISM LLM Settings — web UI page and JSON API helpers.

Served at /settings/llm by prism_asgi / prism_routes_infra.py.
API:
  GET  /settings/llm          → HTML page
  POST /settings/llm          → save config  {provider, key, host, model, preferred}
  POST /settings/llm/test     → test a connection {provider, key?, host?}
"""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

_CONFIG_PATH = Path(__file__).parent / "prism_config.toml"


# ── Config read/write ─────────────────────────────────────────────────────────

def read_llm_config() -> dict:
    try:
        import tomllib
        if _CONFIG_PATH.exists():
            data = tomllib.loads(_CONFIG_PATH.read_text())
            return data.get("llm", {})
    except Exception:
        pass
    return {}


def write_llm_config(updates: dict) -> None:
    """Merge updates into [llm] section of prism_config.toml."""
    import re
    text = _CONFIG_PATH.read_text() if _CONFIG_PATH.exists() else ""
    existing = read_llm_config()
    merged = {**existing, **updates}

    lines = ["[llm]"]
    key_order = ["preferred", "fallback", "claude_api_key",
                 "openai_api_key", "openai_host", "ollama_host", "ollama_model"]
    seen: set[str] = set()
    for k in key_order:
        if k in merged:
            v = merged[k]
            lines.append(f'{k} = {json.dumps(v)}')
            seen.add(k)
    for k, v in merged.items():
        if k not in seen:
            lines.append(f'{k} = {json.dumps(v)}')
    block = "\n".join(lines) + "\n"

    pattern = re.compile(r'^\[llm\].*?(?=^\[|\Z)', re.MULTILINE | re.DOTALL)
    m = pattern.search(text)
    if m:
        new_text = text[:m.start()] + block + text[m.end():]
    else:
        new_text = text.rstrip("\n") + "\n\n" + block

    _CONFIG_PATH.write_text(new_text)


# ── Connection testers ────────────────────────────────────────────────────────

def test_provider(provider: str, key: str = "", host: str = "",
                  model: str = "") -> dict:
    try:
        if provider == "claude":
            return _test_claude(key)
        elif provider == "ollama":
            return _test_ollama(host or "http://localhost:11434")
        elif provider in ("openai", "openai_compat"):
            h = host or "https://api.openai.com"
            return _test_openai(h, key)
        return {"ok": False, "message": f"Unknown provider: {provider}"}
    except Exception as e:
        return {"ok": False, "message": str(e)[:120]}


def _test_claude(api_key: str) -> dict:
    if not api_key:
        return {"ok": False, "message": "No API key provided"}
    try:
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=json.dumps({
                "model": "claude-haiku-4-5-20251001",
                "max_tokens": 10,
                "messages": [{"role": "user", "content": "ping"}]
            }).encode(),
            headers={
                "Content-Type":       "application/json",
                "x-api-key":          api_key,
                "anthropic-version":  "2023-06-01",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=8) as r:
            data = json.loads(r.read())
            return {"ok": True, "message": f"Connected — model: {data.get('model', '?')}"}
    except urllib.error.HTTPError as e:
        body = e.read().decode()[:160]
        return {"ok": False, "message": f"HTTP {e.code}: {body}"}
    except Exception as e:
        return {"ok": False, "message": str(e)[:120]}


def _test_ollama(host: str) -> dict:
    try:
        url = host.rstrip("/") + "/api/tags"
        with urllib.request.urlopen(url, timeout=4) as r:
            data = json.loads(r.read())
            models = [m["name"] for m in data.get("models", [])]
            return {"ok": True, "message": f"Running — {len(models)} model(s)",
                    "models": models}
    except Exception as e:
        return {"ok": False, "message": str(e)[:120], "models": []}


def _test_openai(host: str, api_key: str) -> dict:
    try:
        url = host.rstrip("/") + "/v1/models"
        req = urllib.request.Request(
            url, headers={"Authorization": f"Bearer {api_key}"}
        )
        with urllib.request.urlopen(req, timeout=8) as r:
            data = json.loads(r.read())
            ids = [m["id"] for m in data.get("data", [])[:6]]
            return {"ok": True, "message": f"Connected — {len(ids)} models",
                    "models": ids}
    except urllib.error.HTTPError as e:
        body = e.read().decode()[:160]
        return {"ok": False, "message": f"HTTP {e.code}: {body}"}
    except Exception as e:
        return {"ok": False, "message": str(e)[:120]}


# ── HTML page ─────────────────────────────────────────────────────────────────

def get_llm_settings_html() -> str:
    cfg = read_llm_config()
    claude_key    = cfg.get("claude_api_key", "")
    oai_key       = cfg.get("openai_api_key", "")
    oai_host      = cfg.get("openai_host",    "https://api.openai.com")
    ollama_host   = cfg.get("ollama_host",    "http://localhost:11434")
    ollama_model  = cfg.get("ollama_model",   "mistral")
    preferred     = cfg.get("preferred",      "")

    def _mask(k): return ("•" * 8 + k[-4:]) if len(k) > 8 else ("•" * len(k) if k else "")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PRISM — LLM Settings</title>
<style>
  :root {{
    --bg: #0f0f13; --surface: #18181f; --border: #2a2a35;
    --accent: #7c6af7; --accent2: #56d4a8; --text: #e2e2f0;
    --sub: #8888a8; --ok: #56d4a8; --err: #f76a6a; --warn: #f7b96a;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: var(--bg); color: var(--text); font-family: 'Segoe UI', system-ui, sans-serif;
         font-size: 14px; min-height: 100vh; }}
  a {{ color: var(--accent); text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}

  .topbar {{ background: var(--surface); border-bottom: 1px solid var(--border);
             padding: 0 24px; height: 52px; display: flex; align-items: center; gap: 16px; }}
  .topbar .logo {{ font-size: 17px; font-weight: 700; color: var(--accent); letter-spacing: 2px; }}
  .topbar nav a {{ color: var(--sub); font-size: 13px; }}
  .topbar nav a:hover {{ color: var(--text); }}
  .topbar .spacer {{ flex: 1; }}

  .page {{ max-width: 860px; margin: 0 auto; padding: 36px 24px; }}
  h1 {{ font-size: 22px; font-weight: 700; margin-bottom: 6px; }}
  .subtitle {{ color: var(--sub); margin-bottom: 32px; font-size: 13px; }}

  .status-bar {{ background: var(--surface); border: 1px solid var(--border);
                 border-radius: 10px; padding: 14px 20px; margin-bottom: 28px;
                 display: flex; align-items: center; gap: 12px; }}
  .status-bar .label {{ color: var(--sub); font-size: 12px; }}
  .status-bar .value {{ font-weight: 600; font-size: 15px; }}
  .dot {{ width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }}
  .dot.ok {{ background: var(--ok); box-shadow: 0 0 6px var(--ok); }}
  .dot.err {{ background: var(--err); }}
  .dot.idle {{ background: var(--sub); }}

  .cards {{ display: grid; gap: 18px; }}
  .card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 12px;
           padding: 22px 24px; transition: border-color .15s; }}
  .card.active {{ border-color: var(--accent); }}
  .card-head {{ display: flex; align-items: flex-start; gap: 14px; margin-bottom: 16px; }}
  .card-icon {{ font-size: 26px; line-height: 1; }}
  .card-info h2 {{ font-size: 16px; font-weight: 700; }}
  .card-info p {{ color: var(--sub); font-size: 12px; margin-top: 3px; line-height: 1.5; }}
  .badge {{ display: inline-block; font-size: 10px; font-weight: 700; padding: 2px 7px;
            border-radius: 20px; text-transform: uppercase; letter-spacing: .5px; margin-left: 8px;
            vertical-align: middle; }}
  .badge.active {{ background: #7c6af722; color: var(--accent); border: 1px solid var(--accent); }}
  .badge.free {{ background: #56d4a822; color: var(--ok); border: 1px solid var(--ok); }}

  .field {{ margin-bottom: 12px; }}
  .field label {{ display: block; color: var(--sub); font-size: 11px;
                  text-transform: uppercase; letter-spacing: .5px; margin-bottom: 5px; }}
  .field input, .field select {{
    width: 100%; background: var(--bg); border: 1px solid var(--border);
    border-radius: 7px; padding: 9px 12px; color: var(--text); font-size: 13px;
    outline: none; transition: border-color .15s;
  }}
  .field input:focus, .field select:focus {{ border-color: var(--accent); }}
  .field .row {{ display: flex; gap: 8px; }}
  .field .row input {{ flex: 1; }}

  .btn {{ display: inline-flex; align-items: center; gap: 6px; border: none; border-radius: 7px;
          padding: 9px 18px; font-size: 13px; font-weight: 600; cursor: pointer;
          transition: opacity .15s; }}
  .btn:hover {{ opacity: .85; }}
  .btn:disabled {{ opacity: .4; cursor: default; }}
  .btn.primary {{ background: var(--accent); color: #fff; }}
  .btn.secondary {{ background: transparent; border: 1px solid var(--border); color: var(--text); }}
  .btn.sm {{ padding: 6px 12px; font-size: 12px; }}

  .actions {{ display: flex; gap: 10px; align-items: center; margin-top: 16px; flex-wrap: wrap; }}
  .status-msg {{ font-size: 12px; padding: 6px 12px; border-radius: 6px; flex: 1;
                 min-width: 0; word-break: break-word; }}
  .status-msg.ok  {{ background: #56d4a811; color: var(--ok); border: 1px solid #56d4a833; }}
  .status-msg.err {{ background: #f76a6a11; color: var(--err); border: 1px solid #f76a6a33; }}
  .status-msg.loading {{ background: #7c6af711; color: var(--accent); border: 1px solid #7c6af733; }}

  .model-list {{ display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }}
  .model-pill {{ background: var(--bg); border: 1px solid var(--border); border-radius: 20px;
                 padding: 3px 10px; font-size: 11px; color: var(--sub); cursor: pointer; }}
  .model-pill:hover {{ border-color: var(--accent); color: var(--text); }}
  .model-pill.selected {{ border-color: var(--accent); color: var(--accent); background: #7c6af711; }}

  .divider {{ border: none; border-top: 1px solid var(--border); margin: 10px 0; }}
  .section-label {{ font-size: 11px; color: var(--sub); text-transform: uppercase;
                    letter-spacing: .8px; margin-bottom: 14px; margin-top: 24px; }}

  @media (max-width: 600px) {{ .page {{ padding: 20px 14px; }} }}
</style>
</head>
<body>

<div class="topbar">
  <span class="logo">PRISM</span>
  <nav>
    <a href="/">← Chat</a>
  </nav>
  <span class="spacer"></span>
  <span style="color:var(--sub);font-size:12px">LLM Settings</span>
</div>

<div class="page">
  <h1>LLM Configuration</h1>
  <p class="subtitle">Connect a language model to power PRISM's reasoning chains, organ routing, and synthesis.</p>

  <div class="status-bar" id="statusBar">
    <div class="dot idle" id="statusDot"></div>
    <div>
      <div class="label">Active LLM</div>
      <div class="value" id="statusVal">Detecting…</div>
    </div>
    <div style="margin-left:auto">
      <button class="btn secondary sm" onclick="refreshStatus()">Refresh</button>
    </div>
  </div>

  <div class="cards">

    <!-- Ollama -->
    <div class="card" id="card-ollama">
      <div class="card-head">
        <div class="card-icon">🦙</div>
        <div class="card-info">
          <h2>Ollama <span class="badge free">Free</span></h2>
          <p>Run LLMs locally — fully private, no API key needed.<br>
             Install from <a href="https://ollama.ai" target="_blank">ollama.ai</a> then
             run <code>ollama pull mistral</code>.</p>
        </div>
      </div>
      <div class="field">
        <label>Ollama host</label>
        <input id="ollama_host" value="{ollama_host}" placeholder="http://localhost:11434">
      </div>
      <div class="field" id="ollamaModelField">
        <label>Model</label>
        <div class="row">
          <input id="ollama_model" value="{ollama_model}" placeholder="mistral">
        </div>
        <div class="model-list" id="modelList"></div>
      </div>
      <div class="actions">
        <button class="btn secondary sm" onclick="testProvider('ollama')">Test connection</button>
        <button class="btn primary sm" onclick="saveProvider('ollama')">Save &amp; use Ollama</button>
        <span class="status-msg" id="ollama-msg" style="display:none"></span>
      </div>
    </div>

    <!-- Claude -->
    <div class="card" id="card-claude">
      <div class="card-head">
        <div class="card-icon">🤖</div>
        <div class="card-info">
          <h2>Claude API <span class="badge active">Anthropic</span></h2>
          <p>Best reasoning quality. Needs an API key from
             <a href="https://console.anthropic.com" target="_blank">console.anthropic.com</a>.</p>
        </div>
      </div>
      <div class="field">
        <label>API key</label>
        <input id="claude_api_key" type="password" value="{_mask(claude_key)}"
               placeholder="sk-ant-..." autocomplete="off"
               onfocus="if(this.value.startsWith('•'))this.value=''">
      </div>
      <div class="actions">
        <button class="btn secondary sm" onclick="testProvider('claude')">Test connection</button>
        <button class="btn primary sm" onclick="saveProvider('claude')">Save &amp; use Claude</button>
        <span class="status-msg" id="claude-msg" style="display:none"></span>
      </div>
    </div>

    <!-- OpenAI -->
    <div class="card" id="card-openai">
      <div class="card-head">
        <div class="card-icon">🟢</div>
        <div class="card-info">
          <h2>OpenAI</h2>
          <p>GPT-4o, GPT-4o-mini, and other OpenAI models. API key from
             <a href="https://platform.openai.com/api-keys" target="_blank">platform.openai.com</a>.</p>
        </div>
      </div>
      <div class="field">
        <label>API key</label>
        <input id="openai_api_key" type="password" value="{_mask(oai_key)}"
               placeholder="sk-..." autocomplete="off"
               onfocus="if(this.value.startsWith('•'))this.value=''">
      </div>
      <div class="actions">
        <button class="btn secondary sm" onclick="testProvider('openai')">Test connection</button>
        <button class="btn primary sm" onclick="saveProvider('openai')">Save &amp; use OpenAI</button>
        <span class="status-msg" id="openai-msg" style="display:none"></span>
      </div>
    </div>

    <!-- OpenAI-compatible -->
    <div class="card" id="card-openai_compat">
      <div class="card-head">
        <div class="card-icon">⚡</div>
        <div class="card-info">
          <h2>OpenAI-compatible endpoint</h2>
          <p>Groq · Together AI · LM Studio · llama.cpp · Gemini · Mistral AI ·
             any /v1/chat/completions-compatible server.</p>
        </div>
      </div>
      <div class="field">
        <label>Endpoint URL</label>
        <input id="compat_host" value="{oai_host if oai_host != 'https://api.openai.com' else 'http://localhost:1234'}"
               placeholder="http://localhost:1234">
      </div>
      <div class="field">
        <label>API key (or leave blank for keyless)</label>
        <input id="compat_key" type="password"
               value="{_mask(oai_key) if oai_host != 'https://api.openai.com' else ''}"
               placeholder="your-api-key  or  local" autocomplete="off"
               onfocus="if(this.value.startsWith('•'))this.value=''">
      </div>
      <div class="actions">
        <button class="btn secondary sm" onclick="testProvider('openai_compat')">Test connection</button>
        <button class="btn primary sm" onclick="saveProvider('openai_compat')">Save &amp; use this endpoint</button>
        <span class="status-msg" id="openai_compat-msg" style="display:none"></span>
      </div>
    </div>

  </div><!-- /cards -->

  <hr class="divider" style="margin-top:32px">
  <p style="color:var(--sub);font-size:12px;margin-top:14px">
    Config is saved to <code>prism_config.toml</code> in the PRISM directory.
    Restart the server (or reload via the API) for changes to take effect.<br>
    You can also set env vars: <code>ANTHROPIC_API_KEY</code>, <code>OPENAI_API_KEY</code>.
    Run <code>python3 prism_setup_llm.py</code> for the CLI wizard.
  </p>
</div>

<script>
const preferred = {json.dumps(preferred)};

async function api(path, body) {{
  const r = await fetch(path, {{
    method: 'POST', headers: {{'Content-Type': 'application/json'}},
    body: JSON.stringify(body)
  }});
  return r.json();
}}

function showMsg(id, ok, text) {{
  const el = document.getElementById(id + '-msg');
  el.style.display = '';
  el.className = 'status-msg ' + (ok ? 'ok' : 'err');
  el.textContent = text;
}}

function setMsg(id, text) {{
  const el = document.getElementById(id + '-msg');
  el.style.display = '';
  el.className = 'status-msg loading';
  el.textContent = text;
}}

async function testProvider(p) {{
  setMsg(p, 'Testing…');
  const body = {{ provider: p }};
  if (p === 'ollama')        body.host = document.getElementById('ollama_host').value;
  if (p === 'claude')        body.key  = document.getElementById('claude_api_key').value;
  if (p === 'openai')        body.key  = document.getElementById('openai_api_key').value;
  if (p === 'openai_compat') {{
    body.host = document.getElementById('compat_host').value;
    body.key  = document.getElementById('compat_key').value || 'local';
  }}
  const res = await api('/settings/llm/test', body);
  showMsg(p, res.ok, res.message);
  if (res.ok && p === 'ollama' && res.models) renderModels(res.models);
}}

async function saveProvider(p) {{
  setMsg(p, 'Saving…');
  const body = {{ provider: p }};
  if (p === 'ollama') {{
    body.host  = document.getElementById('ollama_host').value;
    body.model = document.getElementById('ollama_model').value;
  }}
  if (p === 'claude')        body.key  = document.getElementById('claude_api_key').value;
  if (p === 'openai')        body.key  = document.getElementById('openai_api_key').value;
  if (p === 'openai_compat') {{
    body.host = document.getElementById('compat_host').value;
    body.key  = document.getElementById('compat_key').value || 'local';
  }}
  const res = await api('/settings/llm', body);
  showMsg(p, res.ok, res.message);
  if (res.ok) {{
    document.querySelectorAll('.card').forEach(c => c.classList.remove('active'));
    document.getElementById('card-' + p)?.classList.add('active');
    refreshStatus();
  }}
}}

function renderModels(models) {{
  const list = document.getElementById('modelList');
  list.innerHTML = '';
  const current = document.getElementById('ollama_model').value;
  models.forEach(m => {{
    const pill = document.createElement('span');
    pill.className = 'model-pill' + (m === current ? ' selected' : '');
    pill.textContent = m;
    pill.onclick = () => {{
      document.getElementById('ollama_model').value = m;
      list.querySelectorAll('.model-pill').forEach(p => p.classList.remove('selected'));
      pill.classList.add('selected');
    }};
    list.appendChild(pill);
  }});
}}

async function refreshStatus() {{
  const dot = document.getElementById('statusDot');
  const val = document.getElementById('statusVal');
  dot.className = 'dot idle';
  val.textContent = 'Detecting…';
  try {{
    const r = await fetch('/llm/status');
    const d = await r.json();
    if (d.stdlib_only || d.best === 'none') {{
      dot.className = 'dot err';
      val.textContent = 'No LLM — stdlib fallback only';
    }} else {{
      dot.className = 'dot ok';
      val.textContent = d.best || 'Connected';
      // Highlight active card
      document.querySelectorAll('.card').forEach(c => c.classList.remove('active'));
      const best = d.best || '';
      if (best.includes('claude')) document.getElementById('card-claude')?.classList.add('active');
      else if (best.includes('ollama')) document.getElementById('card-ollama')?.classList.add('active');
      else if (best.includes('openai')) document.getElementById('card-openai')?.classList.add('active');
    }}
  }} catch(e) {{
    dot.className = 'dot err';
    val.textContent = 'Server unreachable';
  }}
}}

// On load
refreshStatus();
if (preferred) {{
  if (preferred.startsWith('ollama'))        document.getElementById('card-ollama')?.classList.add('active');
  else if (preferred === 'claude')           document.getElementById('card-claude')?.classList.add('active');
  else if (preferred === 'openai')           document.getElementById('card-openai')?.classList.add('active');
  else if (preferred === 'openai_compat')    document.getElementById('card-openai_compat')?.classList.add('active');
}}

// Auto-load Ollama models if running
(async () => {{
  try {{
    const r = await fetch('/settings/llm/test', {{
      method: 'POST', headers: {{'Content-Type':'application/json'}},
      body: JSON.stringify({{provider:'ollama', host: document.getElementById('ollama_host').value}})
    }});
    const d = await r.json();
    if (d.ok && d.models) renderModels(d.models);
  }} catch(e) {{}}
}})();
</script>
</body>
</html>"""
