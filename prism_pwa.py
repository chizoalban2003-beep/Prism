"""
prism_pwa.py
============
PRISM Mobile Companion — Progressive Web App

Delivers a mobile-first installable interface on top of the existing
REST API.  No app store, no build step, no native toolchain.  Users open
http://prism.local:8742/mobile in their phone browser and tap
"Add to Home Screen".

Served assets
-------------
  GET /mobile          → full mobile SPA (HTML + inline CSS/JS)
  GET /manifest.json   → Web App Manifest (icons, colours, display mode)
  GET /sw.js           → Service Worker (cache-first shell, network-first API)
  GET /icon.svg        → vector app icon (used by manifest + favicon)

Mobile SPA tabs
---------------
  Chat     — full-screen thread, works identically to the desktop chat
  Goals    — horizon goals list with status, add new goal, abandon
  Status   — PRISM online/offline, LLM model, background tasks
  Voice    — Web Speech API STT → sent as chat message

PWA features
------------
  • Installable  — manifest + service worker satisfy browser install criteria
  • Offline      — app shell cached; shows offline banner when API unreachable
  • Push         — deep-links to the ntfy.sh topic configured on the server
  • Safe area    — env(safe-area-inset-*) for notched/dynamic-island phones
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# SVG icon (served at /icon.svg, referenced by manifest and <link> tags)
# ---------------------------------------------------------------------------

ICON_SVG: str = """\
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 192 192">
  <defs>
    <linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#5DCAA5"/>
      <stop offset="100%" stop-color="#3B8BD4"/>
    </linearGradient>
  </defs>
  <rect width="192" height="192" rx="40" fill="#0f1117"/>
  <polygon points="96,32 160,68 160,124 96,160 32,124 32,68"
           fill="none" stroke="url(#g)" stroke-width="8"/>
  <line x1="96" y1="32" x2="96" y2="160" stroke="url(#g)" stroke-width="3" stroke-dasharray="6,6"/>
  <line x1="32" y1="68" x2="160" y2="124" stroke="url(#g)" stroke-width="3" stroke-dasharray="6,6"/>
  <line x1="160" y1="68" x2="32" y2="124" stroke="url(#g)" stroke-width="3" stroke-dasharray="6,6"/>
  <circle cx="96" cy="96" r="14" fill="url(#g)"/>
</svg>"""


# ---------------------------------------------------------------------------
# Web App Manifest
# ---------------------------------------------------------------------------

MANIFEST_JSON: str = """\
{
  "name": "PRISM — Decision Intelligence",
  "short_name": "PRISM",
  "description": "Local-first personal AI assistant",
  "start_url": "/mobile",
  "scope": "/",
  "display": "standalone",
  "orientation": "portrait-primary",
  "background_color": "#0f1117",
  "theme_color": "#5DCAA5",
  "icons": [
    { "src": "/icon.svg", "sizes": "any", "type": "image/svg+xml", "purpose": "any maskable" }
  ],
  "categories": ["productivity", "utilities"],
  "screenshots": []
}"""


# ---------------------------------------------------------------------------
# Service Worker
# ---------------------------------------------------------------------------

SERVICE_WORKER_JS: str = """\
/* PRISM Service Worker — cache-first shell, network-first API */
const CACHE = 'prism-v1';
const SHELL = ['/mobile', '/manifest.json', '/icon.svg'];

self.addEventListener('install', e => {
  e.waitUntil(
    caches.open(CACHE).then(c => c.addAll(SHELL)).then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', e => {
  e.waitUntil(
    caches.keys().then(keys =>
      Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k)))
    ).then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', e => {
  const url = new URL(e.request.url);
  /* API calls — network first, no cache */
  if (['/chat','/horizon','/status','/tasks','/organs'].some(p => url.pathname.startsWith(p))) {
    e.respondWith(
      fetch(e.request).catch(() =>
        new Response(JSON.stringify({error:'offline'}),
          {status:503, headers:{'Content-Type':'application/json'}})
      )
    );
    return;
  }
  /* App shell — cache first */
  e.respondWith(
    caches.match(e.request).then(cached => cached || fetch(e.request).then(resp => {
      if (resp.ok && e.request.method === 'GET') {
        const clone = resp.clone();
        caches.open(CACHE).then(c => c.put(e.request, clone));
      }
      return resp;
    }))
  );
});"""


# ---------------------------------------------------------------------------
# Mobile SPA
# ---------------------------------------------------------------------------

MOBILE_HTML: str = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<meta name="theme-color" content="#5DCAA5">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="apple-mobile-web-app-title" content="PRISM">
<link rel="manifest" href="/manifest.json">
<link rel="icon" href="/icon.svg" type="image/svg+xml">
<link rel="apple-touch-icon" href="/icon.svg">
<title>PRISM</title>
<style>
:root{
  --bg:#0f1117;--surface:#1a1f2e;--surface2:#131824;--border:rgba(255,255,255,.08);
  --text:#e8eaf0;--muted:#93a0b5;--accent:#5DCAA5;--accent2:#3B8BD4;
  --red:#E24B4A;--yellow:#EF9F27;--radius:14px;
  --nav-h:64px;--safe-b:env(safe-area-inset-bottom,0px);
  --safe-t:env(safe-area-inset-top,0px);
}
@media(prefers-color-scheme:light){
  :root{--bg:#f5f7fa;--surface:#fff;--surface2:#eef2f7;--border:rgba(0,0,0,.08);--text:#1a1f2e;--muted:#5f6b7d;}
}
*{box-sizing:border-box;-webkit-tap-highlight-color:transparent;}
html,body{margin:0;height:100%;overflow:hidden;font-family:system-ui,-apple-system,sans-serif;
  background:var(--bg);color:var(--text);}
#app{height:100%;display:flex;flex-direction:column;}

/* ── Header ── */
#topbar{
  flex:0 0 auto;padding:12px 16px;padding-top:calc(12px + var(--safe-t));
  background:var(--surface);border-bottom:1px solid var(--border);
  display:flex;align-items:center;justify-content:space-between;
}
#topbar .brand{display:flex;align-items:center;gap:10px;}
#topbar .brand svg{width:28px;height:28px;}
#topbar h1{margin:0;font-size:1.1rem;font-weight:700;letter-spacing:-.02em;}
#status-pill{display:flex;align-items:center;gap:6px;font-size:.8rem;color:var(--muted);
  background:var(--surface2);border:1px solid var(--border);padding:6px 10px;border-radius:999px;}
.dot{width:8px;height:8px;border-radius:50%;background:var(--yellow);}
.dot.online{background:var(--accent);}

/* ── Tab content ── */
#panels{flex:1 1 0;overflow:hidden;position:relative;}
.panel{position:absolute;inset:0;overflow-y:auto;padding-bottom:calc(var(--nav-h) + var(--safe-b) + 8px);display:none;}
.panel.active{display:block;}

/* ── Bottom nav ── */
#nav{
  flex:0 0 auto;height:calc(var(--nav-h) + var(--safe-b));
  padding-bottom:var(--safe-b);
  background:var(--surface);border-top:1px solid var(--border);
  display:flex;
}
.nav-btn{flex:1;display:flex;flex-direction:column;align-items:center;justify-content:center;
  gap:3px;border:none;background:none;color:var(--muted);font-size:.7rem;font-weight:600;
  cursor:pointer;padding:8px 4px;transition:color .15s;}
.nav-btn svg{width:22px;height:22px;stroke:currentColor;fill:none;stroke-width:2;stroke-linecap:round;stroke-linejoin:round;}
.nav-btn.active{color:var(--accent);}

/* ── Cards ── */
.card{margin:12px;background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:14px;}
.card h2{margin:0 0 10px;font-size:1rem;}
.card h3{margin:0 0 8px;font-size:.9rem;color:var(--muted);}

/* ── Chat ── */
#chat-panel{display:flex;flex-direction:column;}
#chat-panel.active{display:flex;}
#messages{flex:1 1 0;overflow-y:auto;padding:12px;}
.msg{max-width:85%;margin-bottom:10px;padding:10px 13px;border-radius:18px;
  font-size:.95rem;line-height:1.45;word-wrap:break-word;}
.msg.user{margin-left:auto;background:var(--accent);color:#082210;border-bottom-right-radius:4px;}
.msg.ai{margin-right:auto;background:var(--surface);border:1px solid var(--border);border-bottom-left-radius:4px;}
.msg.system{margin:0 auto 6px;font-size:.8rem;color:var(--muted);text-align:center;background:none;}
.msg .title{font-size:.75rem;font-weight:700;color:var(--accent);margin-bottom:4px;text-transform:uppercase;letter-spacing:.05em;}
#input-bar{
  flex:0 0 auto;display:flex;gap:8px;padding:10px 12px;
  padding-bottom:calc(10px + var(--safe-b));
  background:var(--surface);border-top:1px solid var(--border);
}
#chat-input{flex:1;background:var(--surface2);border:1px solid var(--border);border-radius:24px;
  padding:10px 14px;color:var(--text);font:inherit;font-size:.95rem;resize:none;
  max-height:120px;overflow-y:auto;outline:none;}
#chat-input:focus{border-color:var(--accent);}
.icon-btn{width:42px;height:42px;border:none;border-radius:50%;cursor:pointer;display:flex;align-items:center;justify-content:center;flex:0 0 auto;}
#send-btn{background:var(--accent);color:#082210;}
#send-btn svg{width:20px;height:20px;stroke:currentColor;fill:none;stroke-width:2.5;stroke-linecap:round;stroke-linejoin:round;}
#voice-btn{background:var(--surface2);border:1px solid var(--border);color:var(--muted);}
#voice-btn svg{width:20px;height:20px;stroke:currentColor;fill:none;stroke-width:2;stroke-linecap:round;stroke-linejoin:round;}
#voice-btn.recording{background:var(--red);color:#fff;border-color:transparent;}
.typing{display:flex;gap:5px;align-items:center;padding:10px 14px;}
.typing span{width:7px;height:7px;border-radius:50%;background:var(--muted);animation:bounce .9s infinite;}
.typing span:nth-child(2){animation-delay:.2s;}
.typing span:nth-child(3){animation-delay:.4s;}
@keyframes bounce{0%,60%,100%{transform:translateY(0);}30%{transform:translateY(-6px);}}

/* ── Goals ── */
.goal-card{margin:10px 12px;background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:14px;}
.goal-header{display:flex;justify-content:space-between;align-items:flex-start;gap:8px;}
.goal-intent{font-weight:600;font-size:.95rem;flex:1;}
.status-badge{font-size:.72rem;font-weight:700;padding:3px 8px;border-radius:999px;white-space:nowrap;}
.status-watching{background:rgba(239,159,39,.15);color:var(--yellow);}
.status-triggered{background:rgba(93,202,165,.15);color:var(--accent);}
.status-paused{background:rgba(59,139,212,.15);color:var(--accent2);}
.status-completed{background:rgba(93,202,165,.12);color:var(--accent);}
.status-abandoned{background:rgba(226,75,74,.12);color:var(--red);}
.goal-meta{margin-top:6px;font-size:.82rem;color:var(--muted);}
.goal-actions{margin-top:10px;display:flex;gap:8px;}
.btn-sm{padding:6px 12px;border:1px solid var(--border);border-radius:8px;background:var(--surface2);
  color:var(--text);font-size:.8rem;font-weight:600;cursor:pointer;}
.btn-sm.danger{color:var(--red);}
.btn-primary{background:var(--accent);color:#082210;border:none;border-radius:10px;padding:10px 16px;
  font-weight:700;font-size:.9rem;cursor:pointer;width:100%;}
.form-field{margin-bottom:12px;}
.form-field label{display:block;font-size:.85rem;color:var(--muted);margin-bottom:5px;}
.form-field input,.form-field textarea{width:100%;background:var(--surface2);border:1px solid var(--border);
  border-radius:10px;padding:10px 12px;color:var(--text);font:inherit;font-size:.9rem;outline:none;}
.form-field input:focus,.form-field textarea:focus{border-color:var(--accent);}
.form-field textarea{resize:none;height:72px;}

/* ── Status ── */
.stat-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin:12px;}
.stat-tile{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:14px;text-align:center;}
.stat-tile .val{font-size:1.6rem;font-weight:700;margin-bottom:2px;}
.stat-tile .lbl{font-size:.75rem;color:var(--muted);}
.list-item{display:flex;justify-content:space-between;align-items:center;padding:10px 0;
  border-bottom:1px solid var(--border);}
.list-item:last-child{border-bottom:none;}
.list-item .k{font-size:.85rem;color:var(--muted);}
.list-item .v{font-size:.85rem;font-weight:600;}

/* ── Voice ── */
#voice-panel .center{display:flex;flex-direction:column;align-items:center;padding:40px 20px;gap:16px;}
#mic-btn{width:96px;height:96px;border-radius:50%;border:3px solid var(--accent);background:var(--surface);
  display:flex;align-items:center;justify-content:center;cursor:pointer;transition:all .2s;}
#mic-btn svg{width:40px;height:40px;stroke:var(--accent);fill:none;stroke-width:1.8;stroke-linecap:round;stroke-linejoin:round;}
#mic-btn.recording{background:var(--red);border-color:var(--red);animation:pulse 1s infinite;}
#mic-btn.recording svg{stroke:#fff;}
@keyframes pulse{0%,100%{box-shadow:0 0 0 0 rgba(226,75,74,.4);}50%{box-shadow:0 0 0 16px rgba(226,75,74,0);}}
#transcript-box{width:100%;max-width:340px;background:var(--surface);border:1px solid var(--border);
  border-radius:var(--radius);padding:14px;min-height:80px;font-size:.95rem;color:var(--muted);text-align:center;}
#voice-status{font-size:.85rem;color:var(--muted);}

/* ── Offline banner ── */
#offline-banner{display:none;position:fixed;top:0;left:0;right:0;z-index:999;
  background:var(--red);color:#fff;text-align:center;padding:10px;font-weight:600;font-size:.9rem;}

/* ── Install prompt ── */
#install-banner{display:none;margin:8px 12px;background:var(--surface);border:1px solid var(--accent);
  border-radius:var(--radius);padding:14px;display:flex;align-items:center;gap:12px;}
#install-banner p{flex:1;margin:0;font-size:.88rem;}
#install-banner button{background:var(--accent);color:#082210;border:none;border-radius:8px;
  padding:8px 14px;font-weight:700;cursor:pointer;white-space:nowrap;}
#install-banner .dismiss{background:none;color:var(--muted);padding:4px;font-size:1.2rem;}
</style>
</head>
<body>
<div id="offline-banner">You're offline — PRISM is unavailable</div>

<div id="app">
  <!-- Header -->
  <div id="topbar">
    <div class="brand">
      <svg viewBox="0 0 192 192" xmlns="http://www.w3.org/2000/svg">
        <defs><linearGradient id="hg" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stop-color="#5DCAA5"/><stop offset="100%" stop-color="#3B8BD4"/>
        </linearGradient></defs>
        <rect width="192" height="192" rx="40" fill="#0f1117"/>
        <polygon points="96,32 160,68 160,124 96,160 32,124 32,68"
                 fill="none" stroke="url(#hg)" stroke-width="10"/>
        <circle cx="96" cy="96" r="14" fill="url(#hg)"/>
      </svg>
      <h1>PRISM</h1>
    </div>
    <div id="status-pill">
      <span class="dot" id="s-dot"></span>
      <span id="s-text">Connecting…</span>
    </div>
  </div>

  <!-- Panels -->
  <div id="panels">

    <!-- Chat -->
    <div class="panel active" id="chat-panel">
      <div id="messages"></div>
    </div>

    <!-- Goals -->
    <div class="panel" id="goals-panel">
      <div class="card">
        <h2>New horizon goal</h2>
        <div class="form-field">
          <label>Goal intent</label>
          <textarea id="g-intent" placeholder="Book a flight when price drops below $300…"></textarea>
        </div>
        <div class="form-field">
          <label>Trigger condition</label>
          <input id="g-trigger" placeholder="price drops below 300">
        </div>
        <div class="form-field">
          <label>Completion condition (optional)</label>
          <input id="g-complete" placeholder="flight is booked">
        </div>
        <button class="btn-primary" id="g-add-btn">Watch for this</button>
      </div>
      <div id="goals-list"></div>
    </div>

    <!-- Status -->
    <div class="panel" id="status-panel">
      <div class="stat-grid" id="stat-tiles"></div>
      <div class="card">
        <h2>System</h2>
        <div id="sys-list"></div>
      </div>
      <div class="card">
        <h2>Active tasks</h2>
        <div id="tasks-list"></div>
      </div>
      <div class="card">
        <h2>Loaded organs</h2>
        <div id="organs-list"></div>
      </div>
      <div class="card" id="push-card">
        <h2>Push notifications</h2>
        <div id="push-info"></div>
      </div>
    </div>

    <!-- Voice -->
    <div class="panel" id="voice-panel">
      <div class="center">
        <div id="voice-status">Tap the microphone to speak</div>
        <div id="mic-btn">
          <svg viewBox="0 0 24 24"><rect x="9" y="2" width="6" height="13" rx="3"/>
            <path d="M5 10a7 7 0 0 0 14 0"/><line x1="12" y1="19" x2="12" y2="22"/></svg>
        </div>
        <div id="transcript-box">Your words will appear here…</div>
        <button class="btn-primary" id="voice-send" style="max-width:280px;display:none">
          Send to PRISM
        </button>
      </div>
    </div>

  </div><!-- /panels -->

  <!-- Chat input (only visible on chat panel) -->
  <div id="input-bar">
    <textarea id="chat-input" rows="1" placeholder="Ask PRISM anything…"></textarea>
    <button class="icon-btn" id="voice-btn" title="Voice input">
      <svg viewBox="0 0 24 24"><rect x="9" y="2" width="6" height="13" rx="3"/>
        <path d="M5 10a7 7 0 0 0 14 0"/><line x1="12" y1="19" x2="12" y2="22"/></svg>
    </button>
    <button class="icon-btn" id="send-btn" title="Send">
      <svg viewBox="0 0 24 24"><line x1="22" y1="2" x2="11" y2="13"/>
        <polygon points="22 2 15 22 11 13 2 9 22 2"/></svg>
    </button>
  </div>

  <!-- Bottom nav -->
  <nav id="nav">
    <button class="nav-btn active" data-panel="chat">
      <svg viewBox="0 0 24 24"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
      Chat
    </button>
    <button class="nav-btn" data-panel="goals">
      <svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>
      Goals
    </button>
    <button class="nav-btn" data-panel="status">
      <svg viewBox="0 0 24 24"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>
      Status
    </button>
    <button class="nav-btn" data-panel="voice">
      <svg viewBox="0 0 24 24"><rect x="9" y="2" width="6" height="13" rx="3"/>
        <path d="M5 10a7 7 0 0 0 14 0"/><line x1="12" y1="19" x2="12" y2="22"/></svg>
      Voice
    </button>
  </nav>
</div><!-- /app -->

<script>
const API = window.location.origin;
let deferredInstall = null;
let isRecording = false;
let recognition = null;
let pendingTranscript = '';

// ── Service worker ────────────────────────────────────────────────────────
if ('serviceWorker' in navigator) {
  navigator.serviceWorker.register('/sw.js').catch(() => {});
}

// ── Install prompt ─────────────────────────────────────────────────────────
window.addEventListener('beforeinstallprompt', e => {
  e.preventDefault();
  deferredInstall = e;
  showInstallBanner();
});

function showInstallBanner() {
  const b = document.getElementById('install-banner');
  if (b) b.style.display = 'flex';
}

// ── Online / offline ───────────────────────────────────────────────────────
function setOnline(online) {
  document.getElementById('offline-banner').style.display = online ? 'none' : 'block';
}
window.addEventListener('online',  () => setOnline(true));
window.addEventListener('offline', () => setOnline(false));
setOnline(navigator.onLine);

// ── Panel navigation ──────────────────────────────────────────────────────
const inputBar = document.getElementById('input-bar');

function showPanel(name) {
  document.querySelectorAll('.panel').forEach(p =>
    p.classList.toggle('active', p.id === `${name}-panel`));
  document.querySelectorAll('.nav-btn').forEach(b =>
    b.classList.toggle('active', b.dataset.panel === name));
  inputBar.style.display = name === 'chat' ? 'flex' : 'none';
  if (name === 'goals')  loadGoals();
  if (name === 'status') loadStatus();
}

document.querySelectorAll('.nav-btn').forEach(btn =>
  btn.addEventListener('click', () => showPanel(btn.dataset.panel)));

// ── Chat ──────────────────────────────────────────────────────────────────
const messages = document.getElementById('messages');
const chatInput = document.getElementById('chat-input');

function appendMsg(role, text, title) {
  const div = document.createElement('div');
  div.className = `msg ${role}`;
  if (title) div.innerHTML = `<div class="title">${escHtml(title)}</div>${md(text)}`;
  else div.innerHTML = md(text);
  messages.appendChild(div);
  messages.scrollTop = messages.scrollHeight;
  return div;
}

function showTyping() {
  const d = document.createElement('div');
  d.className = 'msg ai'; d.id = 'typing-indicator';
  d.innerHTML = '<div class="typing"><span></span><span></span><span></span></div>';
  messages.appendChild(d);
  messages.scrollTop = messages.scrollHeight;
}
function hideTyping() {
  const t = document.getElementById('typing-indicator');
  if (t) t.remove();
}

async function sendChat(text) {
  if (!text.trim()) return;
  appendMsg('user', text);
  chatInput.value = '';
  chatInput.style.height = '';
  showTyping();
  try {
    const r = await fetch(API + '/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({message: text}),
    });
    const data = await r.json();
    hideTyping();
    if (data.error) { appendMsg('system', '⚠ ' + data.error); return; }
    const body = data.body || data.message || JSON.stringify(data);
    const title = data.title || '';
    appendMsg('ai', body, title);
  } catch (e) {
    hideTyping();
    appendMsg('system', navigator.onLine ? '⚠ PRISM unavailable' : '⚠ Offline');
  }
}

document.getElementById('send-btn').addEventListener('click', () =>
  sendChat(chatInput.value));

chatInput.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendChat(chatInput.value); }
});

chatInput.addEventListener('input', () => {
  chatInput.style.height = '';
  chatInput.style.height = Math.min(chatInput.scrollHeight, 120) + 'px';
});

// ── Horizon Goals ─────────────────────────────────────────────────────────
async function loadGoals() {
  const container = document.getElementById('goals-list');
  container.innerHTML = '<div class="card"><div style="color:var(--muted);font-size:.9rem">Loading…</div></div>';
  try {
    const data = await fetch(API + '/horizon/goals').then(r => r.json());
    if (!data.goals || !data.goals.length) {
      container.innerHTML = '<div class="card"><p style="color:var(--muted);margin:0;font-size:.9rem">No horizon goals yet. Add one above.</p></div>';
      return;
    }
    const statusClass = {watching:'watching',triggered:'triggered',paused:'paused',
                         completed:'completed',abandoned:'abandoned'};
    const statusIcon  = {watching:'👁',triggered:'⚡',paused:'⏸',completed:'✅',abandoned:'🚫'};
    container.innerHTML = data.goals.map(g => `
      <div class="goal-card" data-id="${g.goal_id}">
        <div class="goal-header">
          <div class="goal-intent">${escHtml(g.intent)}</div>
          <span class="status-badge status-${g.status}">
            ${statusIcon[g.status] || ''} ${g.status}
          </span>
        </div>
        <div class="goal-meta">
          Condition: ${escHtml(g.trigger_condition)}
          · Sessions checked: ${g.session_count}
          · Steps done: ${(g.completed_steps||[]).length}
        </div>
        ${g.status === 'watching' || g.status === 'triggered' ? `
          <div class="goal-actions">
            <button class="btn-sm danger" onclick="abandonGoal('${g.goal_id}')">Abandon</button>
            <button class="btn-sm" onclick="completeGoal('${g.goal_id}')">Mark done</button>
          </div>` : ''}
      </div>`).join('');
  } catch (e) {
    container.innerHTML = '<div class="card"><div style="color:var(--red)">Failed to load goals</div></div>';
  }
}

document.getElementById('g-add-btn').addEventListener('click', async () => {
  const intent   = document.getElementById('g-intent').value.trim();
  const trigger  = document.getElementById('g-trigger').value.trim();
  const complete = document.getElementById('g-complete').value.trim();
  if (!intent || !trigger) {
    alert('Please fill in the goal intent and trigger condition.');
    return;
  }
  const btn = document.getElementById('g-add-btn');
  btn.textContent = 'Adding…'; btn.disabled = true;
  try {
    const r = await fetch(API + '/horizon/goal', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({intent, trigger_condition: trigger, completion_condition: complete}),
    });
    const data = await r.json();
    if (data.goal_id) {
      document.getElementById('g-intent').value = '';
      document.getElementById('g-trigger').value = '';
      document.getElementById('g-complete').value = '';
      await loadGoals();
    }
  } finally {
    btn.textContent = 'Watch for this'; btn.disabled = false;
  }
});

async function abandonGoal(gid) {
  if (!confirm('Abandon this goal?')) return;
  await fetch(API + `/horizon/goal/${gid}/abandon`, {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({reason:'abandoned via mobile app'}),
  });
  loadGoals();
}
async function completeGoal(gid) {
  await fetch(API + `/horizon/goal/${gid}/complete`, {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({notes:'marked done via mobile app'}),
  });
  loadGoals();
}

// ── Status ────────────────────────────────────────────────────────────────
async function loadStatus() {
  try {
    const [status, tasks, horizon, organs] = await Promise.all([
      fetch(API+'/status').then(r=>r.json()),
      fetch(API+'/tasks?n=5').then(r=>r.json()),
      fetch(API+'/horizon/status').then(r=>r.json()),
      fetch(API+'/organs').then(r=>r.json()),
    ]);

    const s = document.getElementById('s-dot');
    s.className = 'dot' + (status.ollama_available ? ' online' : '');
    document.getElementById('s-text').textContent =
      (status.profile || 'PRISM') + (status.ollama_available ? ' · online' : ' · LLM offline');

    document.getElementById('stat-tiles').innerHTML = `
      <div class="stat-tile">
        <div class="val">${horizon.counts?.watching ?? 0}</div>
        <div class="lbl">Watching</div>
      </div>
      <div class="stat-tile">
        <div class="val">${horizon.counts?.completed ?? 0}</div>
        <div class="lbl">Completed</div>
      </div>
      <div class="stat-tile">
        <div class="val">${organs.count ?? 0}</div>
        <div class="lbl">Organs</div>
      </div>
      <div class="stat-tile">
        <div class="val">${(tasks.tasks||[]).filter(t=>t.status==='running').length}</div>
        <div class="lbl">Running</div>
      </div>`;

    const sysItems = [
      ['LLM', status.ollama_available ? (status.ollama_model||'Ollama') : 'Offline'],
      ['Name', status.profile || '—'],
      ['Role', status.role || '—'],
      ['Devices', (status.devices||[]).length + ' connected'],
    ];
    document.getElementById('sys-list').innerHTML = sysItems.map(([k,v]) =>
      `<div class="list-item"><span class="k">${k}</span><span class="v">${escHtml(String(v))}</span></div>`
    ).join('');

    const taskList = tasks.tasks || [];
    document.getElementById('tasks-list').innerHTML = taskList.length
      ? taskList.map(t =>
          `<div class="list-item">
            <span class="k">${escHtml(t.title||t.task_id)}</span>
            <span class="v" style="color:${t.status==='running'?'var(--accent)':t.status==='failed'?'var(--red)':'var(--muted)'}">${t.status}</span>
          </div>`).join('')
      : '<div class="list-item"><span class="k">No recent tasks</span></div>';

    const organMap = organs.organs || {};
    const organKeys = Object.keys(organMap);
    document.getElementById('organs-list').innerHTML = organKeys.length
      ? organKeys.map(k =>
          `<div class="list-item"><span class="k">${escHtml(k)}</span>
           <span class="v" style="color:var(--muted);font-weight:400;max-width:55%;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${escHtml(organMap[k]||'')}</span></div>`
        ).join('')
      : '<div class="list-item"><span class="k" style="color:var(--muted)">None loaded yet</span></div>';

    const pushStatus = await fetch(API+'/push/status').then(r=>r.json()).catch(()=>({}));
    document.getElementById('push-info').innerHTML = pushStatus.configured
      ? `<div class="list-item"><span class="k">Topic</span><span class="v">${escHtml(pushStatus.topic||'')}</span></div>
         <div class="list-item"><span class="k">Server</span><span class="v">${escHtml(pushStatus.server||'ntfy.sh')}</span></div>
         <div style="margin-top:10px;font-size:.82rem;color:var(--muted)">
           Subscribe to <strong>${escHtml(pushStatus.topic||'')}</strong> in the
           <a href="https://ntfy.sh" style="color:var(--accent)">ntfy app</a> for push alerts.
         </div>`
      : '<div style="color:var(--muted);font-size:.88rem">Push not configured. Add <code>[push] topic = "..."</code> to prism_config.toml then install the ntfy app.</div>';

  } catch (e) {
    document.getElementById('stat-tiles').innerHTML =
      '<div style="grid-column:span 2;color:var(--red);padding:12px">Failed to load status</div>';
  }
}

// ── Voice input ───────────────────────────────────────────────────────────
const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

function setupVoice() {
  const micBtn   = document.getElementById('mic-btn');
  const voiceBtn = document.getElementById('voice-btn');   // chat bar mic
  const txBox    = document.getElementById('transcript-box');
  const sendVoice= document.getElementById('voice-send');
  const voiceSts = document.getElementById('voice-status');

  if (!SpeechRecognition) {
    voiceSts.textContent = 'Speech recognition not available in this browser.';
    micBtn.style.opacity = '0.4'; micBtn.style.cursor = 'default';
    return;
  }

  function startRec(onResult, statusEl, btn) {
    if (isRecording) { recognition && recognition.stop(); return; }
    recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = true;
    recognition.lang = 'en-US';
    isRecording = true;
    btn.classList.add('recording');
    if (statusEl) statusEl.textContent = 'Listening…';

    recognition.onresult = e => {
      const t = Array.from(e.results).map(r => r[0].transcript).join('');
      onResult(t, e.results[e.results.length-1].isFinal);
    };
    recognition.onend = () => {
      isRecording = false;
      btn.classList.remove('recording');
    };
    recognition.onerror = err => {
      isRecording = false;
      btn.classList.remove('recording');
      if (statusEl) statusEl.textContent = 'Error: ' + err.error;
    };
    recognition.start();
  }

  // Big mic on Voice tab
  micBtn.addEventListener('click', () => {
    if (isRecording) { recognition.stop(); return; }
    txBox.textContent = '';
    sendVoice.style.display = 'none';
    pendingTranscript = '';
    startRec((text, final) => {
      txBox.textContent = text;
      pendingTranscript = text;
      if (final) {
        voiceSts.textContent = 'Tap "Send to PRISM" or speak again';
        sendVoice.style.display = 'block';
      }
    }, voiceSts, micBtn);
  });

  sendVoice.addEventListener('click', () => {
    if (!pendingTranscript.trim()) return;
    showPanel('chat');
    sendChat(pendingTranscript);
    pendingTranscript = '';
    sendVoice.style.display = 'none';
  });

  // Small mic in chat input bar
  voiceBtn.addEventListener('click', () => {
    if (isRecording) { recognition.stop(); return; }
    startRec((text, final) => {
      chatInput.value = text;
      if (final) sendChat(text);
    }, null, voiceBtn);
  });
}

// ── Markdown micro-renderer ───────────────────────────────────────────────
function md(text) {
  return escHtml(text)
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.+?)\*/g, '<em>$1</em>')
    .replace(/`(.+?)`/g, '<code style="background:rgba(127,127,127,.15);padding:1px 4px;border-radius:4px">$1</code>')
    .replace(/\n/g, '<br>');
}

function escHtml(s) {
  return String(s)
    .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
    .replace(/"/g,'&quot;').replace(/'/g,'&#39;');
}

// ── Init ──────────────────────────────────────────────────────────────────
async function init() {
  showPanel('chat');
  setupVoice();

  // Greet
  appendMsg('system', 'PRISM mobile — connected to ' + API);

  // Load status in background for the status badge
  try {
    const s = await fetch(API+'/status').then(r=>r.json());
    const dot = document.getElementById('s-dot');
    dot.className = 'dot' + (s.ollama_available ? ' online' : '');
    document.getElementById('s-text').textContent =
      (s.profile||'PRISM') + (s.ollama_available ? ' · online' : ' · LLM offline');
  } catch (e) {
    document.getElementById('s-text').textContent = 'Offline';
  }
}

init();
</script>
</body>
</html>"""


def get_mobile_html() -> str:
    return MOBILE_HTML


def get_manifest() -> str:
    return MANIFEST_JSON


def get_service_worker() -> str:
    return SERVICE_WORKER_JS


def get_icon_svg() -> str:
    return ICON_SVG
