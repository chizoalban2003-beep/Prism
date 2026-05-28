from __future__ import annotations

UI_HTML: str = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>KDE — Kinetic Decision Engine</title>
<style>
:root {
  --bg: #0f1117; --surface: #1a1f2e; --surface-2: #131824; --border: rgba(255,255,255,0.08);
  --text: #e8eaf0; --muted: #93a0b5; --accent: #5DCAA5; --accent-2:#3B8BD4;
  --red: #E24B4A; --yellow: #EF9F27; --blue: #3B8BD4; --radius: 12px;
  --font: system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
}
@media (prefers-color-scheme: light) {
  :root {
    --bg:#f5f7fa; --surface:#ffffff; --surface-2:#eef2f7; --border:rgba(0,0,0,0.08);
    --text:#1a1f2e; --muted:#5f6b7d;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0; font-family: var(--font); background: var(--bg); color: var(--text);
}
.app { min-height: 100vh; display: flex; flex-direction: column; }
header {
  position: sticky; top: 0; z-index: 10; background: rgba(15,17,23,0.92);
  backdrop-filter: blur(10px); border-bottom: 1px solid var(--border);
}
@media (prefers-color-scheme: light) { header { background: rgba(245,247,250,0.92); } }
.topbar, .content { width: min(1200px, calc(100% - 24px)); margin: 0 auto; }
.topbar { display:flex; gap:16px; align-items:center; justify-content:space-between; padding:16px 0; }
.brand { display:flex; flex-direction:column; }
.brand strong { font-size: 1.05rem; }
.brand span { color: var(--muted); font-size: 0.9rem; }
.status-badge {
  display:inline-flex; align-items:center; gap:8px; padding:8px 12px;
  border-radius: 999px; background: var(--surface); border:1px solid var(--border);
}
.dot { width:10px; height:10px; border-radius:50%; background: var(--yellow); }
.dot.online { background: var(--accent); }
nav { display:flex; gap:8px; flex-wrap:wrap; padding-bottom:16px; }
.tab-btn {
  border:1px solid var(--border); background: var(--surface); color: var(--text);
  border-radius:999px; padding:10px 14px; cursor:pointer; font-weight:600;
}
.tab-btn.active { background: var(--accent); color:#08110d; border-color: transparent; }
.content { padding: 24px 0 40px; }
.tab { display:none; }
.tab.active { display:block; }
.grid { display:grid; grid-template-columns: repeat(12, 1fr); gap:16px; }
.card {
  grid-column: span 12; background: var(--surface); border:1px solid var(--border);
  border-radius: var(--radius); padding:16px;
}
.card h2, .card h3 { margin:0 0 12px; }
.two-up { grid-column: span 6; }
.six-up { grid-column: span 6; }
.three-up { grid-column: span 4; }
label { display:block; font-size:0.9rem; color:var(--muted); margin-bottom:8px; }
input, select, button {
  width:100%; border-radius:10px; border:1px solid var(--border); background: var(--surface-2);
  color:var(--text); padding:10px 12px; font: inherit;
}
button.primary { background: var(--accent); color:#08110d; border: none; font-weight:700; }
button.secondary { background: var(--accent-2); color: white; border: none; }
.controls { display:grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap:12px; }
.slider { display:flex; flex-direction:column; gap:6px; }
.slider-value { color: var(--muted); font-size:0.85rem; }
.list, .bars, .schedule, .chips { display:flex; flex-direction:column; gap:10px; }
.list-item, .bar-row, .schedule-row, .chip-row {
  border:1px solid var(--border); background: var(--surface-2); border-radius: 10px; padding: 12px;
}
.bar-track { width:100%; height:10px; background: rgba(127,127,127,0.18); border-radius:999px; overflow:hidden; }
.bar-fill { height:100%; background: linear-gradient(90deg, var(--accent), var(--accent-2)); }
.gauge {
  width: 128px; height:128px; border-radius:50%; margin: 8px auto 0;
  display:grid; place-items:center; background:
    radial-gradient(circle at center, var(--surface) 48%, transparent 49%),
    conic-gradient(var(--accent) var(--value, 0%), rgba(127,127,127,0.18) 0);
}
.gauge span { font-size: 1.35rem; font-weight: 700; }
.muted { color: var(--muted); }
.loading, .error { padding:12px; border-radius:10px; }
.loading { background: rgba(93,202,165,0.12); }
.error { background: rgba(226,75,74,0.12); color: var(--red); }
.inline { display:flex; gap:12px; align-items:center; }
.chip-wrap { display:flex; flex-wrap:wrap; gap:8px; }
.chip {
  padding:8px 10px; border-radius:999px; background: var(--surface-2);
  border:1px solid var(--border); font-size:0.9rem;
}
@media (max-width: 900px) {
  .two-up, .six-up, .three-up { grid-column: span 12; }
}
@media (max-width: 600px) {
  .topbar { align-items:flex-start; flex-direction:column; }
  nav { overflow:auto; white-space:nowrap; flex-wrap:nowrap; width:100%; padding-bottom:8px; }
}
</style>
</head>
<body>
<div class="app">
  <header>
    <div class="topbar">
      <div class="brand">
        <strong>KDE</strong>
        <span>Kinetic Decision Engine</span>
      </div>
      <div class="status-badge">
        <span class="dot" id="status-dot"></span>
        <span id="status-text">Loading…</span>
      </div>
    </div>
    <div class="topbar">
      <nav id="tabs">
        <button class="tab-btn active" data-tab="morning">Morning</button>
        <button class="tab-btn" data-tab="match">Match</button>
        <button class="tab-btn" data-tab="moment">Moment</button>
        <button class="tab-btn" data-tab="domains">Domains</button>
        <button class="tab-btn" data-tab="settings">Settings</button>
      </nav>
    </div>
  </header>
  <main class="content">
    <section class="tab active" id="tab-morning">
      <div class="grid">
        <div class="card three-up">
          <h2>Recovery</h2>
          <div class="gauge" id="recovery-gauge"><span>--%</span></div>
          <p class="muted" id="wearable-summary">Waiting for /plan</p>
        </div>
        <div class="card three-up">
          <h2>Plan Focus</h2>
          <p id="plan-focus">—</p>
          <p class="muted" id="plan-rationale">Load a morning plan to see details.</p>
        </div>
        <div class="card six-up">
          <h2>Daily Plan</h2>
          <div class="list" id="plan-list"></div>
        </div>
        <div class="card six-up">
          <div class="inline" style="justify-content:space-between">
            <h2>Schedule</h2>
            <button class="secondary" id="refresh-morning">Refresh</button>
          </div>
          <div class="schedule" id="schedule-list"></div>
        </div>
        <div class="card six-up">
          <h2>Reflection</h2>
          <div class="chips" id="reflect-list"></div>
        </div>
      </div>
    </section>

    <section class="tab" id="tab-match">
      <div class="grid">
        <div class="card six-up">
          <h2>Match Predictor</h2>
          <div class="controls">
            <label>Home team<input id="match-home" value="Home Team"></label>
            <label>Away team<input id="match-away" value="Away Team"></label>
            <label>Sport<input id="match-sport" value="football"></label>
            <div class="slider"><label>Home form<input type="range" min="0" max="1" step="0.05" value="0.5" id="match-home-form"></label><span class="slider-value" id="match-home-form-value">0.50</span></div>
            <div class="slider"><label>Away form<input type="range" min="0" max="1" step="0.05" value="0.5" id="match-away-form"></label><span class="slider-value" id="match-away-form-value">0.50</span></div>
          </div>
          <div style="margin-top:12px"><button class="primary" id="run-match">Run match prediction</button></div>
        </div>
        <div class="card six-up">
          <h2>Win / Draw / Loss</h2>
          <div class="bars" id="match-bars"></div>
          <div class="list" id="match-meta"></div>
        </div>
      </div>
    </section>

    <section class="tab" id="tab-moment">
      <div class="grid">
        <div class="card six-up">
          <h2>Moment Analyzer</h2>
          <div class="controls">
            <label>Sport<select id="moment-sport"></select></label>
            <label>Moment type<select id="moment-type"></select></label>
            <label>Player<input id="moment-player" value="Player"></label>
            <div class="slider"><label>Base<input type="range" min="0" max="1" step="0.05" value="0.6" id="moment-base"></label><span class="slider-value" id="moment-base-value">0.60</span></div>
            <div class="slider"><label>Pitch X<input type="range" min="0" max="1" step="0.05" value="0.8" id="moment-pitch-x"></label><span class="slider-value" id="moment-pitch-x-value">0.80</span></div>
            <div class="slider"><label>Pitch Y<input type="range" min="0" max="1" step="0.05" value="0.5" id="moment-pitch-y"></label><span class="slider-value" id="moment-pitch-y-value">0.50</span></div>
            <div class="slider"><label>Fatigue<input type="range" min="0" max="1" step="0.05" value="0.2" id="moment-fatigue"></label><span class="slider-value" id="moment-fatigue-value">0.20</span></div>
            <div class="slider"><label>Confidence<input type="range" min="0" max="1" step="0.05" value="0.8" id="moment-confidence"></label><span class="slider-value" id="moment-confidence-value">0.80</span></div>
            <label>Goalkeeper<input id="moment-gk" value=""></label>
          </div>
          <div style="margin-top:12px"><button class="primary" id="run-moment">Analyze moment</button></div>
        </div>
        <div class="card six-up">
          <h2>Option Distribution</h2>
          <div class="bars" id="moment-bars"></div>
          <div class="list" id="moment-summary"></div>
        </div>
      </div>
    </section>

    <section class="tab" id="tab-domains">
      <div class="grid">
        <div class="card six-up">
          <h2>Domain Evaluation</h2>
          <div class="controls">
            <label>Domain<select id="domain-name"></select></label>
            <label>Profile<select id="domain-profile"></select></label>
          </div>
          <div class="controls" id="domain-sliders" style="margin-top:12px"></div>
          <div style="margin-top:12px"><button class="primary" id="run-domain">Evaluate domain</button></div>
        </div>
        <div class="card six-up">
          <h2>Domain Output</h2>
          <div class="chip-wrap" id="domain-profile-chips"></div>
          <div class="bars" id="domain-bars" style="margin-top:12px"></div>
          <div class="list" id="domain-summary"></div>
        </div>
      </div>
    </section>

    <section class="tab" id="tab-settings">
      <div class="grid">
        <div class="card six-up">
          <h2>Profile</h2>
          <div class="list" id="profile-info"></div>
        </div>
        <div class="card three-up">
          <h2>Devices</h2>
          <div class="list" id="device-list"></div>
        </div>
        <div class="card three-up">
          <h2>Ollama</h2>
          <div class="list" id="ollama-info"></div>
        </div>
      </div>
    </section>
  </main>
</div>
<script>
const FALLBACK_API = 'http://127.0.0.1:8742';
const API = window.location && window.location.origin && window.location.origin !== 'null'
  ? window.location.origin
  : FALLBACK_API;
const state = { momentConfigs: [], domainList: [], domainProfiles: new Map(), status: null };

async function api(path) {
  const response = await fetch(API + path);
  if (!response.ok) {
    let message = `HTTP ${response.status}`;
    try {
      const data = await response.json();
      message = data.error || message;
    } catch (_) {}
    throw new Error(message);
  }
  return response.json();
}

function setTab(name) {
  document.querySelectorAll('.tab').forEach((tab) => tab.classList.toggle('active', tab.id === `tab-${name}`));
  document.querySelectorAll('.tab-btn').forEach((button) => button.classList.toggle('active', button.dataset.tab === name));
}

function setLoading(targetId, message) {
  document.getElementById(targetId).innerHTML = `<div class="loading">${message}</div>`;
}

function setError(targetId, error) {
  document.getElementById(targetId).innerHTML = `<div class="error">${error.message || error}</div>`;
}

function renderList(targetId, items, formatter) {
  const container = document.getElementById(targetId);
  if (!items || !items.length) {
    container.innerHTML = '<div class="list-item muted">No data available.</div>';
    return;
  }
  container.innerHTML = items.map((item) => `<div class="list-item">${formatter(item)}</div>`).join('');
}

function renderBars(targetId, items, labelKey, valueKey, suffix = '%') {
  const container = document.getElementById(targetId);
  if (!items || !items.length) {
    container.innerHTML = '<div class="bar-row muted">No data available.</div>';
    return;
  }
  container.innerHTML = items.map((item) => {
    const rawValue = Number(item[valueKey] || 0);
    const percent = Math.max(0, Math.min(100, rawValue * 100));
    return `
      <div class="bar-row">
        <div class="inline" style="justify-content:space-between">
          <strong>${item[labelKey]}</strong>
          <span>${percent.toFixed(0)}${suffix}</span>
        </div>
        <div class="bar-track"><div class="bar-fill" style="width:${percent}%"></div></div>
      </div>`;
  }).join('');
}

function attachSliderValue(id) {
  const input = document.getElementById(id);
  const label = document.getElementById(`${id}-value`);
  if (!input || !label) return;
  const update = () => { label.textContent = Number(input.value).toFixed(2); };
  input.addEventListener('input', update);
  update();
}

async function loadStatus() {
  try {
    const data = await api('/status');
    state.status = data;
    document.getElementById('status-dot').classList.toggle('online', !!data.ollama_available);
    document.getElementById('status-text').textContent = `${data.profile || 'KDE'} · ${data.role || 'unknown'}`;
    renderList('profile-info', [
      ['Name', data.profile || '—'],
      ['Role', data.role || '—'],
      ['Sport', data.sport || '—'],
      ['Team', data.team || '—'],
      ['Capabilities', (data.capabilities || []).join(', ') || '—']
    ], ([label, value]) => `<strong>${label}</strong><div class="muted">${value}</div>`);
    renderList('device-list', data.devices || [], (device) => `<strong>${device.name}</strong><div class="muted">${device.enabled ? 'Connected' : 'Disabled'}</div>`);
    renderList('ollama-info', [
      ['Available', data.ollama_available ? 'Yes' : 'No'],
      ['ffmpeg', data.ffmpeg_available ? 'Yes' : 'No'],
      ['Plans', data.plans_this_month ?? 0],
      ['Sessions', data.sessions_this_month ?? 0]
    ], ([label, value]) => `<strong>${label}</strong><div class="muted">${value}</div>`);
  } catch (error) {
    document.getElementById('status-text').textContent = 'Offline';
    setError('profile-info', error);
    setError('device-list', error);
    setError('ollama-info', error);
  }
}

async function loadMorning() {
  setLoading('plan-list', 'Loading plan…');
  setLoading('schedule-list', 'Loading schedule…');
  setLoading('reflect-list', 'Loading reflection…');
  try {
    const [planData, reflectData] = await Promise.all([api('/plan'), api('/reflect')]);
    const activation = Math.max(0, Math.min(100, Number((planData.plan && planData.plan.activation) || 0) * 100));
    const gauge = document.getElementById('recovery-gauge');
    gauge.style.setProperty('--value', `${activation}%`);
    gauge.innerHTML = `<span>${activation.toFixed(0)}%</span>`;
    document.getElementById('wearable-summary').textContent = planData.wearable_summary || 'No wearable summary';
    document.getElementById('plan-focus').textContent = (planData.plan && planData.plan.primary_focus) || '—';
    document.getElementById('plan-rationale').textContent = (planData.plan && planData.plan.rationale) || 'No rationale available.';
    renderList('plan-list', planData.priority_tasks || [], (item) => `<strong>${item}</strong>`);
    renderList('schedule-list', (planData.plan && planData.plan.tasks) || [], (task) =>
      `<strong>${task.time_slot} · ${task.title}</strong><div class="muted">${task.duration_min} min · ${task.category}${task.notes ? ` · ${task.notes}` : ''}</div>`);
    renderList('reflect-list', Object.entries(reflectData || {}), ([key, value]) =>
      `<strong>${key}</strong><div class="muted">${Array.isArray(value) ? value.join(', ') : value}</div>`);
  } catch (error) {
    setError('plan-list', error);
    setError('schedule-list', error);
    setError('reflect-list', error);
  }
}

async function runMatchPrediction() {
  setLoading('match-bars', 'Running prediction…');
  setLoading('match-meta', 'Preparing summary…');
  try {
    const params = new URLSearchParams({
      home: document.getElementById('match-home').value,
      away: document.getElementById('match-away').value,
      sport: document.getElementById('match-sport').value,
      home_form: document.getElementById('match-home-form').value,
      away_form: document.getElementById('match-away-form').value
    });
    const data = await api(`/predict/match?${params.toString()}`);
    renderBars('match-bars', [
      { name: 'Home win', value: data.p_home_win || 0 },
      { name: 'Draw', value: data.p_draw || 0 },
      { name: 'Away win', value: data.p_away_win || 0 }
    ], 'name', 'value');
    renderList('match-meta', [
      ['Prediction', data.prediction || '—'],
      ['Confidence', ((data.confidence || 0) * 100).toFixed(0) + '%'],
      ['Margin', data.predicted_margin ?? '—']
    ], ([label, value]) => `<strong>${label}</strong><div class="muted">${value}</div>`);
  } catch (error) {
    setError('match-bars', error);
    setError('match-meta', error);
  }
}

async function loadMomentConfigs() {
  try {
    const data = await api('/moment/configs');
    state.momentConfigs = data.configs || [];
    const sports = [...new Set(state.momentConfigs.map((item) => item.sport))];
    const sportSelect = document.getElementById('moment-sport');
    sportSelect.innerHTML = sports.map((sport) => `<option value="${sport}">${sport}</option>`).join('');
    updateMomentTypes();
  } catch (error) {
    setError('moment-bars', error);
  }
}

function updateMomentTypes() {
  const sport = document.getElementById('moment-sport').value;
  const types = state.momentConfigs.filter((item) => item.sport === sport);
  const typeSelect = document.getElementById('moment-type');
  typeSelect.innerHTML = types.map((item) => `<option value="${item.moment_type}">${item.moment_type}</option>`).join('');
}

async function runMomentAnalysis() {
  setLoading('moment-bars', 'Analyzing moment…');
  setLoading('moment-summary', 'Preparing context…');
  try {
    const params = new URLSearchParams({
      sport: document.getElementById('moment-sport').value,
      moment_type: document.getElementById('moment-type').value,
      player: document.getElementById('moment-player').value || 'Player',
      base: document.getElementById('moment-base').value,
      pitch_x: document.getElementById('moment-pitch-x').value,
      pitch_y: document.getElementById('moment-pitch-y').value,
      fatigue: document.getElementById('moment-fatigue').value,
      confidence: document.getElementById('moment-confidence').value
    });
    const gk = document.getElementById('moment-gk').value.trim();
    if (gk) {
      params.set('gk_name', gk);
      params.set('gk_distance', '6.0');
    }
    const data = await api(`/moment/analyze?${params.toString()}`);
    renderBars('moment-bars', (data.options || []).map((item) => ({ name: item.name, value: item.activation })), 'name', 'value');
    renderList('moment-summary', [
      ['Recommended', data.recommended || '—'],
      ['Contextual xG', data.xg_contextual ?? '—'],
      ['Time pressure', data.time_pressure ?? '—'],
      ['Fulcrum', data.fulcrum ?? '—']
    ], ([label, value]) => `<strong>${label}</strong><div class="muted">${value}</div>`);
  } catch (error) {
    setError('moment-bars', error);
    setError('moment-summary', error);
  }
}

async function loadDomains() {
  try {
    const data = await api('/domain/list');
    state.domainList = data.domains || [];
    const select = document.getElementById('domain-name');
    select.innerHTML = state.domainList.map((item) => `<option value="${item.domain}">${item.name}</option>`).join('');
    await loadDomainProfiles();
  } catch (error) {
    setError('domain-bars', error);
  }
}

async function loadDomainProfiles() {
  const domain = document.getElementById('domain-name').value;
  if (!domain) return;
  try {
    const [profilesData, listData] = await Promise.all([
      api(`/domain/profiles?domain=${encodeURIComponent(domain)}`),
      api('/domain/list')
    ]);
    const domainEntry = (listData.domains || []).find((item) => item.domain === domain);
    state.domainProfiles.set(domain, profilesData.profiles || []);
    document.getElementById('domain-profile').innerHTML = (profilesData.profiles || []).map((item) => `<option value="${item.name}">${item.name}</option>`).join('');
    document.getElementById('domain-profile-chips').innerHTML = (profilesData.profiles || []).map((item) => `<span class="chip">${item.name}</span>`).join('');
    const defaults = {
      Medical: ['severity', 'vital_signs', 'deteriorating'],
      Financial: ['risk', 'liquidity', 'returns'],
      Legal: ['precedent', 'evidence', 'urgency'],
      HR: ['performance', 'retention', 'fairness'],
      'Supply Chain': ['demand', 'inventory', 'disruption'],
      Climate: ['exposure', 'mitigation', 'resilience']
    };
    const factorIds = defaults[domain] || ['factor_a', 'factor_b', 'factor_c'];
    const sliders = factorIds.map((factor) => `
      <div class="slider">
        <label>${factor}<input type="range" min="0" max="1" step="0.05" value="0.5" id="domain-${factor}"></label>
        <span class="slider-value" id="domain-${factor}-value">0.50</span>
      </div>`).join('');
    document.getElementById('domain-sliders').innerHTML = sliders;
    factorIds.forEach((factor) => attachSliderValue(`domain-${factor}`));
    if (domainEntry) {
      document.getElementById('domain-summary').innerHTML = `<div class="list-item"><strong>${domainEntry.name}</strong><div class="muted">${domainEntry.n_profiles} profiles · ${domainEntry.n_planks} planks</div></div>`;
    }
  } catch (error) {
    setError('domain-summary', error);
  }
}

async function runDomainEvaluation() {
  setLoading('domain-bars', 'Evaluating domain…');
  try {
    const domain = document.getElementById('domain-name').value;
    const profile = document.getElementById('domain-profile').value;
    const params = new URLSearchParams({ domain, profile });
    document.querySelectorAll('#domain-sliders input[type="range"]').forEach((input) => {
      params.set(input.id.replace('domain-', ''), input.value);
    });
    const data = await api(`/domain/evaluate?${params.toString()}`);
    renderBars('domain-bars', (data.options || []).map((item) => ({ name: item.name, value: item.activation })), 'name', 'value');
    renderList('domain-summary', [
      ['Recommended', data.recommended || '—'],
      ['Confidence', ((data.confidence || 0) * 100).toFixed(0) + '%'],
      ['Fulcrum', data.fulcrum ?? '—']
    ].concat((data.key_factors || []).map((item) => [item.name, Number(item.contribution || 0).toFixed(3)])),
    ([label, value]) => `<strong>${label}</strong><div class="muted">${value}</div>`);
  } catch (error) {
    setError('domain-bars', error);
    setError('domain-summary', error);
  }
}

function initEvents() {
  document.querySelectorAll('.tab-btn').forEach((button) => {
    button.addEventListener('click', () => setTab(button.dataset.tab));
  });
  ['match-home-form', 'match-away-form', 'moment-base', 'moment-pitch-x', 'moment-pitch-y', 'moment-fatigue', 'moment-confidence'].forEach(attachSliderValue);
  document.getElementById('refresh-morning').addEventListener('click', loadMorning);
  document.getElementById('run-match').addEventListener('click', runMatchPrediction);
  document.getElementById('run-moment').addEventListener('click', runMomentAnalysis);
  document.getElementById('moment-sport').addEventListener('change', updateMomentTypes);
  document.getElementById('domain-name').addEventListener('change', loadDomainProfiles);
  document.getElementById('run-domain').addEventListener('click', runDomainEvaluation);
}

async function init() {
  initEvents();
  await Promise.all([loadStatus(), loadMorning(), loadMomentConfigs(), loadDomains()]);
  await runMatchPrediction();
}

init().catch((error) => {
  document.getElementById('status-text').textContent = error.message || 'Failed to load';
});
</script>
</body>
</html>"""


def get_ui_html() -> str:
    """Return the full SPA HTML. Called by KDEHandler for GET / and GET /app."""
    return UI_HTML
