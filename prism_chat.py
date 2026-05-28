PRISM_CHAT_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>PRISM</title>
  <style>
    :root{--bg:#0f1117;--surface:#1a1f2e;--border:rgba(255,255,255,.08);--text:#e8eaf0;--muted:#6b7280;--accent:#5DCAA5;--red:#E24B4A;--yellow:#EF9F27;--blue:#3B8BD4}
    @media(prefers-color-scheme:light){:root{--bg:#f0f2f5;--surface:#fff;--border:rgba(0,0,0,.08);--text:#1a1f2e}}
    *{box-sizing:border-box}html,body{height:100%}body{margin:0;font:14px/1.45 Inter,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;background:var(--bg);color:var(--text)}
    .app{min-height:100vh;display:grid;grid-template-columns:240px minmax(0,1fr);gap:20px;padding:20px}
    .panel{background:var(--surface);border:1px solid var(--border);border-radius:20px;box-shadow:0 16px 40px rgba(0,0,0,.18)}
    .sidebar{padding:20px;display:flex;flex-direction:column;gap:18px}
    .brand{font-size:22px;font-weight:700;letter-spacing:.08em}
    .section{padding-top:16px;border-top:1px solid var(--border)}
    .section:first-of-type{padding-top:0;border-top:0}
    .section-title{margin:0 0 10px;font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}
    .nav-item,.mini-item{width:100%;display:flex;align-items:center;gap:10px;padding:10px 12px;border-radius:12px;border:1px solid transparent;background:transparent;color:var(--text);text-align:left;cursor:pointer}
    .nav-item:hover,.mini-item:hover{border-color:var(--border);background:rgba(255,255,255,.03)}
    .nav-item.active{background:rgba(93,202,165,.12);border-color:rgba(93,202,165,.3)}
    .dot{width:9px;height:9px;border-radius:50%;background:var(--muted);flex:0 0 auto}
    .dot.connected{background:var(--accent);box-shadow:0 0 0 4px rgba(93,202,165,.12)}
    .dot.offline{background:var(--red);box-shadow:0 0 0 4px rgba(226,75,74,.12)}
    .sidebar-status{margin-top:auto;display:flex;align-items:center;gap:10px;padding:12px;border:1px solid var(--border);border-radius:14px}
    .chat-shell{display:flex;flex-direction:column;min-height:calc(100vh - 40px);overflow:hidden}
    .chat-header{display:flex;align-items:center;justify-content:space-between;padding:18px 20px;border-bottom:1px solid var(--border)}
    .chat-title{font-size:18px;font-weight:700}
    .chat-subtitle{color:var(--muted);font-size:13px}
    .messages{flex:1;overflow:auto;padding:20px;display:flex;flex-direction:column;gap:14px;background:linear-gradient(180deg,rgba(255,255,255,.02),transparent)}
    .message{max-width:min(760px,100%);display:flex;flex-direction:column;gap:8px}
    .message.user{margin-left:auto;align-items:flex-end}
    .message-label{font-size:12px;color:var(--muted);padding:0 4px}
    .bubble,.card{border:1px solid var(--border);border-radius:18px;padding:14px 16px;background:rgba(255,255,255,.03)}
    .message.user .bubble{background:rgba(59,139,212,.18);border-color:rgba(59,139,212,.25)}
    .card-title{font-weight:700;margin:0 0 10px}
    .card-body{color:var(--text)}
    .row{display:flex;align-items:center;gap:10px}
    .row + .row{margin-top:10px}
    .grow{flex:1}
    .meta{display:flex;flex-wrap:wrap;gap:8px;margin-top:12px}
    .meta span{padding:6px 10px;border-radius:999px;background:rgba(255,255,255,.05);border:1px solid var(--border);color:var(--muted);font-size:12px}
    .schedule-time{width:74px;color:var(--muted);font-variant-numeric:tabular-nums}
    .mini-dot{width:8px;height:8px;border-radius:50%;background:var(--accent)}
    .bar-track{height:10px;border-radius:999px;background:rgba(255,255,255,.07);overflow:hidden;border:1px solid var(--border)}
    .bar-fill{height:100%;border-radius:999px;background:linear-gradient(90deg,var(--accent),var(--blue))}
    .bar-fill.warn{background:linear-gradient(90deg,var(--yellow),var(--red))}
    .bar-fill.blue{background:linear-gradient(90deg,var(--blue),var(--accent))}
    .bar-label{display:flex;justify-content:space-between;gap:10px;margin-bottom:6px;font-size:13px}
    .badge{padding:4px 8px;border-radius:999px;font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.04em}
    .badge.low{background:rgba(93,202,165,.15);color:var(--accent)}
    .badge.medium{background:rgba(239,159,39,.14);color:var(--yellow)}
    .badge.high,.badge.critical{background:rgba(226,75,74,.14);color:var(--red)}
    .chips{padding:14px 20px;border-top:1px solid var(--border);display:flex;flex-wrap:wrap;gap:10px}
    .chip{padding:10px 12px;border-radius:999px;border:1px solid var(--border);background:transparent;color:var(--text);cursor:pointer}
    .chip:hover{border-color:rgba(93,202,165,.35);background:rgba(93,202,165,.08)}
    .composer{padding:16px 20px;border-top:1px solid var(--border);display:grid;grid-template-columns:minmax(0,1fr) auto;gap:12px}
    textarea{width:100%;min-height:64px;max-height:180px;resize:vertical;padding:14px 16px;border-radius:16px;border:1px solid var(--border);background:rgba(255,255,255,.03);color:var(--text);font:inherit}
    button.send{min-width:56px;border:0;border-radius:16px;background:var(--accent);color:#08120f;font-weight:800;cursor:pointer;padding:0 18px}
    .muted{color:var(--muted)} .insight{margin-top:12px;padding:12px;border-radius:14px;background:rgba(255,255,255,.04);border:1px solid var(--border)}
    @media(max-width:900px){.app{grid-template-columns:1fr;padding:12px}.chat-shell{min-height:70vh}.sidebar{padding:16px}}
    @media(max-width:640px){.chat-header,.messages,.chips,.composer{padding-left:14px;padding-right:14px}.composer{grid-template-columns:1fr}.message{max-width:100%}}
  </style>
</head>
<body>
  <div class="app">
    <aside class="panel sidebar">
      <div class="brand">◈ PRISM</div>
      <section class="section">
        <h2 class="section-title">Modules</h2>
        <button class="nav-item active" data-mode="general"><span class="dot connected"></span><span>General</span></button>
        <button class="nav-item" data-mode="sport"><span class="dot"></span><span>Sport</span></button>
        <button class="nav-item" data-mode="medical"><span class="dot"></span><span>Medical</span></button>
        <button class="nav-item" data-mode="financial"><span class="dot"></span><span>Financial</span></button>
        <button class="nav-item" data-mode="legal"><span class="dot"></span><span>Legal</span></button>
        <button class="nav-item" data-mode="developer"><span class="dot"></span><span>Developer(KSA)</span></button>
      </section>
      <section class="section">
        <h2 class="section-title">Identity</h2>
        <button class="mini-item" data-chip="My profile">My profile</button>
        <button class="mini-item" data-chip="Artifacts">Artifacts</button>
      </section>
      <div class="sidebar-status">
        <span class="dot" id="sidebar-status-dot"></span>
        <div>
          <div id="sidebar-status-label">checking</div>
          <div class="muted" id="sidebar-status-detail">Trying local server</div>
        </div>
      </div>
    </aside>

    <main class="panel chat-shell">
      <header class="chat-header">
        <div>
          <div class="chat-title">PRISM</div>
          <div class="chat-subtitle">Local intelligence chat</div>
        </div>
        <div class="chat-subtitle" id="header-status">checking…</div>
      </header>
      <section class="messages" id="messages"></section>
      <section class="chips" id="chips"></section>
      <form class="composer" id="composer">
        <textarea id="composer-input" placeholder="Ask PRISM to plan, predict, triage, profile, or inspect developer tasks."></textarea>
        <button class="send" type="submit" aria-label="Send">▲</button>
      </form>
    </main>
  </div>

  <script>
    const API = 'http://127.0.0.1:8742';
    const MODE_CHIPS = {
      general:   ['Plan my day','Match prediction','Injury risk','My identity','Status'],
      sport:     ['Morning briefing','Predict next match','Squad risk','Analyse footage'],
      medical:   ['Triage: chest pain, elderly','Triage: child high fever'],
      financial: ['Portfolio: 35yr growth','Portfolio: retiree','Bear market'],
      legal:     ['Strong evidence, deep pockets','Weak case, time pressure'],
      developer: ['Scan project files','Search TODOs','Run tests quietly'],
    };

    const state = { mode: 'general', status: 'checking', history: [] };
    const messagesEl = document.getElementById('messages');
    const chipsEl = document.getElementById('chips');
    const inputEl = document.getElementById('composer-input');
    const headerStatusEl = document.getElementById('header-status');
    const sidebarStatusDotEl = document.getElementById('sidebar-status-dot');
    const sidebarStatusLabelEl = document.getElementById('sidebar-status-label');
    const sidebarStatusDetailEl = document.getElementById('sidebar-status-detail');

    function escapeHtml(value) {
      return String(value ?? '').replace(/[&<>"]/g, function(ch) {
        return ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'})[ch];
      });
    }

    function clampPercent(value) {
      const n = Number(value);
      if (!Number.isFinite(n)) return 0;
      return Math.max(0, Math.min(100, Math.round((n <= 1 ? n * 100 : n))));
    }

    function riskClass(level) {
      const text = String(level || '').toLowerCase();
      if (text.includes('high') || text.includes('critical')) return 'high';
      if (text.includes('medium') || text.includes('moderate')) return 'medium';
      return 'low';
    }

    function updateStatus(status, detail) {
      state.status = status;
      const connected = status === 'connected';
      headerStatusEl.textContent = 'PRISM · ' + status;
      sidebarStatusLabelEl.textContent = status;
      sidebarStatusDetailEl.textContent = detail || (connected ? 'Local server ready' : 'Demo fallback enabled');
      sidebarStatusDotEl.className = 'dot ' + (connected ? 'connected' : 'offline');
    }

    async function callPrism(message, mode, context) {
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 5000);
      try {
        const response = await fetch(API + '/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message, mode, context }),
          signal: controller.signal
        });
        clearTimeout(timeout);
        if (!response.ok) throw new Error('HTTP ' + response.status);
        updateStatus('connected', 'Serving local PRISM responses');
        return await response.json();
      } catch (error) {
        clearTimeout(timeout);
        updateStatus('demo mode', 'Server offline — using local examples');
        return demoFallback(message);
      }
    }

    function renderCard(card) {
      const data = card && card.data ? card.data : {};
      const title = card && card.title ? '<div class="card-title">' + escapeHtml(card.title) + '</div>' : '';
      const meta = [];
      function metaHtml(items) {
        return items.length ? '<div class="meta">' + items.map(function(item){ return '<span>' + escapeHtml(item) + '</span>'; }).join('') + '</div>' : '';
      }
      switch ((card && card.type) || 'text') {
        case 'plan':
          meta.push('Focus: ' + (data.primary_focus || 'Today'));
          meta.push('Activation ' + clampPercent(data.activation) + '%');
          meta.push(((data.warnings || []).length) + ' warning(s)');
          return '<div class="card">' + title + (data.tasks || []).map(function(task) {
            return '<div class="row"><div class="schedule-time">' + escapeHtml(task.time || 'Anytime') + '</div><div class="mini-dot"></div><div class="grow">' + escapeHtml(task.title || task.category || 'Task') + '</div></div>';
          }).join('') + '<div class="insight">' + escapeHtml(card.body || '') + '</div>' + metaHtml(meta) + '</div>';
        case 'prediction':
          meta.push((data.home || 'Home') + ' vs ' + (data.away || 'Away'));
          meta.push('Confidence ' + clampPercent(data.confidence) + '%');
          return '<div class="card">' + title + [
            ['Home', data.p_home, 'blue'],
            ['Draw', data.p_draw, ''],
            ['Away', data.p_away, 'warn']
          ].map(function(item) {
            return '<div class="row"><div class="grow"><div class="bar-label"><span>' + item[0] + '</span><strong>' + clampPercent(item[1]) + '%</strong></div><div class="bar-track"><div class="bar-fill ' + item[2] + '" style="width:' + clampPercent(item[1]) + '%"></div></div></div></div>';
          }).join('') + '<div class="insight">' + escapeHtml(card.body || data.predicted || '') + '</div>' + metaHtml(meta) + '</div>';
        case 'risk':
        case 'squad':
          const players = card.type === 'risk'
            ? [{ name: data.athlete || 'Athlete', risk_level: data.risk_level || 'unknown', confidence: data.fulcrum || 0 }]
            : (data.players || []);
          return '<div class="card">' + title + players.map(function(player) {
            const badge = riskClass(player.risk_level);
            const width = clampPercent(player.confidence || 0.5);
            return '<div class="row"><div class="grow"><div class="bar-label"><span>' + escapeHtml(player.name || 'Player') + '</span><span class="badge ' + badge + '">' + escapeHtml(player.risk_level || 'unknown') + '</span></div><div class="bar-track"><div class="bar-fill ' + (badge === 'high' ? 'warn' : '') + '" style="width:' + width + '%"></div></div></div></div>';
          }).join('') + '<div class="insight">' + escapeHtml(card.body || '') + '</div></div>';
        case 'domain':
        case 'moment':
          meta.push('Recommendation: ' + (data.recommended || card.body || 'Ready'));
          if (data.confidence !== undefined) meta.push('Confidence ' + clampPercent(data.confidence) + '%');
          if (data.xg !== undefined) meta.push('xG ' + Number(data.xg).toFixed(2));
          return '<div class="card">' + title + (data.options || []).map(function(option) {
            return '<div class="row"><div class="grow"><div class="bar-label"><span>' + escapeHtml(option.name || 'Option') + '</span><strong>' + clampPercent(option.activation) + '%</strong></div><div class="bar-track"><div class="bar-fill blue" style="width:' + clampPercent(option.activation) + '%"></div></div></div></div>';
          }).join('') + metaHtml(meta) + '</div>';
        case 'identity':
          meta.push((data.n_decisions || 0) + ' signals');
          meta.push('Confidence ' + clampPercent(data.confidence) + '%');
          return '<div class="card">' + title + (data.domains || []).map(function(domain) {
            return '<div class="row"><div class="grow"><div class="bar-label"><span>' + escapeHtml(domain.label || 'Domain') + '</span><span>' + (domain.crystallised ? '● crystallised' : '○ emerging') + '</span></div><div class="bar-track"><div class="bar-fill" style="width:' + clampPercent(domain.value) + '%"></div></div></div></div>';
          }).join('') + '<div class="insight">' + escapeHtml(data.insight || card.body || '') + '</div>' + metaHtml(meta) + '</div>';
        case 'text':
        default:
          return '<div class="card">' + title + '<div class="card-body">' + (card.body || '') + '</div></div>';
      }
    }

    function demoFallback(msg) {
      const text = String(msg || '').toLowerCase();
      if (text.includes('plan') || text.includes('day') || text.includes('morning')) {
        return {type:'plan',title:'Daily plan',body:'Recovery-first day with spaced focus blocks and a lighter afternoon.',data:{primary_focus:'Recovery',activation:0.62,warnings:['Hydrate','Keep workload controlled'],tasks:[{time:'07:30',title:'Mobility + breakfast'},{time:'09:00',title:'Deep work sprint'},{time:'13:30',title:'Walk + reset'},{time:'16:00',title:'Light review'}]},actions:['Adjust my workload']};
      }
      if (text.includes('predict') || text.includes('match')) {
        return {type:'prediction',title:'Match prediction',body:'Home side edge from form and transitions.',data:{home:'PRISM FC',away:'Vector United',p_home:0.52,p_draw:0.24,p_away:0.24,confidence:0.68},actions:['Why this forecast?']};
      }
      if (text.includes('injury') || text.includes('risk') || text.includes('squad')) {
        return {type:'squad',title:'Squad overview',body:'Two monitored players need lighter load decisions.',data:{players:[{name:'A. Malik',risk_level:'high',confidence:0.82},{name:'J. Silva',risk_level:'medium',confidence:0.58},{name:'R. Cole',risk_level:'low',confidence:0.28}]},actions:['Show highest risks']};
      }
      if (text.includes('identity') || text.includes('profile') || text.includes('dna')) {
        return {type:'identity',title:'Identity profile',body:'You favour steady activation with recovery-aware decisions.',data:{domains:[{label:'Fulcrum',value:0.63,crystallised:true},{label:'Recovery focus',value:0.37,crystallised:true},{label:'Day rating',value:0.74,crystallised:false}],insight:'PRISM sees a stable planner who protects future energy.',confidence:0.71,n_decisions:9},actions:['How did I evolve?']};
      }
      if (text.includes('triage') || text.includes('medical')) {
        return {type:'domain',title:'Medical decision',body:'Escalate rapid assessment and monitor deterioration.',data:{recommended:'Urgent review',confidence:0.8,options:[{name:'Observe',activation:0.28},{name:'Urgent review',activation:0.8},{name:'Immediate transfer',activation:0.64}]},actions:['Explain the trade-offs']};
      }
      if (text.includes('portfolio') || text.includes('invest')) {
        return {type:'domain',title:'Financial decision',body:'Balanced growth beats pure aggression under this profile.',data:{recommended:'Balanced growth',confidence:0.73,options:[{name:'Conservative',activation:0.34},{name:'Balanced growth',activation:0.73},{name:'Aggressive tilt',activation:0.49}]},actions:['Compare another profile']};
      }
      return {type:'text',title:'PRISM',body:'Try a quick prompt like <strong>Plan my day</strong>, <strong>Match prediction</strong>, <strong>Triage: chest pain, elderly</strong>, or <strong>My identity</strong>.',data:{},actions:['Plan my day','My identity']};
    }

    function appendMessage(role, payload) {
      const wrap = document.createElement('div');
      wrap.className = 'message ' + role;
      if (role === 'user') {
        wrap.innerHTML = '<div class="message-label">You</div><div class="bubble">' + escapeHtml(payload) + '</div>';
      } else {
        wrap.innerHTML = '<div class="message-label">PRISM</div>' + renderCard(payload);
      }
      messagesEl.appendChild(wrap);
      messagesEl.scrollTop = messagesEl.scrollHeight;
    }

    function buildChips(mode, extraActions) {
      const chipValues = (MODE_CHIPS[mode] || MODE_CHIPS.general).slice();
      (extraActions || []).forEach(function(action) {
        if (action && !chipValues.includes(action)) chipValues.push(action);
      });
      chipsEl.innerHTML = chipValues.map(function(chip) {
        return '<button class="chip" type="button" data-chip="' + escapeHtml(chip) + '">' + escapeHtml(chip) + '</button>';
      }).join('');
    }

    async function sendMessage(message) {
      const text = String(message || '').trim();
      if (!text) return;
      appendMessage('user', text);
      inputEl.value = '';
      const card = await callPrism(text, state.mode, { mode: state.mode, history: state.history.slice(-6) });
      state.history.push({ role: 'user', message: text });
      state.history.push({ role: 'assistant', card: card });
      appendMessage('agent', card);
      buildChips(state.mode, card.actions || []);
    }

    async function initStatus() {
      const controller = new AbortController();
      const timeout = setTimeout(function() { controller.abort(); }, 1500);
      try {
        const response = await fetch(API + '/status', { signal: controller.signal });
        clearTimeout(timeout);
        if (!response.ok) throw new Error('offline');
        updateStatus('connected', 'Local server responded to /status');
      } catch (error) {
        clearTimeout(timeout);
        updateStatus('demo mode', 'Server offline — standalone preview');
      }
    }

    document.getElementById('composer').addEventListener('submit', function(event) {
      event.preventDefault();
      sendMessage(inputEl.value);
    });

    chipsEl.addEventListener('click', function(event) {
      const chip = event.target.closest('[data-chip]');
      if (chip) sendMessage(chip.getAttribute('data-chip'));
    });

    document.querySelector('.sidebar').addEventListener('click', function(event) {
      const modeButton = event.target.closest('[data-mode]');
      const chipButton = event.target.closest('.mini-item[data-chip]');
      if (modeButton) {
        state.mode = modeButton.getAttribute('data-mode');
        document.querySelectorAll('.nav-item').forEach(function(node) { node.classList.remove('active'); });
        modeButton.classList.add('active');
        buildChips(state.mode);
      } else if (chipButton) {
        sendMessage(chipButton.getAttribute('data-chip'));
      }
    });

    initStatus();
    appendMessage('agent', {type:'text',title:'Welcome to PRISM',body:'Ask for plans, predictions, risk checks, identity snapshots, domain decisions, or developer support.',data:{},actions:['Plan my day','Status']});
    buildChips('general');
  </script>
</body>
</html>
"""


def get_chat_html() -> str:
    """Return the PRISM chat SPA."""
    return PRISM_CHAT_HTML
