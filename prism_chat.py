def get_chat_html() -> str:
    return """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PRISM</title>
<style>
:root{--bg:#0f1117;--sf:#171b22;--br:rgba(255,255,255,.08);--tx:#ecf0f6;--mu:#9aa4b2;--ac:#5DCAA5;--rd:#E24B4A;--ye:#EF9F27;--bl:#3B8BD4}
@media (prefers-color-scheme: light){:root{--bg:#f5f7fb;--sf:#ffffff;--br:rgba(15,23,42,.12);--tx:#162030;--mu:#5b6778}}
*{box-sizing:border-box}html,body{height:100%}body{margin:0;background:var(--bg);color:var(--tx);font:14px/1.45 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
.app{display:grid;grid-template-columns:220px minmax(0,1fr);min-height:100vh}.sidebar{border-right:1px solid var(--br);background:var(--sf);padding:18px;display:flex;flex-direction:column;gap:18px}.brand{font-weight:800;letter-spacing:.1em}.group-title{font-size:11px;color:var(--mu);text-transform:uppercase;letter-spacing:.08em;margin:0 0 8px}.nav button,.identity button,.quick button,.send{appearance:none;border:1px solid var(--br);background:transparent;color:var(--tx);border-radius:12px;padding:10px 12px;text-align:left;cursor:pointer}.nav,.identity,.quick{display:grid;gap:8px}.status{margin-top:auto;padding:12px;border:1px solid var(--br);border-radius:14px;color:var(--mu)}.main{display:flex;flex-direction:column;min-width:0}.header{padding:18px 20px;border-bottom:1px solid var(--br);background:var(--sf);display:flex;justify-content:space-between;gap:12px;align-items:center}.messages{flex:1;overflow:auto;padding:20px;display:flex;flex-direction:column;gap:14px}.msg{max-width:860px}.msg.user{align-self:flex-end}.bubble,.card{border:1px solid var(--br);background:var(--sf);border-radius:16px;padding:14px 16px}.msg.user .bubble{background:rgba(59,139,212,.14)}.label{font-size:11px;color:var(--mu);margin:0 0 6px 4px}.card h3{margin:0 0 8px;font-size:15px}.sub{color:var(--mu)}.row{display:flex;gap:10px;align-items:center}.row+.row,.stack+.stack{margin-top:10px}.time{width:72px;color:var(--mu)}.bar{height:10px;border-radius:999px;background:rgba(255,255,255,.06);border:1px solid var(--br);overflow:hidden;flex:1}.fill{height:100%;background:linear-gradient(90deg,var(--ac),var(--bl))}.fill.warn{background:linear-gradient(90deg,var(--ye),var(--rd))}.pill{display:inline-block;padding:4px 8px;border-radius:999px;border:1px solid var(--br);font-size:11px;color:var(--mu);margin:8px 8px 0 0}.composer{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:12px;padding:16px 20px;border-top:1px solid var(--br);background:var(--sf)}textarea{width:100%;min-height:64px;max-height:180px;resize:vertical;border-radius:14px;border:1px solid var(--br);background:transparent;color:var(--tx);padding:12px 14px;font:inherit}.send{background:var(--ac);color:#082117;font-weight:700;padding:0 18px}.quick{padding:0 20px 16px}.muted{color:var(--mu)}
@media (max-width:580px){.app{grid-template-columns:1fr}.sidebar{display:none}.composer{grid-template-columns:1fr}.messages,.header,.composer,.quick{padding-left:14px;padding-right:14px}}
</style>
</head>
<body>
<div class="app">
  <aside class="sidebar">
    <div class="brand">PRISM</div>
    <div>
      <p class="group-title">Modules</p>
      <div class="nav">
        <button data-mode="General">General</button>
        <button data-mode="Sport">Sport</button>
        <button data-mode="Medical">Medical</button>
        <button data-mode="Financial">Financial</button>
        <button data-mode="Legal">Legal</button>
        <button data-mode="Developer">Developer</button>
      </div>
    </div>
    <div>
      <p class="group-title">Identity</p>
      <div class="identity">
        <button data-demo="identity profile">Profile</button>
        <button data-demo="artifacts">Artifacts</button>
      </div>
    </div>
    <div class="status">Status: <span id="statusText">checking…</span></div>
  </aside>
  <main class="main">
    <div class="header"><div><strong>PRISM Chat</strong><div class="sub">Unified platform layer</div></div><div id="roleText" class="sub">offline</div></div>
    <div id="messages" class="messages"></div>
    <div class="quick" id="quick"></div>
    <form class="composer" id="composer">
      <textarea id="message" placeholder="Ask PRISM anything…"></textarea>
      <button class="send" type="submit">Send</button>
    </form>
  </main>
</div>
<script>
const API='http://127.0.0.1:8742';
const state={mode:'General'};
const messages=document.getElementById('messages');
const quick=document.getElementById('quick');
const input=document.getElementById('message');
const statusText=document.getElementById('statusText');
const roleText=document.getElementById('roleText');
const quickPrompts=['Plan my day','Predict City vs Arsenal','Show squad risk','Medical triage','Portfolio invest','Identity profile'];
function esc(value){return String(value==null?'':value).replace(/[&<>"]/g,function(ch){return({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'})[ch];});}
function pct(value){const n=Number(value);if(!Number.isFinite(n))return 0;return Math.max(0,Math.min(100,Math.round((n<=1?n*100:n))));}
function bubble(role,html){const wrap=document.createElement('div');wrap.className='msg '+role;wrap.innerHTML='<div class="label">'+(role==='user'?'You':'PRISM')+'</div>'+html;messages.appendChild(wrap);messages.scrollTop=messages.scrollHeight;}
function renderCard(card){card=card||{type:'text',body:''};const data=card.data||{};if(card.type==='plan'){return '<div class="card"><h3>'+esc(card.title||'Daily Plan')+'</h3>'+(data.tasks||[]).map(function(task){return '<div class="row"><div class="time">'+esc(task.time||'')+'</div><div>'+esc(task.title||task.category||'Task')+'</div></div>';}).join('')+'<div class="pill">'+esc(data.primary_focus||'')+'</div></div>';}if(card.type==='prediction'){return '<div class="card"><h3>'+esc(card.title||'Match Prediction')+'</h3><div class="stack"><div>'+esc(data.home||'Home')+' vs '+esc(data.away||'Away')+'</div><div class="row"><span>Home</span><div class="bar"><div class="fill" style="width:'+pct(data.p_home)+'%"></div></div><strong>'+pct(data.p_home)+'%</strong></div><div class="row"><span>Draw</span><div class="bar"><div class="fill" style="width:'+pct(data.p_draw)+'%"></div></div><strong>'+pct(data.p_draw)+'%</strong></div><div class="row"><span>Away</span><div class="bar"><div class="fill warn" style="width:'+pct(data.p_away)+'%"></div></div><strong>'+pct(data.p_away)+'%</strong></div></div></div>';}if(card.type==='squad'){return '<div class="card"><h3>'+esc(card.title||'Squad Risk Overview')+'</h3>'+(data.players||[]).map(function(player){return '<div class="row"><span>'+esc(player.name)+'</span><div class="bar"><div class="fill warn" style="width:'+pct(player.confidence)+'%"></div></div><span>'+esc(player.risk_level)+'</span></div>';}).join('')+'</div>';}if(card.type==='domain'){return '<div class="card"><h3>'+esc(card.title||'Domain Recommendation')+'</h3><div class="sub">'+esc(data.recommended||'')+'</div>'+(data.options||[]).map(function(option){return '<div class="row"><span>'+esc(option.name)+'</span><div class="bar"><div class="fill" style="width:'+pct(option.activation)+'%"></div></div><strong>'+pct(option.activation)+'%</strong></div>';}).join('')+'</div>';}if(card.type==='identity'){return '<div class="card"><h3>'+esc(card.title||'Your Decision Profile')+'</h3>'+(data.domains||[]).map(function(domain){return '<div class="row"><span>'+esc(domain.label)+'</span><div class="bar"><div class="fill" style="width:'+pct(domain.value)+'%"></div></div><span>'+esc(domain.crystallised?'set':'emerging')+'</span></div>';}).join('')+'<div class="sub">'+esc(data.insight||'')+'</div></div>';}if(c.type==='plan'&&d.strategies){const strats=d.strategies||[];const top=strats[0]||{};const bd='<div class="label">PRISM</div>';const stepRows=(top.steps||[]).slice(0,4).map(function(s){return '<div class="si"><div class="st">'+esc(s.timeline||'')+'</div><div class="sd" style="background:var(--ac)"></div><div class="stx">'+esc(s.action)+'</div></div>';}).join('');const altRows=strats.slice(1,4).map(function(s,i){return '<div class="br"><div class="bl">Alt '+(i+1)+': '+esc(s.name)+'</div><div class="bt"><div class="bf" style="width:'+(s.activation*100).toFixed(0)+'%;background:var(--mu);opacity:.6"></div></div><div class="bp">'+(s.activation*100).toFixed(0)+'%</div></div>';}).join('');return bd+'<div class="card"><div class="card-t">\u2605 '+esc(top.name||'Optimal strategy')+' <span style="font-size:10px;opacity:.5">('+((top.activation||0)*100).toFixed(0)+'%)</span></div>'+stepRows+altRows+'<div class="meta">Domain <b>'+esc(d.domain||'')+'</b> \u00b7 Timeline <b>'+esc(d.timeline||'')+'</b> \u00b7 Fulcrum <b>'+(d.fulcrum||0).toFixed(3)+'</b></div></div>';}return '<div class="card"><h3>'+esc(card.title||'PRISM')+'</h3><div>'+String(card.body||'')+'</div></div>';}
function demo(msg){const text=String(msg||'').toLowerCase();if(/plan|day|morning/.test(text)){return {type:'plan',title:'Daily Plan',body:'',data:{primary_focus:'Recovery',tasks:[{time:'07:00',category:'prep',title:'Mobility primer',duration:20},{time:'09:00',category:'focus',title:'Deep work block',duration:90},{time:'13:00',category:'review',title:'Film review',duration:45},{time:'17:00',category:'reset',title:'Recovery walk',duration:30}]}};}if(/predict|match|vs/.test(text)){return {type:'prediction',title:'Match Prediction',body:'',data:{home:'City',away:'Arsenal',p_home:0.46,p_draw:0.26,p_away:0.28,predicted:'City',confidence:0.68}};}if(/injury|risk|squad/.test(text)){return {type:'squad',title:'Squad Risk Overview',body:'',data:{players:[{name:'Player A',risk_level:'high',confidence:0.82},{name:'Player B',risk_level:'medium',confidence:0.61},{name:'Player C',risk_level:'low',confidence:0.24}]}};}if(/identity|profile/.test(text)){return {type:'identity',title:'Your Decision Profile',body:'',data:{domains:[{label:'Medical',value:0.74,crystallised:true},{label:'Financial',value:0.43,crystallised:true},{label:'Sport',value:0.58,crystallised:false},{label:'Developer',value:0.36,crystallised:false}],insight:'Pattern still crystallising',confidence:0.63,n_decisions:12}};}if(/triage|medical/.test(text)){return {type:'domain',title:'Medical Recommendation',body:'',data:{domain:'Medical',recommended:'Urgent_GP',confidence:0.78,fulcrum:0.61,options:[{name:'GP',activation:0.42},{name:'Urgent_GP',activation:0.78},{name:'AE_4hr',activation:0.57}]}};}if(/portfolio|invest/.test(text)){return {type:'domain',title:'Financial Recommendation',body:'',data:{domain:'Financial',recommended:'Mod_balanced',confidence:0.74,fulcrum:0.49,options:[{name:'Gov_bonds',activation:0.31},{name:'Mod_balanced',activation:0.74},{name:'Growth',activation:0.52}]}};}return {type:'text',title:'PRISM',body:'Try: plan my day, predict City vs Arsenal, squad risk, identity profile, medical triage, or portfolio invest.',data:{}};}
async function send(override){const message=(override!=null?override:input.value).trim();if(!message)return;bubble('user','<div class="bubble">'+esc(message)+'</div>');input.value='';try{const response=await fetch(API+'/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:message,mode:state.mode})});if(!response.ok)throw new Error('offline');bubble('agent',renderCard(await response.json()));}catch(error){bubble('agent',renderCard(demo(message)));}}
function renderQuick(){quick.innerHTML=quickPrompts.map(function(prompt){return '<button type="button" data-prompt="'+esc(prompt)+'">'+esc(prompt)+'</button>';}).join('');}
document.getElementById('composer').addEventListener('submit',function(event){event.preventDefault();send();});
document.body.addEventListener('click',function(event){const prompt=event.target.getAttribute('data-prompt');const mode=event.target.getAttribute('data-mode');const demoPrompt=event.target.getAttribute('data-demo');if(prompt)send(prompt);if(demoPrompt)send(demoPrompt);if(mode)state.mode=mode;});
(function(){const controller=new AbortController();const timer=setTimeout(function(){controller.abort();},1500);fetch(API+'/status',{signal:controller.signal}).then(function(response){return response.json();}).then(function(data){clearTimeout(timer);statusText.textContent='online';roleText.textContent=data.role||'connected';}).catch(function(){clearTimeout(timer);statusText.textContent='offline';roleText.textContent='demo';});renderQuick();setTimeout(function(){bubble('agent',renderCard({type:'text',title:'PRISM',body:'Hello — ask for plans, predictions, domain decisions, identity, or developer help.',data:{}}));},250);})();
</script>
</body>
</html>"""
