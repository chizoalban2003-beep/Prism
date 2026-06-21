#!/usr/bin/env bash
# A day-in-the-life multitask session with PRISM, driven over the local HTTP API.
# Recorded to an animated SVG with termtosvg. Set PRISM_PORT / PRISM_HOME first.
set -u
PORT="${PRISM_PORT:-8760}"; HOME_DIR="${PRISM_HOME:-/tmp/prismdemo2}"
B="http://127.0.0.1:${PORT}"; T="$(cat "${HOME_DIR}/.prism/auth_token")"
A=(-H "Authorization: Bearer ${T}" -H "Content-Type: application/json")

C_U='\033[1;36m'; C_P='\033[0;37m'; C_N='\033[1;33m'; C_G='\033[1;32m'; C_D='\033[0;90m'; Z='\033[0m'

say()  { printf "\n${C_N}# %s${Z}\n" "$1"; sleep 0.7; }
you()  { printf "${C_U}maya ▸${Z} ${C_U}%s${Z}\n" "$1"; sleep 0.5; }
prism(){ printf "${C_D}prism ◂${Z} ${C_P}%s${Z}\n" "$1"; sleep 0.5; }
chat() { you "$1"; r="$(curl -s "${A[@]}" -d "{\"message\":$(python3 -c 'import json,sys;print(json.dumps(sys.argv[1]))' "$1")}" "$B/chat")"; body="$(echo "$r" | python3 -c 'import sys,json,re;d=json.load(sys.stdin);t=re.sub("<[^>]+>"," ",(d.get("body","") or ""));t=re.sub(r"\s+"," ",t).strip();print((d.get("title","")+" — "+t)[:200])')"; prism "$body"; sleep 0.6; }

printf "${C_G}╭──────────────────────────────────────────────────────────────────╮${Z}\n"
printf "${C_G}│  PRISM — a multitask morning with Maya   (all local, 127.0.0.1)   │${Z}\n"
printf "${C_G}╰──────────────────────────────────────────────────────────────────╯${Z}\n"
sleep 1

say "1) Decide — an interpretable call, not a black box"
you "should Arsenal beat Chelsea this weekend?"
curl -s "${A[@]}" "$B/predict/match?home=Arsenal&away=Chelsea&sport=football" | python3 -c "import sys,json;d=json.load(sys.stdin);print('\033[0;37mprism ◂ %s (conf %.0f%%) · key factors: %s\033[0m'%(d['prediction'],d['confidence']*100,', '.join(f for f,_,_ in d['key_factors'])))"
sleep 1

say "2) Same engine, life-or-death domain — medical triage spectrum"
you "triage: chest tightness, short of breath"
curl -s "${A[@]}" "$B/domain/evaluate?domain=Medical" | python3 -c "import sys,json;d=json.load(sys.stdin);print('\033[0;37mprism ◂ recommend %s (fulcrum %.2f) → next: %s\033[0m'%(d['recommended'],d['fulcrum'],', '.join(o['name'] for o in d['options'][:3])))"
sleep 1

say "3) Quick utilities, no LLM needed (deterministic organ routing)"
chat "convert 10 km to miles"
chat "convert 250 USD to EUR"

say "4) Web research"
chat "search the web for python asyncio best practices"

say "5) A sensitive action — human-in-the-loop approval gate"
chat "send an email to coach@club.com about Saturday's lineup"
prism "(PRISM pauses for explicit approval before any irreversible action)"
sleep 0.6

say "6) Get things done — tasks"
chat "add a task to review the scouting report"
chat "list my tasks"

say "7) Borrow a capability — import an Organ Pack, then use it instantly"
python3 - "$HOME_DIR" <<'PY'
import json,sys,prism_organ_pack as p
code='ORGAN_META={"intent":"shout","description":"uppercase","version":"1.0","capabilities":[]}\nORGAN_POLICY={"risk_level":"low","requires_approval":False,"irreversible":False,"max_per_session":None}\ndef execute(intent,message,ctx):\n    from prism_responses import text_card\n    return text_card(message.upper(),"Shout")\n'
o={"intent":"shout","description":"uppercase","version":"1.0","capabilities":[],"code":code,"sha256":p._sha256(code)}
pk={"format":p.PACK_FORMAT,"name":"shout-pack","author":"demo","organs":[o]};pk["sha256"]=p._pack_digest(pk["organs"])
json.dump(pk,open(sys.argv[1]+"/pack.json","w"))
PY
you "import this shout-pack a friend shared"
printf "${C_D}prism ◂${Z} ${C_P}%s${Z}\n" "$(curl -s "${A[@]}" -d @"${HOME_DIR}/pack.json" "$B/organs/pack/import" | python3 -c 'import sys,json;d=json.load(sys.stdin);print("installed:",d.get("installed"))')"
you "now shout: ship it"
printf "${C_D}prism ◂${Z} ${C_P}%s${Z}\n" "$(curl -s "${A[@]}" -d '{"intents":["shout"],"message":"ship it"}' "$B/organs/execute" | python3 -c 'import sys,json;print(json.load(sys.stdin)["outputs"]["shout"]["body"])')"
sleep 1

say "8) Self-awareness — crystallisation phase + durability health"
printf "${C_D}prism ◂${Z} ${C_P}phase=%s${Z}\n" "$(curl -s "${A[@]}" $B/status | python3 -c 'import sys,json;print(json.load(sys.stdin)["phase"])')"
printf "${C_D}prism ◂${Z} ${C_P}%s${Z}\n" "$(curl -s "${A[@]}" $B/metrics | python3 -c 'import sys,json;d=json.load(sys.stdin);print("durability: wal_replays=%s drift=%s critical=%s"%(d["layer1_counters"].get("wal_replays"),d["layer3_dm_trend"],d["layer3_critical"]))')"

printf "\n${C_G}✔ decide · triage · convert · research · gate · do · borrow · know — one session, all on-device.${Z}\n\n"
sleep 1.6
