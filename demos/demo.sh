#!/usr/bin/env bash
# PRISM infrastructure walkthrough — drives the live local daemon over HTTP.
# Recorded to an animated SVG with termtosvg. Set PRISM_PORT / PRISM_HOME first.
set -u

PORT="${PRISM_PORT:-8744}"
HOME_DIR="${PRISM_HOME:-/tmp/prismdemo}"
BASE="http://127.0.0.1:${PORT}"
TOKEN="$(cat "${HOME_DIR}/.prism/auth_token")"
AUTH=(-H "Authorization: Bearer ${TOKEN}" -H "Content-Type: application/json")

C_T='\033[1;36m'   # cyan title
C_S='\033[1;33m'   # yellow step
C_G='\033[1;32m'   # green
C_D='\033[0;90m'   # dim
C_R='\033[1;31m'   # red
Z='\033[0m'

pp() { python3 -m json.tool 2>/dev/null || cat; }
step() { printf "\n${C_S}▶ %s${Z}\n" "$1"; sleep 0.6; }
cmd()  { printf "${C_D}\$ %s${Z}\n" "$1"; sleep 0.4; }
banner() { printf "\n${C_T}══════════════════════════════════════════════════════════════${Z}\n${C_T}  %s${Z}\n${C_T}══════════════════════════════════════════════════════════════${Z}\n" "$1"; sleep 0.8; }

banner "PRISM — local decision-intelligence daemon (127.0.0.1:${PORT})"

step "1. Liveness + bearer auth (401 without a token, 200 with)"
cmd  "curl /_health   &&   curl /status  (no token -> 401)"
printf "health: "; curl -s -o /dev/null -w "%{http_code}\n" "${BASE}/_health"
printf "status (no token): "; curl -s -o /dev/null -w "%{http_code}\n" "${BASE}/status"
printf "status (token):    "; curl -s -o /dev/null -w "%{http_code}\n" "${AUTH[@]}" "${BASE}/status"
sleep 1

step "2. Physics decision engine — interpretable match prediction"
cmd  "curl '/predict/match?home=Arsenal&away=Chelsea'"
curl -s "${AUTH[@]}" "${BASE}/predict/match?home=Arsenal&away=Chelsea&sport=football" | pp
sleep 1.2

step "3. Same engine, different domain — medical triage spectrum"
cmd  "curl '/domain/evaluate?domain=Medical'"
curl -s "${AUTH[@]}" "${BASE}/domain/evaluate?domain=Medical" | pp
sleep 1.2

step "4. Natural-language chat -> organ routing (unit conversion)"
cmd  "POST /chat  {'message':'convert 10 kg to pounds'}"
curl -s "${AUTH[@]}" -d '{"message":"convert 10 kg to pounds"}' "${BASE}/chat" | pp
sleep 1

step "5. Security: SSRF guard blocks internal targets"
cmd  "organs/execute web_scrape http://169.254.169.254/  (cloud metadata)"
curl -s "${AUTH[@]}" -d '{"intents":["web_scrape"],"message":"scrape http://169.254.169.254/latest/meta-data/"}' "${BASE}/organs/execute" | pp
sleep 1.2

step "6. Capability sharing: export -> import -> run an Organ Pack"
cmd  "build a 'shout' organ pack, import it, run it"
python3 - "$HOME_DIR" <<'PY'
import json, sys, prism_organ_pack as p
code='''ORGAN_META={"intent":"shout","description":"uppercase","version":"1.0","capabilities":[]}
ORGAN_POLICY={"risk_level":"low","requires_approval":False,"irreversible":False,"max_per_session":None}
def execute(intent,message,ctx):
    from prism_responses import text_card
    return text_card(message.upper(),"Shout")
'''
o={"intent":"shout","description":"uppercase","version":"1.0","capabilities":[],"code":code,"sha256":p._sha256(code)}
pk={"format":p.PACK_FORMAT,"name":"shouter","author":"demo","organs":[o]}; pk["sha256"]=p._pack_digest(pk["organs"])
json.dump(pk, open(sys.argv[1]+"/shout.json","w"))
PY
printf "${C_D}import:${Z} "; curl -s "${AUTH[@]}" -d @"${HOME_DIR}/shout.json" "${BASE}/organs/pack/import" | pp
printf "${C_D}run:${Z}    "; curl -s "${AUTH[@]}" -d '{"intents":["shout"],"message":"prism shares capabilities"}' "${BASE}/organs/execute" \
  | python3 -c "import sys,json;print(json.load(sys.stdin)['outputs']['shout']['body'])"
sleep 1.2

step "7. Human-in-the-loop: approval gate on a sensitive action"
cmd  "POST /chat  {'message':'send an email to alice@example.com'}"
curl -s "${AUTH[@]}" -d '{"message":"send an email to alice@example.com saying hi"}' "${BASE}/chat" \
  | python3 -c "import sys,json;d=json.load(sys.stdin);print('card:',d['type'],'-',d['title'])"
sleep 1

step "8. Self-awareness: crystallisation phase + durability metrics"
cmd  "curl /status   &&   curl /metrics"
printf "${C_D}phase:${Z}   "; curl -s "${AUTH[@]}" "${BASE}/status" | python3 -c "import sys,json;print(json.load(sys.stdin)['phase'])"
printf "${C_D}metrics:${Z} "; curl -s "${AUTH[@]}" "${BASE}/metrics" | python3 -c "import sys,json;d=json.load(sys.stdin);print('wal_replays=%s  dm_trend=%s  critical=%s'%(d['layer1_counters'].get('wal_replays'),d['layer3_dm_trend'],d['layer3_critical']))"

printf "\n${C_G}✔ Walkthrough complete — decide · route · execute · share · stay safe, all local.${Z}\n\n"
sleep 1.5
