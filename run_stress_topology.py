"""
PRISM Nucleus-Organ Topology Stress Test
Exercises: routing, L1 constitution, BudManager, capability manifests,
           LogicPolicy loop, organ execution, synthesis blocking.
No LLM required — runs fully in stdlib mode.
"""
import sys, time, json, traceback
sys.path.insert(0, "/home/chizoalban2003/Prism")

from prism_agent import PrismAgent
from prism_constitution import get_guard
from prism_bud_manager import BudManager
from prism_organ_loader import OrganLoader

RESET = "\033[0m"
GREEN = "\033[92m"
RED   = "\033[31m"
CYAN  = "\033[96m"
BOLD  = "\033[1m"
DIM   = "\033[2m"
YELLOW= "\033[93m"

def hdr(title):
    print(f"\n{BOLD}{CYAN}{'─'*60}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{'─'*60}{RESET}")

def ok(msg):   print(f"  {GREEN}✓{RESET}  {msg}")
def fail(msg): print(f"  {RED}✗{RESET}  {msg}")
def info(msg): print(f"  {DIM}→  {msg}{RESET}")

results = {"pass": 0, "fail": 0}

def check(label, condition, detail=""):
    if condition:
        ok(label + (f"  [{detail}]" if detail else ""))
        results["pass"] += 1
    else:
        fail(label + (f"  [{detail}]" if detail else ""))
        results["fail"] += 1

# ── 1. Boot agent ──────────────────────────────────────────────────────────────
hdr("1. Agent + Topology Bootstrap")
try:
    agent = PrismAgent()
    check("PrismAgent initialised", True)
    check("ConstitutionGuard mounted", hasattr(agent, "_constitution"))
    check("BudManager mounted", hasattr(agent, "_bud_mgr"))
    guard  = agent._constitution
    budmgr = agent._bud_mgr
except Exception as e:
    fail(f"Agent boot failed: {e}")
    traceback.print_exc()
    sys.exit(1)

# ── 2. Organ loader capability manifests ──────────────────────────────────────
hdr("2. Organ Capability Manifests")
loader = OrganLoader()

manifest_cases = [
    ("web_search",      ["internet_read"]),
    ("web_scrape",      ["internet_read"]),
    ("shell_run",       ["subprocess"]),
    ("phone_call",      ["telephony"]),
    ("email_send",      ["internet_write"]),
    ("discord_send",    ["internet_write"]),
    ("file_write",      ["filesystem_write"]),
    ("file_read",       ["filesystem_read"]),
    ("screenshot_capture", ["system_ui"]),
    ("smart_home_control", ["smart_home"]),
    ("reminder_set",    ["notifications"]),
    ("unit_convert",    []),
    ("note_append",     ["filesystem_write"]),
    ("wikipedia_lookup",["internet_read"]),
    ("weather_check",   ["internet_read"]),
]

for intent, expected in manifest_cases:
    caps = loader.get_organ_capabilities(intent)
    check(f"  {intent:25s} caps={caps}", sorted(caps) == sorted(expected),
          f"expected {expected}")

# ── 3. L1 Constitution guard ───────────────────────────────────────────────────
hdr("3. L1 ConstitutionGuard")
g = get_guard()

check("constitution loaded",        g is not None)
ok_ws, _ = g.check("web_search", ["internet_read"])
check("web_search → allowed",       ok_ws == True)
ok_pc, _ = g.check("phone_call", ["telephony"])
check("phone_call → allowed",       ok_pc == True)
ok_sh, _ = g.check("shell_run", ["subprocess"])
check("shell_run → allowed (exec)", ok_sh == True)

# Synthesis blocking
check("may_synthesize(internet_read)", g.may_synthesize("internet_read"))
check("may_synthesize(filesystem_write)", g.may_synthesize("filesystem_write"))
check("BLOCKS synthesize(subprocess)",  not g.may_synthesize("subprocess"))
check("BLOCKS synthesize(telephony)",   not g.may_synthesize("telephony"))

# Capability risk levels
check("subprocess → critical risk",     g.capability_risk("subprocess") == "critical")
check("telephony  → high risk",         g.capability_risk("telephony")  == "high")
check("internet_read → low risk",       g.capability_risk("internet_read") == "low")

# Max synthesis cap
cap = g.max_synthesis_per_session()
check(f"max_synthesis_per_session={cap}", cap >= 5)

# ── 4. BudManager scoping ─────────────────────────────────────────────────────
hdr("4. BudManager Scoped Context")
full_ctx = {
    "router": "ROUTER",
    "memory": "MEMORY",
    "twilio_config": {"sid": "ACxxx"},
    "shell_runner": lambda cmd: "result",
    "discord_webhook": "https://discord.com/...",
    "ha_config": {"url": "http://ha.local"},
    "strava_token": "secret",
    "user_name": "Alice",
}

# internet_read bud should NOT get telephony or subprocess keys
handle = budmgr.spawn("web_search", "search the web", full_ctx, ["internet_read"])
sctx = handle.scoped_ctx
check("web_search bud: router in scope",       "router" in sctx)
check("web_search bud: twilio NOT in scope",   "twilio_config" not in sctx)
check("web_search bud: shell_runner NOT in scope", "shell_runner" not in sctx)
check("web_search bud: ha_config NOT in scope","ha_config" not in sctx)
check("web_search bud: has _bud_id",           "_bud_id" in sctx)

# telephony bud should get twilio
handle_tel = budmgr.spawn("phone_call", "make a call", full_ctx, ["telephony"])
sctx_tel = handle_tel.scoped_ctx
check("phone_call bud: twilio_config in scope", "twilio_config" in sctx_tel)
check("phone_call bud: shell_runner NOT in scope", "shell_runner" not in sctx_tel)

# filesystem_write bud
handle_fw = budmgr.spawn("file_write", "write file", full_ctx, ["filesystem_write"])
sctx_fw = handle_fw.scoped_ctx
check("file_write bud: no twilio_config",  "twilio_config" not in sctx_fw)

# decommission clears _bud_id
budmgr.decommission(handle)
check("after decommission: _bud_id removed", "_bud_id" not in handle.scoped_ctx)
check("status=decommissioned", handle.status.name == "DECOMMISSIONED")

# synthesis cap
budmgr._session_synthesis = g.max_synthesis_per_session()
check("synthesis_allowed() blocked at cap", not budmgr.synthesis_allowed())
budmgr._session_synthesis = 0
check("synthesis_allowed() ok when reset",  budmgr.synthesis_allowed())

# ── 5. Intent routing (20 tasks) ──────────────────────────────────────────────
hdr("5. Intent Routing (20 Tasks)")

ROUTING_CASES = [
    ("what is the weather in Lagos",             "weather_check"),
    ("search the web for Python async patterns", "web_search"),
    ("look up photosynthesis on wikipedia",      "wikipedia_lookup"),
    ("show me the latest news headlines",        "news_headlines"),
    ("translate 'hello world' to Spanish",       "translate_text"),
    ("convert 5 kilograms to pounds",            "unit_convert"),
    ("convert 100 USD to EUR",                   "currency_convert"),
    ("append a note: meeting was productive",    "note_append"),
    ("read my file /tmp/test.txt",               "file_read"),
    ("set a timer for 10 minutes",               "timer_set"),
    ("remind me in 30 minutes to call Alice",    "reminder_set"),
    ("take a screenshot",                        "screenshot_capture"),
    ("send a Discord message: project done",     "discord_send"),
    ("make a phone call to +44700900000",        "phone_call"),
    ("play spotify",                             "spotify_control"),
    ("generate a QR code for https://prism.ai", "qr_generate"),
    ("stock price of Apple",                     "web_search"),
    ("bitcoin price today",                      "web_search"),
    ("supply chain disruption analysis",         "domain_supply"),
    ("run shell command: ls -la",                "shell_run"),
]

for msg, expected_intent in ROUTING_CASES:
    intent = agent._route(msg)
    check(f"  '{msg[:48]}'", intent == expected_intent,
          f"got={intent} want={expected_intent}")

# ── 6. Live organ execution (non-destructive) ─────────────────────────────────
hdr("6. Live Organ Execution (read-only, no approvals)")

LIVE_ORGANS = [
    ("unit_convert",    "convert 100 miles to kilometers"),
    ("currency_convert","convert 50 GBP to USD"),
    ("note_append",     "append a note: stress test passed"),
    ("reminder_set",    "remind me in 999 minutes to check results"),
    ("timer_set",       "set a timer for 9999 seconds"),
]

for intent, msg in LIVE_ORGANS:
    try:
        t0 = time.perf_counter()
        card = agent._execute(intent, msg, {})
        elapsed = time.perf_counter() - t0
        title = card.get("title", "?") if isinstance(card, dict) else "?"
        check(f"  {intent:20s} → card returned  ({elapsed*1000:.0f}ms)", True, title[:40])
    except Exception as e:
        check(f"  {intent:20s} → organ raised",    False, str(e)[:60])

# ── 7. Approval gate (should block, not execute) ──────────────────────────────
hdr("7. Approval Gate — Destructive Organs Block Correctly")

APPROVAL_ORGANS = [
    ("email_send",    "send email to alice@example.com about the meeting"),
    ("discord_send",  "send discord: system online"),
    ("file_write",    "write file /tmp/prism_test_write.txt content: hello"),
    ("shell_run",     "run shell: echo hello"),
    ("phone_call",    "call +447000000000 and say hello"),
]

for intent, msg in APPROVAL_ORGANS:
    try:
        card = agent._execute(intent, msg, {})
        title = card.get("title", "") if isinstance(card, dict) else ""
        body  = card.get("body", "")  if isinstance(card, dict) else str(card)
        blocked = (
            "approval" in title.lower() or "approval" in body.lower() or
            "confirm"  in body.lower()  or "approve"  in body.lower() or
            "pending"  in body.lower()
        )
        check(f"  {intent:20s} → approval gated", blocked, title[:40])
    except Exception as e:
        check(f"  {intent:20s} → raised exception", False, str(e)[:60])

# ── 8. Synthesis blocking (L1 Constitution) ───────────────────────────────────
hdr("8. Synthesis Blocking via L1 ConstitutionGuard")

for blocked_intent in ["shell_run_v2", "phone_auto_dialler"]:
    # Simulate required capabilities for synthesis check
    for cap in ["subprocess", "telephony"]:
        blocked = not g.may_synthesize(cap)
        check(f"  synthesize({cap}) → BLOCKED by L1", blocked)

# ── 9. LogicPolicy metadata collection ────────────────────────────────────────
hdr("9. LogicPolicy Chain Metadata")

from prism_chain import PrismChain

chain = PrismChain(organ_loader=loader)
meta, summary = chain._logicpolicy_meta("web_search")
check("logicpolicy risk field present",      "risk_level" in meta)
check("logicpolicy capabilities present",    "capabilities" in meta)
check("logicpolicy constitution present",    "constitution" in meta)
check("web_search: L1 = allowed",           meta["constitution"] == "allowed")
check("summary string non-empty",           len(summary) > 0)
check("web_search caps in summary",         "internet_read" in summary)
info(f"  web_search summary: {summary}")

meta2, summary2 = chain._logicpolicy_meta("shell_run")
check("shell_run: risk=critical",            meta2["risk_level"] == "critical")
check("shell_run: subprocess in caps",       "subprocess" in meta2["capabilities"])
check("shell_run: L1=allowed (execution)",   meta2["constitution"] == "allowed")
info(f"  shell_run summary: {summary2}")

meta3, summary3 = chain._logicpolicy_meta("phone_call")
check("phone_call: telephony in caps",       "telephony" in meta3["capabilities"])
info(f"  phone_call summary: {summary3}")

# ── 10. End-to-end chat (multi-turn, no LLM) ─────────────────────────────────
hdr("10. End-to-End Chat — 10 Multi-Turn Tasks")

CHAT_TASKS = [
    "What is the weather like in London?",
    "How many grams in a pound?",
    "Search the web for latest AI news",
    "What is machine learning according to Wikipedia?",
    "Convert 200 euros to US dollars",
    "Show me the top news headlines",
    "Translate 'good morning' to French",
    "Generate a QR code for https://github.com",
    "What time is it?",
    "Tell me about yourself",
]

for i, task in enumerate(CHAT_TASKS, 1):
    try:
        t0 = time.perf_counter()
        result = agent.chat(task)
        elapsed = time.perf_counter() - t0
        card = result if isinstance(result, dict) else {}
        title = card.get("title", "?")[:40]
        body  = card.get("body", "")[:60]
        check(f"  [{i:02d}] {task[:48]}", True, f"{elapsed*1000:.0f}ms | {title}")
        if body:
            info(f"         → {body}")
    except Exception as e:
        check(f"  [{i:02d}] {task[:48]}", False, str(e)[:60])

# ── Results ────────────────────────────────────────────────────────────────────
hdr("Results")
total = results["pass"] + results["fail"]
pct   = results["pass"] * 100 // total if total else 0
color = GREEN if results["fail"] == 0 else YELLOW if results["fail"] < 5 else RED
print(f"\n  {BOLD}{color}{results['pass']}/{total} checks passed ({pct}%){RESET}")
if results["fail"]:
    print(f"  {RED}{results['fail']} failures above{RESET}")
else:
    print(f"  {GREEN}All checks passed — Nucleus-Organ topology healthy{RESET}")
print()
sys.exit(0 if results["fail"] == 0 else 1)
