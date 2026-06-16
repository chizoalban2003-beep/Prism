# PRISM Quickstart — your first 15 minutes

Goal: clone, boot, do the identity ceremony, and have your first real
conversation. No optional integrations, no config-file editing — those
come after you know it works.

If anything in this guide is wrong on your machine, that's a bug — please
open an issue. The README has the long-form reference; this page is the
shortest viable path.

---

## Before you start (2 min)

You need three things on your PATH:

| Tool | Why | Check |
|---|---|---|
| **Python 3.11+** | Daemon runtime | `python3 --version` |
| **git** | Clone the repo | `git --version` |
| **Ollama** | Local LLM (free, ~4 GB) | `ollama --version` |

If you'd rather use Claude via API key, you can skip Ollama — there's a
detour at the bottom of step 3.

Install Ollama: <https://ollama.ai> (one click on macOS/Windows; one
curl on Linux).

---

## Step 1 — Install PRISM (3 min)

```bash
git clone https://github.com/chizoalban2003-beep/Prism.git
cd Prism
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[full]"
```

The `[full]` extra pulls in voice, browser automation, and the optional
integrations. If you're on a slow link, `pip install -e .` works too —
you can add extras later.

---

## Step 2 — Pick an LLM (2 min)

**Option A — Ollama (recommended, free, fully local):**

```bash
ollama serve &                     # start the engine in the background
ollama pull mistral                # ~4 GB download, one time
```

**Option B — Claude API (no local download, costs per call):**

```bash
python3 prism_daemon.py --setup-llm
# Paste your API key when prompted; pick "anthropic" as the provider.
```

You can change providers later from the web UI; this just gets you to a
working chat.

---

## Step 3 — Identity ceremony (3 min)

This is the part that makes your Prism *yours*. Seven conversational
questions seed the soul model — your values, what you defer on, what
"too aggressive" means for you.

```bash
python3 prism_daemon.py --ceremony
```

Answer honestly and briefly. There are no wrong answers — the ceremony
is feeding the crystallisation engine, not grading you. It runs in the
terminal; you can Ctrl-C and resume later with the same command.

When it finishes, the daemon keeps running and the HTTP server comes up
on `http://127.0.0.1:8742`.

---

## Step 4 — Get your auth token (30 sec)

PRISM gates every endpoint except `/_health` behind a bearer token.
First boot creates one for you:

```bash
cat ~/.prism/auth_token
```

Copy it. You'll paste it into the web UI once.

---

## Step 5 — First conversation (3 min)

Open <http://127.0.0.1:8742> in a browser. Paste the auth token into
the prompt at the top of the page.

Try these in order — they exercise three different paths through the
system:

1. **A plain question** — *"What's the weather in Berlin tomorrow?"*
   This routes through `web_search` + `weather_check`. Watch the
   reasoning chain on the right-hand panel.

2. **A standing instruction** — *"Remind me to drink water every two
   hours between 9am and 6pm."* This writes a proactive trigger; you'll
   see the notification fire even after you close the tab.

3. **An action with an approval gate** — *"Create a file at
   ~/Documents/prism-hello.txt that says hello."* The L2 policy on
   `file_write` will ask for confirmation before it touches your
   filesystem. Approve it and check the file landed.

If all three work, you have a working PRISM.

---

## What's next

Once you trust it, layer integrations on one at a time — don't
front-load them all. Each one is a config block in
`~/.prism/prism_config.toml`:

- **Calendar** (CalDAV, iCal, or Google) — `prism_calendar.py` reference
- **Email** (IMAP read + SMTP send) — `prism_email.py` reference
- **Phone/SMS** (Twilio) — README §"Phone calls and SMS"
- **Smart home** (Home Assistant) — README §"Capabilities"
- **Voice in** (Whisper) — `pip install openai-whisper`
- **Browser automation** (Playwright) — `playwright install chromium`

A good rule: add an integration the *first time you wish PRISM could do
that thing*, not preemptively.

---

## When something breaks

| Symptom | Most likely cause | Fix |
|---|---|---|
| `connection refused` on :8742 | Daemon not running | `python3 prism_daemon.py` in a fresh terminal |
| `401 Unauthorized` from the UI | Token mismatch | `cat ~/.prism/auth_token` and re-paste |
| LLM calls hang | Ollama not serving | `ollama serve &` then `ollama list` to confirm `mistral` is there |
| Ceremony loops on the same question | Soul seed write failed | `ls -la ~/.prism/` — fix permissions, rerun `--ceremony` |
| Approval prompt never appears | UI not connected to WS | Reload the page; check browser console for the WS handshake |

For anything else: `tail -f ~/.prism/prism.log` is the source of truth.
The daemon logs every chain step, every policy verdict, and every organ
invocation with timing.

---

## A note on trust

PRISM runs **only on your machine**. The decision history, soul model,
and crystallised persona live in `~/.prism/` and never leave unless you
explicitly federate to another device you own. The auth token is the
key — if it leaks, anyone on your network can act as you through the
daemon. `chmod 600 ~/.prism/auth_token` is set automatically, but if
you're on shared hardware, this is worth knowing.

If you find a security issue, see [SECURITY.md](../SECURITY.md) — please
don't open a public GitHub issue.
