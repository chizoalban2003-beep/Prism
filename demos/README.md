# PRISM demos

End-to-end walkthroughs of the running PRISM daemon, driven entirely over the
local HTTP API on `127.0.0.1`.

## A multitask morning ("day in the life")

A single user session that decides, triages, converts, researches, hits an
approval gate, manages tasks, borrows a capability via an Organ Pack, and
checks its own crystallisation phase — all on-device.

![PRISM multitask session](multitask_demo.svg)

Run it: `PRISM_PORT=8742 PRISM_HOME="$HOME" bash demos/multitask_demo.sh`

## Feature walkthrough

![PRISM walkthrough](demo.svg)

> If the animation doesn't autoplay in your viewer, open
> [`demo.svg`](demo.svg) directly (it's a self-contained animated SVG), or run
> the script yourself (below).

## What it shows

1. **Liveness + bearer auth** — `/_health` is open; everything else is `401` without a token, `200` with.
2. **Physics decision engine** — `/predict/match` returns an interpretable prediction (fulcrum, distribution, named factors).
3. **Same engine, new domain** — `/domain/evaluate?domain=Medical` returns a triage spectrum.
4. **NL chat → organ routing** — "convert 10 kg to pounds" routes to the unit-convert organ.
5. **SSRF protection** — `web_scrape` of a cloud-metadata IP is refused.
6. **Capability sharing** — build → import → run an **Organ Pack** (hash-verified, AST-checked) with no restart.
7. **Human-in-the-loop** — a sensitive action ("send email") returns an approval card.
8. **Self-awareness** — crystallisation `phase` + durability `metrics`.

## Run it yourself

```bash
# 1. Start the daemon (writes an auth token under ~/.prism/)
python3 prism_daemon.py            # binds 127.0.0.1:8742

# 2. In another shell, point the script at it and run
PRISM_PORT=8742 PRISM_HOME="$HOME" bash demos/demo.sh
```

## Re-record the SVG

```bash
pip install termtosvg
termtosvg demos/demo.svg -g 100x46 -t window_frame_js -M 2200 \
  -c "env PRISM_PORT=8742 PRISM_HOME=$HOME bash demos/demo.sh"
```
