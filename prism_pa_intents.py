"""
prism_pa_intents.py
===================
Personal-assistant integration intent handlers, grouped out of
PrismAgent._execute to keep the agent module focused: smart-home, email,
calendar, browser, standing instructions, service discovery, web search,
push, contacts, tasks, and reminders.

handle_pa_intent(agent, intent, message, ctx) returns a PrismCard for a
handled intent, or None so the caller continues dispatching. Behaviour is
identical to the original inline blocks.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Optional

from prism_responses import PrismCard, setup_required_card, text_card

logger = logging.getLogger(__name__)


def _hardware_status_card(message: str) -> PrismCard:
    """Render a focused hardware/system status card based on which
    component the user actually asked about. Pure observation — no
    approval flow, no LLM call. Bridges user to hardware state.
    """
    import shutil
    msg = (message or "").lower()
    lines: list[str] = []
    section_title = "System status"

    wants_battery = re.search(r"\bbatter(?:y|ies)\b", msg) is not None
    wants_disk    = re.search(r"\b(?:disk|storage|free\s+space)\b", msg) is not None
    wants_mem     = re.search(r"\b(?:memory|ram)\b", msg) is not None
    wants_cpu     = re.search(r"\b(?:cpu|load(?:\s+average)?|system\s+load)\b", msg) is not None
    wants_net     = re.search(r"\b(?:wi-?fi|network|internet)\b", msg) is not None
    wants_uptime  = re.search(r"\buptime\b|how\s+long.*been\s+(?:up|running|on)", msg) is not None
    wants_all     = re.search(r"\b(?:hardware|system)\s+(?:status|stats|health|info)\b", msg) is not None

    if not any((wants_battery, wants_disk, wants_mem, wants_cpu,
                wants_net, wants_uptime)) or wants_all:
        wants_battery = wants_disk = wants_mem = wants_cpu = wants_net = wants_uptime = True

    try:
        import psutil  # type: ignore
    except Exception:
        psutil = None  # type: ignore

    if wants_battery:
        section_title = "Battery" if not wants_all else section_title
        try:
            b = psutil.sensors_battery() if psutil is not None else None
            if b is None:
                lines.append("• Battery: no battery (desktop or VM)")
            else:
                pct = int(b.percent)
                bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
                plugged = "charging" if b.power_plugged else "on battery"
                lines.append(f"• Battery: {pct}% [{bar}] ({plugged})")
        except Exception as exc:
            lines.append(f"• Battery: unavailable ({exc.__class__.__name__})")

    if wants_disk:
        section_title = "Disk" if not wants_all else section_title
        try:
            d = shutil.disk_usage("/")
            free_gb = d.free / (1024 ** 3)
            total_gb = d.total / (1024 ** 3)
            used_pct = int((d.used / d.total) * 100)
            bar = "█" * (used_pct // 5) + "░" * (20 - used_pct // 5)
            lines.append(
                f"• Disk /: {free_gb:.1f} GB free of {total_gb:.1f} GB "
                f"[{bar}] {used_pct}% used"
            )
        except Exception as exc:
            lines.append(f"• Disk: unavailable ({exc.__class__.__name__})")

    if wants_mem:
        section_title = "Memory" if not wants_all else section_title
        try:
            if psutil is None:
                raise RuntimeError("psutil unavailable")
            m = psutil.virtual_memory()
            free_gb = m.available / (1024 ** 3)
            total_gb = m.total / (1024 ** 3)
            # m.percent is a float; str * float raises TypeError (battery
            # and disk cast first — this branch rendered "unavailable").
            bar = "█" * int(m.percent // 5) + "░" * (20 - int(m.percent // 5))
            lines.append(
                f"• Memory: {free_gb:.1f} GB free of {total_gb:.1f} GB "
                f"[{bar}] {int(m.percent)}% used"
            )
        except Exception as exc:
            lines.append(f"• Memory: unavailable ({exc.__class__.__name__})")

    if wants_cpu:
        section_title = "CPU" if not wants_all else section_title
        try:
            if psutil is None:
                raise RuntimeError("psutil unavailable")
            pct = psutil.cpu_percent(interval=0.2)
            cores = psutil.cpu_count(logical=True) or 1
            try:
                import os as _os
                load1, load5, load15 = _os.getloadavg()
                lines.append(
                    f"• CPU: {pct:.0f}% ({cores} cores) — "
                    f"load avg 1/5/15: {load1:.2f} / {load5:.2f} / {load15:.2f}"
                )
            except Exception:
                lines.append(f"• CPU: {pct:.0f}% ({cores} cores)")
        except Exception as exc:
            lines.append(f"• CPU: unavailable ({exc.__class__.__name__})")

    if wants_net:
        section_title = "Network" if not wants_all else section_title
        try:
            if psutil is None:
                raise RuntimeError("psutil unavailable")
            stats = psutil.net_if_stats()
            up_ifs = [name for name, s in stats.items()
                      if s.isup and name != "lo"]
            if up_ifs:
                lines.append(f"• Network: up — interfaces: {', '.join(up_ifs)}")
            else:
                lines.append("• Network: no interfaces up (besides loopback)")
        except Exception as exc:
            lines.append(f"• Network: unavailable ({exc.__class__.__name__})")

    if wants_uptime:
        section_title = "Uptime" if not wants_all else section_title
        try:
            if psutil is None:
                raise RuntimeError("psutil unavailable")
            import time as _time
            secs = int(_time.time() - psutil.boot_time())
            days, rem = divmod(secs, 86400)
            hours, rem = divmod(rem, 3600)
            mins, _ = divmod(rem, 60)
            parts: list[str] = []
            if days:
                parts.append(f"{days}d")
            if hours or days:
                parts.append(f"{hours}h")
            parts.append(f"{mins}m")
            lines.append(f"• Uptime: {' '.join(parts)}")
        except Exception as exc:
            lines.append(f"• Uptime: unavailable ({exc.__class__.__name__})")

    body = "\n".join(lines) if lines else "No readings available."
    return text_card(body, section_title)


def _pipeline_store(agent: Any):
    store = getattr(agent, "_pipelines", None)
    if store is None:
        from prism_pipelines import PipelineStore
        store = PipelineStore()
        agent._pipelines = store
    return store


def handle_pa_intent(agent: Any, intent: str, message: str,
                     ctx: dict) -> Optional[PrismCard]:
    if intent.startswith("pipeline_"):
        from prism_pipelines import human_schedule, parse_save
        store = _pipeline_store(agent)

        if intent == "pipeline_save":
            try:
                name, instruction, secs = parse_save(message)
                pipe = store.save(name, instruction, secs)
            except ValueError as exc:
                return text_card(str(exc), "Pipeline")
            sched = (f" · runs {human_schedule(pipe.schedule_secs)}"
                     if pipe.schedule_secs else " · manual (say \"run "
                     f"pipeline {pipe.name}\")")
            return text_card(
                f"Saved pipeline **{pipe.name}**{sched}.\n\nSteps: "
                f"{pipe.instruction}", "Pipeline saved")

        if intent == "pipeline_list":
            pipes = store.list_all()
            if not pipes:
                return text_card(
                    "No pipelines yet. Save one with:\n"
                    "\"save pipeline morning: check the weather and my "
                    "calendar, then summarise — every day\"", "Pipelines")
            lines = "\n".join(
                f"· **{p.name}** ({human_schedule(p.schedule_secs)}"
                f"{'' if p.enabled else ', disabled'}) — {p.instruction[:80]}"
                for p in pipes)
            return text_card(lines, f"Pipelines ({len(pipes)})")

        if intent == "pipeline_delete":
            m = re.search(r"(?:pipeline|routine|workflow|pipe)\s+(.+)",
                          message, re.IGNORECASE)
            name = (m.group(1).strip() if m else "").rstrip("?.!")
            if name and store.delete(name):
                return text_card(f"Deleted pipeline **{name.lower()}**.",
                                 "Pipeline deleted")
            return text_card(
                f"No pipeline named '{name}'. Say \"list pipelines\" to see "
                "them.", "Pipeline")

        if intent == "pipeline_run":
            m = re.search(r"(?:pipeline|routine|workflow|pipe)\s+(.+)",
                          message, re.IGNORECASE)
            name = (m.group(1).strip() if m else "").rstrip("?.!")
            pipe = store.get(name)
            if pipe is None:
                return text_card(
                    f"No pipeline named '{name}'. Say \"list pipelines\".",
                    "Pipeline")
            card = agent.run_pipeline(pipe.instruction, ctx)
            store.mark_run(pipe.name)
            return card or text_card(
                f"Ran **{pipe.name}** but it produced no output.", "Pipeline")

    if intent == "smart_home":
        if not agent._smarthome.configured:
            return text_card(
                "Smart home not configured. "
                "Add ha_url and ha_token to prism_config.toml.",
                "Smart Home")
        msg_lower = message.lower()
        entity = None
        for word in message.split():
            found = agent._smarthome.find_entity(word)
            if found:
                entity = found
                break
        if "turn on" in msg_lower and entity:
            ok = agent._smarthome.turn_on(entity.entity_id)
            return text_card(
                f"{'Done' if ok else 'Failed'}: {entity.friendly_name} on",
                "Smart Home")
        if "turn off" in msg_lower and entity:
            ok = agent._smarthome.turn_off(entity.entity_id)
            return text_card(
                f"{'Done' if ok else 'Failed'}: {entity.friendly_name} off",
                "Smart Home")
        summary = agent._smarthome.status_summary()
        return text_card(
            f"{summary['on_count']} devices on · "
            f"{summary['total_entities']} total · "
            f"domains: {', '.join(summary.get('domains', [])[:5])}",
            "Smart Home Status")

    if intent == "email_read":
        if not agent._email.configured:
            return setup_required_card(
                service        = "Email",
                why            = (
                    "PRISM needs IMAP credentials to read your inbox. For Gmail use "
                    "an App Password (NOT your normal password) — 2FA must already be on."
                ),
                config_section = "email",
                snippet        = (
                    'provider  = "gmail"\n'
                    'address   = "you@gmail.com"\n'
                    'imap_host = "imap.gmail.com"\n'
                    'imap_port = 993\n'
                    'password  = "xxxx xxxx xxxx xxxx"   # 16-char App Password\n'
                    'max_fetch = 20'
                ),
                steps = [
                    "Open https://myaccount.google.com/apppasswords",
                    "Generate an App Password labeled 'PRISM' and copy it",
                    "Paste it above as password",
                    "Restart PRISM: pkill -f prism_daemon && python3 -m prism_daemon &",
                    "Try 'check my emails' again",
                ],
                docs_url = "https://support.google.com/accounts/answer/185833",
            )
        messages = agent._email.fetch_unread(n=10)
        summary  = agent._email.summarise_inbox(
            messages, llm_router=getattr(agent, '_router', None))
        return text_card(summary, f"Inbox — {len(messages)} unread")

    if intent == "calendar_read":
        if not agent._calendar.configured:
            return setup_required_card(
                service        = "Calendar",
                why            = (
                    "PRISM needs read access to surface today's events, find free "
                    "slots, or schedule new ones. iCal URL is the simplest provider."
                ),
                config_section = "calendar",
                snippet        = (
                    'provider = "ical_url"          # or "google" or "caldav"\n'
                    'ical_url = "webcal://..."      # paste your private iCal feed URL\n'
                    '# google_token = ""            # OAuth2 token  (provider="google")\n'
                    '# caldav_url   = ""            # CalDAV server (provider="caldav")\n'
                    '# username     = ""\n'
                    '# password     = ""'
                ),
                steps = [
                    "Google Calendar → Settings → 'Integrate calendar' → copy the Secret iCal address",
                    "Paste that URL above as ical_url",
                    "Restart PRISM: pkill -f prism_daemon && python3 -m prism_daemon &",
                    "Ask 'what is on my calendar today?' again",
                ],
                docs_url = "https://support.google.com/calendar/answer/37648",
            )
        today    = agent._calendar.today()
        next_ev  = agent._calendar.next_event()
        if not today:
            msg = "Nothing scheduled today."
        else:
            msg = "\n".join(str(e) for e in today)
        if next_ev and next_ev.starts_in_mins <= 30:
            msg = f"⚠ {next_ev.title} starts in {next_ev.starts_in_mins} minutes\n\n" + msg
        return text_card(msg, f"Today — {len(today)} events")

    if intent == "browser_task":
        if not agent._browser.available:
            return text_card(
                "Browser agent not available. "
                "Install with: pip install playwright && playwright install chromium",
                "Browser")
        queue = getattr(agent, '_queue', None)
        if queue:
            def run_browser():
                return agent._browser.execute(message)
            task_id = queue.submit_single(f"Browser: {message[:40]}", run_browser)
            return text_card(
                f"Browser task started. I'll let you know when done.\n"
                f"Task ID: {task_id}",
                "Browser Task")
        else:
            result = agent._browser.execute(message)
            body   = result.extracted[:500] if result.success else result.error
            return text_card(body, "Browser Result")

    if intent == "show_instructions":
        instrs = agent._instructions.all_active()
        if not instrs:
            return text_card("No standing instructions set. "
                             "Tell me to 'always...' or 'never...' "
                             "to set one.", "Standing Instructions")
        lines = "\n".join(f"• [{i.trigger}] {i.text}" for i in instrs)
        return text_card(lines, f"Your instructions ({len(instrs)})")

    if intent == "remove_instruction":
        # Score each instruction by how many significant words from its
        # text appear in the user's message. The previous matcher used
        # `w in message.lower()` — substring containment on the first 3
        # words. A 1-letter token like "i" from "I prefer short emails"
        # matched inside "instruction", so the request "remove the never
        # mind instruction" deleted the wrong instruction. Word-boundary
        # tokenization + stopword filtering picks the named one.
        import re as _re
        _STOPWORDS = {
            "i","a","an","the","that","this","these","those","of","to",
            "in","on","at","for","with","by","my","me","is","are","was",
            "were","be","and","or","not","do","does","did","you","it",
            "as","but","if","so","please","just","ok","okay",
            "instruction","instructions","rule","rules","remove","delete",
            "forget","remember","stop",
        }
        def _tokens(t: str) -> list[str]:
            return [w for w in _re.findall(r"[a-z0-9]+", t.lower())
                    if w not in _STOPWORDS and len(w) > 1]
        msg_tokens = set(_tokens(message))
        instrs = agent._instructions.all_active()
        best, best_score = None, 0
        if instrs and msg_tokens:
            for instr in instrs:
                score = sum(1 for w in _tokens(instr.text) if w in msg_tokens)
                if score > best_score:
                    best, best_score = instr, score
        if best is not None:
            agent._instructions.remove(best.instr_id)
            return text_card(f"Removed: {best.text}",
                             "Instruction removed")
        return text_card("Couldn't find a matching instruction to remove.",
                         "Instructions")

    if intent == "hardware_status":
        return _hardware_status_card(message)

    if intent == "discover_service":
        router = getattr(agent, '_router', None)
        if router:
            name_prompt = (f"Extract the service/app/platform name from: "
                           f"'{message}'. Return ONLY the name, nothing else.")
            service_name, _ = router.call(name_prompt, min_capability=1,
                                          max_tokens=20)
            service_name = service_name.strip().strip('"\'')
        else:
            import re as _re
            words = _re.findall(r'[A-Z][a-zA-Z]+', message)
            service_name = words[0] if words else "unknown service"

        if not service_name:
            service_name = "unknown service"

        if agent._discovery.is_known(service_name):
            existing = agent._discovery.get(service_name)
            if existing and existing.configured:
                return text_card(
                    f"I already have {service_name} connected "
                    f"via {existing.access_method}. "
                    f"What would you like to do with it?",
                    f"{service_name} — already integrated")

        service, questions = agent._discovery.discover(
            service_name = service_name,
            user_intent  = message,
            constraints  = ctx.get("user_constraints", {}),
        )
        steps_text = "\n".join(f"{i+1}. {s}"
                               for i, s in enumerate(service.setup_steps))
        q_text     = "\n".join(f"• {q}" for q in questions[:2])
        body = (
            f"I've researched **{service_name}** — {service.description}\n\n"
            f"Best integration method: **{service.access_method}**\n\n"
            f"To set this up:\n{steps_text}"
            + (f"\n\nI also need a few answers:\n{q_text}" if q_text else "")
        )
        return text_card(body, f"Connecting: {service_name}")

    if intent == "web_search":
        _q = re.sub(
            r'^(?:search(?:\s+the\s+web|\s+online|\s+for)?|look\s+up|'
            r'find\s+(?:out|info|information)\s+(?:about|on)|'
            r'research|who\s+is|where\s+is|when\s+(?:did|does|is)|'
            r'what(?:\'s| is) (?:the )?(?:latest|current|today))[:\s]+',
            '', message, flags=re.IGNORECASE,
        ).strip().rstrip('?.')
        results = agent._search.search(_q or message, n=5)
        if not results:
            answer = agent._search.quick_answer(message)
            if answer:
                return text_card(answer, "Search result")
            # Fall through to web_search organ (DDG Lite)
            organ_fn = agent._organ_loader.get("web_search")
            if organ_fn is not None:
                try:
                    if agent._bud_mgr is not None:
                        caps = agent._organ_loader.get_organ_capabilities("web_search")
                        handle = agent._bud_mgr.spawn("web_search", message, ctx, caps)
                        return agent._bud_mgr.execute(handle, organ_fn)
                    return organ_fn("web_search", message, ctx)
                except Exception:
                    pass
            return text_card("No results found.", "Search")
        router = getattr(agent, '_router', None)
        if router and results:
            context_str = "\n".join(
                f"{r.title}: {r.snippet}" for r in results[:4])
            prompt  = (f"Answer this query using the search results below.\n"
                       f"Query: {message}\nResults:\n{context_str}\n"
                       f"Give a concise factual answer in 2-3 sentences.")
            answer, _ = router.call(
                prompt, min_capability=1, max_tokens=300,
                conversation_history=agent._chat_history[-4:])
            body = answer or "\n".join(
                f"• {r.title}  {r.url}" for r in results[:4])
        else:
            body = "\n".join(
                f"• {r.title}\n  {r.snippet}\n  {r.url}"
                for r in results[:4])
        return text_card(body, f"Search · {agent._search.status_summary()['provider']}")

    if intent == "show_notifications":
        proactive = getattr(agent, "_proactive", None)
        if proactive is None:
            return text_card(
                "Proactive notifications aren't running on this daemon.",
                "Notifications")
        try:
            events = proactive.pending_events(n=10)
        except Exception as exc:
            logger.debug("[show_notifications] read failed: %s", exc)
            return text_card("Couldn't read notifications right now.",
                             "Notifications")
        if not events:
            return text_card(
                "No new notifications. I'll surface anything that needs "
                "your attention here.", "Notifications")
        import time as _time
        def _fmt(ev):
            secs = max(0, int(_time.time() - ev.timestamp))
            if secs < 60:
                rel = f"{secs}s ago"
            elif secs < 3600:
                rel = f"{secs // 60}m ago"
            elif secs < 86400:
                rel = f"{secs // 3600}h ago"
            else:
                rel = f"{secs // 86400}d ago"
            return f"• {ev.message}  ({rel})"
        body = "\n".join(_fmt(e) for e in events)
        return text_card(body, f"Notifications ({len(events)})")

    if intent == "send_push":
        if not agent._push.configured:
            return text_card(
                "Push not configured. Add topic to prism_config.toml [push]. "
                "Get the free ntfy app at ntfy.sh — no account needed.",
                "Push notifications")
        agent._push.alert(message)
        return text_card("Notification sent to your device.", "Push")

    if intent == "contacts":
        query = message.lower().replace("find","").replace(
            "contact","").replace("who is","").strip()
        contacts = agent._contacts.search(query)
        if not contacts:
            return text_card(f"No contact found for '{query}'.", "Contacts")
        c = contacts[0]
        clines: list[str] = [f"{c.name}"]
        if c.organisation:
            clines.append(f"  {c.role} at {c.organisation}")
        if c.emails:
            clines.append(f"  Email: {', '.join(c.emails)}")
        if c.phones:
            clines.append(f"  Phone: {', '.join(c.phones)}")
        if c.notes:
            clines.append(f"  Notes: {c.notes[:200]}")
        return text_card("\n".join(clines),
                          f"Contact · {c.source}")

    if intent == "add_task":
        router = getattr(agent, '_router', None)
        parsed = None
        if router:
            prompt = (f"Extract task details from: '{message}'. "
                      f"Return JSON: {{\"title\":\"...\",\"notes\":\"...\","
                      f"\"due_date\":\"YYYY-MM-DD or empty\","
                      f"\"priority\":1}}")
            raw, _ = router.call(prompt, min_capability=1, max_tokens=200,
                                  json_mode=True)
            try:
                import json as _j
                clean = raw.strip().lstrip("```json").rstrip("```").strip()
                parsed = _j.loads(clean)
            except Exception:
                pass
        if parsed:
            llm_title = (parsed.get("title") or "").strip()
            task = agent._task_mgr.add(
                title    = llm_title or message[:80],
                notes    = parsed.get("notes",""),
                due_date = parsed.get("due_date",""),
                priority = parsed.get("priority",1),
            )
            return text_card(
                f"Added: {task.title}"
                + (f"  Due: {task.due_date}" if task.due_date else ""),
                f"Task added · {task.source}")
        task = agent._task_mgr.add(title=message[:80])
        return text_card(f"Added: {task.title}", "Task added")

    if intent == "complete_task":
        open_tasks = [t for t in agent._task_mgr.list_tasks(done=False)
                      if (t.title or "").strip()]
        if not open_tasks:
            return text_card("No open tasks.", "Tasks")
        # Strip command words; what's left names the task.
        frag = re.sub(
            r"(?i)\b(?:complete|finish|close|remove|delete|drop|did|done"
            r"|with|mark|as|completed?|finished|task|todo|the|my|a)\b|[:—-]",
            " ", message)
        frag = " ".join(frag.split()).lower()
        if not frag:
            return text_card(
                "Which task? Say e.g. 'complete task buy milk'.", "Tasks")
        matches = [t for t in open_tasks if frag in (t.title or "").lower()]
        if not matches:
            # Token overlap fallback — "done with the milk one" still hits
            # "buy milk" if the overlap is unambiguous.
            ftok = set(frag.split())
            scored = sorted(
                ((len(ftok & set((t.title or "").lower().split())), t)
                 for t in open_tasks),
                key=lambda x: x[0], reverse=True)
            scored = [s for s in scored if s[0] > 0]
            if len(scored) == 1 or (len(scored) > 1 and scored[0][0] > scored[1][0]):
                matches = [scored[0][1]]
            else:
                matches = [t for _, t in scored]
        if len(matches) == 1:
            t = matches[0]
            agent._task_mgr.complete(t.task_id)
            return text_card(f"Done: {t.title} ✓",
                             f"Task completed · {t.source}")
        if matches:
            lines = "\n".join(f"· {t.title}" for t in matches[:8])
            return text_card(
                f"Several tasks match '{frag}':\n{lines}\n\n"
                "Say more of the title to pick one.", "Tasks")
        return text_card(f"No open task matching '{frag}'.", "Tasks")

    if intent == "list_tasks":
        tasks = [t for t in agent._task_mgr.list_tasks(done=False)
                 if (t.title or "").strip()]
        if not tasks:
            return text_card("No open tasks.", "Tasks")
        provider = agent._task_mgr._resolve_provider()
        lines = "\n".join(
            f"{'⚡' if t.priority>=3 else '·'} {t.title}"
            + (f"  (due {t.due_date})" if t.due_date else "")
            for t in tasks[:15])
        return text_card(lines, f"Tasks ({len(tasks)}) · {provider}")

    if intent == "reminder":
        router = getattr(agent, '_router', None)
        parsed_time = None
        if router:
            prompt = (f"Extract reminder details from: '{message}'. "
                      f"Return JSON: {{\"message\":\"...\","
                      f"\"seconds_from_now\": <integer seconds or null>,"
                      f"\"iso_datetime\": \"YYYY-MM-DDTHH:MM or null\"}}")
            raw, _ = router.call(prompt, min_capability=1, max_tokens=150, json_mode=True)
            try:
                import json as _j
                clean = raw.strip().lstrip("```json").rstrip("```").strip()
                parsed_time = _j.loads(clean)
            except Exception:
                pass
        if parsed_time and agent._proactive:
            msg = parsed_time.get("message", message)
            secs = parsed_time.get("seconds_from_now")
            iso  = parsed_time.get("iso_datetime")
            if secs:
                agent._proactive.schedule_in(msg, float(secs))
                mins = int(float(secs) // 60)
                return text_card(f"Reminder set: '{msg}' in {mins} minutes.", "Reminder")
            elif iso:
                from datetime import datetime as _dt
                try:
                    fire_at = _dt.fromisoformat(iso).timestamp()
                    agent._proactive.schedule(msg, fire_at)
                    return text_card(f"Reminder set: '{msg}' at {iso}.", "Reminder")
                except Exception:
                    pass
        return text_card("Could not parse reminder time. Try: 'remind me in 30 minutes to call Alice'.", "Reminder")

    return None
