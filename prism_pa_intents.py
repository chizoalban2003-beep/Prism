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


def handle_pa_intent(agent: Any, intent: str, message: str,
                     ctx: dict) -> Optional[PrismCard]:
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
            task = agent._task_mgr.add(
                title    = parsed.get("title", message[:80]),
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

    if intent == "list_tasks":
        tasks = agent._task_mgr.list_tasks(done=False)
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
