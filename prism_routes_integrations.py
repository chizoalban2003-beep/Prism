"""
prism_routes_integrations.py
============================
FastAPI router for email, calendar, browser, instructions, and discovery endpoints.

Routes:
  GET  /email/status
  GET  /email/inbox
  GET  /email/unread
  POST /email/send
  GET  /calendar/status
  GET  /calendar/today
  GET  /browser/status
  GET  /instructions
  POST /instructions
  GET  /discovery/services
  POST /discovery/build
"""
from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from prism_state import _get_agent

router = APIRouter()




# ---------------------------------------------------------------------------
# /email
# ---------------------------------------------------------------------------

@router.get("/email/status")
async def email_status():
    try:
        from prism_email import PrismEmail
    except ImportError as exc:
        return JSONResponse({"error": str(exc), "status": 503}, status_code=503)
    agent = _get_agent()
    em    = getattr(agent, "_email", None) if agent else None
    if em is None:
        em = PrismEmail()
    return em.status_summary()


@router.get("/email/inbox")
async def email_inbox(n: int = 20, folder: str = "INBOX", unread: str = "true"):
    try:
        from prism_email import PrismEmail
    except ImportError as exc:
        return JSONResponse({"error": str(exc), "status": 503}, status_code=503)
    agent = _get_agent()
    em    = getattr(agent, "_email", None) if agent else None
    if em is None:
        em = PrismEmail()
    if not em.configured:
        return JSONResponse(
            {"error": "Email not configured", "status": 503}, status_code=503
        )
    unread_only = unread.lower() != "false"
    msgs = em.fetch_unread(folder=folder, n=n) if unread_only else em.fetch_recent(n=n)
    return {
        "count": len(msgs),
        "messages": [
            {
                "msg_id":  m.msg_id,
                "subject": m.subject,
                "sender":  m.sender,
                "date":    m.date,
                "unread":  m.unread,
                "snippet": m.body[:200],
            }
            for m in msgs
        ],
    }


@router.get("/email/unread")
async def email_unread():
    agent = _get_agent()
    if agent and hasattr(agent, "_email") and agent._email.configured:
        msgs = agent._email.fetch_unread(n=10)
        return {
            "count": len(msgs),
            "messages": [
                {"from": m.sender, "subject": m.subject, "date": m.date, "body": m.body[:500]}
                for m in msgs
            ],
        }
    return JSONResponse(
        {"error": "Email not configured", "status": 503}, status_code=503
    )


@router.post("/email/send")
async def email_send(request: Request):
    body: Dict[str, Any] = {}
    try:
        body = await request.json()
    except Exception:
        pass

    try:
        from prism_email import PrismEmail
    except ImportError as exc:
        return JSONResponse({"error": str(exc), "status": 503}, status_code=503)

    agent = _get_agent()
    em    = getattr(agent, "_email", None) if agent else None
    if em is None:
        em = PrismEmail()
    if not em.configured:
        return JSONResponse(
            {"error": "Email not configured", "status": 503}, status_code=503
        )
    to      = body.get("to", "")
    subject = body.get("subject", "")
    text    = body.get("body", "")
    if not to or not subject or not text:
        return JSONResponse(
            {"error": "'to', 'subject' and 'body' fields required", "status": 400},
            status_code=400,
        )
    ok = em.send(to=to, subject=subject, body=text, reply_to=body.get("reply_to", ""))
    return {"ok": ok}


# ---------------------------------------------------------------------------
# /calendar
# ---------------------------------------------------------------------------

@router.get("/calendar/status")
async def calendar_status():
    agent = _get_agent()
    if agent and hasattr(agent, "_calendar"):
        return agent._calendar.status_summary()
    return {"configured": False}


@router.get("/calendar/today")
async def calendar_today():
    agent = _get_agent()
    if agent and hasattr(agent, "_calendar") and agent._calendar.configured:
        events = agent._calendar.today()
        return {
            "count": len(events),
            "events": [
                {
                    "title":    e.title,
                    "start":    e.start.isoformat(),
                    "end":      e.end.isoformat(),
                    "location": e.location,
                }
                for e in events
            ],
        }
    return JSONResponse(
        {"error": "Calendar not configured", "status": 503}, status_code=503
    )


# ---------------------------------------------------------------------------
# /browser
# ---------------------------------------------------------------------------

@router.get("/browser/status")
async def browser_status():
    agent = _get_agent()
    if agent and hasattr(agent, "_browser"):
        return agent._browser.status()
    return {"available": False}


# ---------------------------------------------------------------------------
# /instructions
# ---------------------------------------------------------------------------

@router.get("/instructions")
async def instructions_get():
    agent = _get_agent()
    if agent and hasattr(agent, "_instructions"):
        instrs = agent._instructions.all_active()
        return {
            "count": len(instrs),
            "instructions": [
                {
                    "id":        i.instr_id,
                    "text":      i.text,
                    "trigger":   i.trigger,
                    "use_count": i.use_count,
                }
                for i in instrs
            ],
        }
    return {"count": 0, "instructions": []}


@router.post("/instructions")
async def instructions_post(request: Request):
    body: Dict[str, Any] = {}
    try:
        body = await request.json()
    except Exception:
        pass

    agent = _get_agent()
    if agent and hasattr(agent, "_instructions"):
        instr = agent._instructions.add(body.get("text", ""), body.get("trigger", "always"))
        return {"id": instr.instr_id, "text": instr.text}
    return JSONResponse(
        {"error": "Instructions not initialised", "status": 503}, status_code=503
    )


# ---------------------------------------------------------------------------
# /discovery
# ---------------------------------------------------------------------------

@router.get("/discovery/services")
async def discovery_services():
    agent = _get_agent()
    if agent and hasattr(agent, "_discovery"):
        services = agent._discovery.list_all()
        return {
            "count": len(services),
            "services": [
                {
                    "name":       s.name,
                    "category":   s.category,
                    "method":     s.access_method,
                    "configured": s.configured,
                }
                for s in services
            ],
        }
    return {"count": 0, "services": []}


@router.post("/discovery/build")
async def discovery_build(request: Request):
    body: Dict[str, Any] = {}
    try:
        body = await request.json()
    except Exception:
        pass

    agent = _get_agent()
    if agent and hasattr(agent, "_discovery"):
        service = agent._discovery.get(body.get("service_id", ""))
        if service:
            ok  = agent._discovery.build_integration(service, body.get("answers", {}))
            msg = agent._discovery.confirmation_message(service)
            return {"success": ok, "message": msg}
        else:
            return JSONResponse(
                {"error": "Service not found", "status": 404}, status_code=404
            )
    return JSONResponse(
        {"error": "Discovery not initialised", "status": 503}, status_code=503
    )
