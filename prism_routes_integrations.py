"""
prism_routes_integrations.py
============================
FastAPI router for email, calendar, browser, instructions, discovery, and
messaging gateway endpoints.

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
  GET  /integrations/messaging/status
  POST /integrations/messaging/send
  POST /integrations/messaging/webhook/whatsapp
"""
from __future__ import annotations

from typing import Any

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
    body: dict[str, Any] = {}
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
    body: dict[str, Any] = {}
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
    body: dict[str, Any] = {}
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


# ---------------------------------------------------------------------------
# /integrations/messaging
# ---------------------------------------------------------------------------

@router.get("/integrations/messaging/status")
async def messaging_status():
    """Return status of all registered messaging gateways."""
    try:
        from prism_messaging_gateway import gateway_registry
    except ImportError as exc:
        return JSONResponse({"error": str(exc), "status": 503}, status_code=503)

    return {
        "gateways": [
            {"name": gw.name, "running": gw.running, "platform": gw.name}
            for gw in gateway_registry.values()
        ]
    }


@router.post("/integrations/messaging/send")
async def messaging_send(request: Request):
    """Send a message through a registered gateway.

    Body: {"platform": str, "chat_id": str, "text": str}
    """
    body: dict[str, Any] = {}
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body", "status": 400}, status_code=400)

    try:
        from prism_messaging_gateway import gateway_registry
    except ImportError as exc:
        return JSONResponse({"error": str(exc), "status": 503}, status_code=503)

    platform = body.get("platform", "")
    chat_id  = body.get("chat_id", "")
    text     = body.get("text", "")

    if not platform or not chat_id or not text:
        return JSONResponse(
            {"error": "'platform', 'chat_id' and 'text' are required", "status": 400},
            status_code=400,
        )

    gw = gateway_registry.get(platform)
    if gw is None:
        return JSONResponse(
            {"error": f"Gateway '{platform}' not registered or not running", "status": 404},
            status_code=404,
        )

    try:
        if platform == "telegram":
            tg_app = getattr(gw, "_app", None)
            if tg_app is None:
                return JSONResponse({"error": "Telegram app not initialised"}, status_code=503)
            await tg_app.bot.send_message(chat_id=int(chat_id), text=text)
        elif platform == "whatsapp":
            await gw._send_whatsapp(chat_id, text)
        else:
            return JSONResponse(
                {"error": f"Direct send not implemented for '{platform}'"}, status_code=501
            )
        return {"ok": True}
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)


@router.post("/integrations/messaging/webhook/whatsapp")
async def messaging_webhook_whatsapp(request: Request):
    """Handle an inbound Twilio WhatsApp webhook."""
    try:
        from prism_messaging_gateway import _dispatch, gateway_registry
    except ImportError as exc:
        return JSONResponse({"error": str(exc), "status": 503}, status_code=503)

    try:
        form = await request.form()
        body = dict(form)
    except Exception:
        body = {}

    gw = gateway_registry.get("whatsapp")
    if gw is None:
        # Gateway not running — still try to dispatch so the webhook never 5xx.
        try:
            from prism_messaging_gateway import WhatsAppGateway
            gw = WhatsAppGateway("", "", "")
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=503)

    try:
        envelope = gw.receive(body)
        response = await _dispatch(envelope)
        await envelope.reply_fn(response)
        return {"ok": True}
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)


# ---------------------------------------------------------------------------
# /lora  — QLoRA fine-tuning pipeline
# ---------------------------------------------------------------------------

from typing import Optional as _Optional  # noqa: E402

_lora_trainer: _Optional[Any] = None


def _get_lora_trainer():
    global _lora_trainer
    if _lora_trainer is None:
        try:
            from prism_lora_trainer import PrismLoraTrainer
            _lora_trainer = PrismLoraTrainer()
        except Exception as exc:
            return None, str(exc)
    return _lora_trainer, None


def _job_to_dict(job) -> dict[str, Any]:
    return {
        "job_id":       job.job_id,
        "base_model":   job.base_model,
        "status":       job.status,
        "pairs_used":   job.pairs_used,
        "started_at":   job.started_at,
        "finished_at":  job.finished_at,
        "ollama_model": job.ollama_model,
        "error":        job.error,
    }


@router.post("/lora/train")
async def lora_train(request: Request):
    """
    POST /lora/train
    Body (optional JSON): {"base_model": "llama3.2:3b", "min_pairs": 10}
    Returns {"job_id": ..., "status": "pending"} or {"error": "not_enough_data"}.
    """
    body: dict[str, Any] = {}
    try:
        body = await request.json()
    except Exception:
        pass

    trainer, err = _get_lora_trainer()
    if trainer is None:
        return JSONResponse({"error": f"lora trainer unavailable: {err}"}, status_code=503)

    base_model = body.get("base_model", "llama3.2:3b")
    min_pairs  = int(body.get("min_pairs", 10))

    job_id = trainer.start_training(base_model=base_model, min_pairs=min_pairs)
    if job_id is None:
        return JSONResponse({"error": "not_enough_data"}, status_code=422)

    job = trainer.get_job(job_id)
    return {"job_id": job_id, "status": job.status if job else "pending"}


@router.get("/lora/status")
async def lora_status_all():
    """GET /lora/status — list all training jobs."""
    trainer, err = _get_lora_trainer()
    if trainer is None:
        return JSONResponse({"error": f"lora trainer unavailable: {err}"}, status_code=503)

    return {"jobs": [_job_to_dict(j) for j in trainer.list_jobs()]}


@router.get("/lora/status/{job_id}")
async def lora_status_one(job_id: str):
    """GET /lora/status/{job_id} — single job status."""
    trainer, err = _get_lora_trainer()
    if trainer is None:
        return JSONResponse({"error": f"lora trainer unavailable: {err}"}, status_code=503)

    job = trainer.get_job(job_id)
    if job is None:
        return JSONResponse({"error": "job not found"}, status_code=404)
    return _job_to_dict(job)
