"""
prism_routes_ide.py
===================
FastAPI APIRouter — VS Code / IDE integration bridge.

Routes
------
GET  /ide/status      — health + agent readiness for IDE ping
POST /ide/complete    — code completion
POST /ide/explain     — explain selected code
POST /ide/review      — review/critique a file
POST /ide/fix         — suggest fix for an error
POST /ide/chat        — general chat with code context
GET  /ide/context     — current agent context (active session, phase, user)
"""
from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from prism_state import _get_agent, _state

router = APIRouter()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _router_llm():
    agent = _get_agent()
    return getattr(agent, "_router", None) if agent else None


def _phase_engine():
    agent = _get_agent()
    return getattr(agent, "_phase", None) if agent else None


async def _llm_call(prompt: str, system: str) -> str:
    """Call the agent LLM router; return text or an error string."""
    llm = _router_llm()
    if llm is None:
        return "[PRISM: agent not ready — start prism_daemon first]"
    try:
        result = llm.call(prompt=prompt, system=system)
        # LLMRouter.call() returns a (text, model_name) tuple. Unwrap it so the
        # endpoint returns the answer text rather than a stringified tuple like
        # "('', 'none')" (which is what leaked before this fix).
        if isinstance(result, tuple):
            result = result[0] if result else ""
        if isinstance(result, str):
            return result
        # Some router implementations return a dict with an "answer" key
        if isinstance(result, dict):
            return result.get("answer") or result.get("text") or str(result)
        return str(result)
    except Exception as exc:  # noqa: BLE001
        return f"[PRISM error: {exc}]"


# ---------------------------------------------------------------------------
# GET /ide/status
# ---------------------------------------------------------------------------


@router.get("/ide/status")
async def ide_status():
    """Health check + agent readiness for IDE ping."""
    agent = _get_agent()
    agent_ready = agent is not None
    llm_ready = _router_llm() is not None

    current_phase = "UNKNOWN"
    phase_engine = _phase_engine()
    if phase_engine is not None:
        try:
            current_phase = str(getattr(phase_engine, "current_phase", "UNKNOWN"))
        except Exception:  # noqa: BLE001
            pass

    return {
        "ok": True,
        "agent_ready": agent_ready,
        "llm_ready": llm_ready,
        "phase": current_phase,
        "server": "prism-asgi",
        "timestamp": time.time(),
    }


# ---------------------------------------------------------------------------
# POST /ide/complete
# ---------------------------------------------------------------------------


@router.post("/ide/complete")
async def ide_complete(request: Request):
    """
    Code completion at the given cursor position.

    Body::

        {
            "code":        "...",
            "cursor_line": 10,
            "cursor_col":  4,
            "language":    "python",
            "filename":    "main.py"   # optional
        }
    """
    body: dict[str, Any] = {}
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        pass

    code: str = (body.get("code") or "").strip()
    if not code:
        return JSONResponse({"error": "'code' is required"}, status_code=400)

    language: str = body.get("language") or "unknown"
    filename: str = body.get("filename") or ""
    cursor_line: int = int(body.get("cursor_line") or 0)
    cursor_col: int = int(body.get("cursor_col") or 0)

    file_hint = f" in `{filename}`" if filename else ""
    system = (
        f"You are an expert {language} programmer acting as an inline code-completion assistant. "
        "Return ONLY the completion text — no explanation, no markdown fences, no preamble."
    )
    prompt = (
        f"Complete the following {language} code{file_hint}. "
        f"Cursor is at line {cursor_line}, column {cursor_col}.\n\n"
        f"```{language}\n{code}\n```\n\n"
        "Return only the text to insert at the cursor position."
    )

    completion = await _llm_call(prompt=prompt, system=system)
    return {"completion": completion, "language": language, "filename": filename}


# ---------------------------------------------------------------------------
# POST /ide/explain
# ---------------------------------------------------------------------------


@router.post("/ide/explain")
async def ide_explain(request: Request):
    """
    Explain selected code.

    Body::

        {
            "code":     "...",
            "language": "python",
            "question": "What does this do?"   # optional
        }
    """
    body: dict[str, Any] = {}
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        pass

    code: str = (body.get("code") or "").strip()
    if not code:
        return JSONResponse({"error": "'code' is required"}, status_code=400)

    language: str = body.get("language") or "unknown"
    question: str = (body.get("question") or "Explain this code clearly and concisely.").strip()

    system = (
        f"You are an expert {language} programmer and educator. "
        "Explain code clearly, mentioning what it does, how it works, and any notable patterns or pitfalls."
    )
    prompt = (
        f"```{language}\n{code}\n```\n\n"
        f"{question}"
    )

    explanation = await _llm_call(prompt=prompt, system=system)
    return {"explanation": explanation, "language": language}


# ---------------------------------------------------------------------------
# POST /ide/review
# ---------------------------------------------------------------------------


@router.post("/ide/review")
async def ide_review(request: Request):
    """
    Review / critique a code file.

    Body::

        {
            "code":     "...",
            "language": "python",
            "filename": "app.py"   # optional
        }
    """
    body: dict[str, Any] = {}
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        pass

    code: str = (body.get("code") or "").strip()
    if not code:
        return JSONResponse({"error": "'code' is required"}, status_code=400)

    language: str = body.get("language") or "unknown"
    filename: str = body.get("filename") or ""

    file_hint = f" (`{filename}`)" if filename else ""
    system = (
        f"You are a senior {language} code reviewer. "
        "Provide structured, actionable feedback covering correctness, readability, performance, "
        "and security. Use short sections with clear headings. Be direct and specific."
    )
    prompt = (
        f"Review the following {language} code{file_hint}:\n\n"
        f"```{language}\n{code}\n```\n\n"
        "Provide a structured code review with: Summary, Issues (numbered), and Suggestions."
    )

    review = await _llm_call(prompt=prompt, system=system)
    return {"review": review, "language": language, "filename": filename}


# ---------------------------------------------------------------------------
# POST /ide/fix
# ---------------------------------------------------------------------------


@router.post("/ide/fix")
async def ide_fix(request: Request):
    """
    Suggest a fix for an error.

    Body::

        {
            "code":          "...",
            "error_message": "NameError: name 'x' is not defined",
            "language":      "python"
        }
    """
    body: dict[str, Any] = {}
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        pass

    code: str = (body.get("code") or "").strip()
    if not code:
        return JSONResponse({"error": "'code' is required"}, status_code=400)

    error_message: str = (body.get("error_message") or "").strip()
    if not error_message:
        return JSONResponse({"error": "'error_message' is required"}, status_code=400)

    language: str = body.get("language") or "unknown"

    system = (
        f"You are an expert {language} debugger. "
        "Given code and an error message, provide a corrected version of the code and a brief "
        "explanation of what was wrong and what you changed."
    )
    prompt = (
        f"Fix the following {language} code.\n\n"
        f"Error: {error_message}\n\n"
        f"Code:\n```{language}\n{code}\n```\n\n"
        "Respond with:\n1. The fixed code in a fenced code block.\n2. A brief explanation of the fix."
    )

    fix = await _llm_call(prompt=prompt, system=system)
    return {"fix": fix, "language": language, "error_message": error_message}


# ---------------------------------------------------------------------------
# POST /ide/chat
# ---------------------------------------------------------------------------


@router.post("/ide/chat")
async def ide_chat(request: Request):
    """
    General chat with optional code context.

    Body::

        {
            "message":      "How do I optimise this loop?",
            "code_context": "for i in range(len(items)):...",   # optional
            "filename":     "utils.py"                          # optional
        }
    """
    body: dict[str, Any] = {}
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        pass

    message: str = (body.get("message") or "").strip()
    if not message:
        return JSONResponse({"error": "'message' is required"}, status_code=400)

    code_context: str = (body.get("code_context") or "").strip()
    filename: str = body.get("filename") or ""

    system = (
        "You are PRISM, a personal AI assistant with deep software engineering expertise. "
        "Answer the user's question helpfully and concisely. "
        "If code context is provided, refer to it in your answer."
    )

    if code_context:
        file_hint = f" from `{filename}`" if filename else ""
        prompt = (
            f"Code context{file_hint}:\n```\n{code_context}\n```\n\n"
            f"User question: {message}"
        )
    else:
        prompt = message

    reply = await _llm_call(prompt=prompt, system=system)
    return {"reply": reply, "filename": filename}


# ---------------------------------------------------------------------------
# GET /ide/context
# ---------------------------------------------------------------------------


@router.get("/ide/context")
async def ide_context():
    """Return current agent context: active session, phase, user info."""
    agent = _get_agent()

    active_session = _state.get("active_session_id")

    current_phase = "UNKNOWN"
    phase_engine = _phase_engine()
    if phase_engine is not None:
        try:
            current_phase = str(getattr(phase_engine, "current_phase", "UNKNOWN"))
        except Exception:  # noqa: BLE001
            pass

    user_info: dict[str, Any] = {}
    if agent is not None:
        soul = getattr(agent, "_soul", None)
        if soul is not None:
            try:
                seed = soul.get_seed()
                if seed is not None:
                    user_info = {
                        "values": list(getattr(seed, "stated_values", [])[:3]),
                        "goals": list(getattr(seed, "stated_goals", [])[:2]),
                    }
            except Exception:  # noqa: BLE001
                pass

    return {
        "active_session": active_session,
        "phase": current_phase,
        "agent_ready": agent is not None,
        "llm_ready": _router_llm() is not None,
        "user": user_info,
        "timestamp": time.time(),
    }
