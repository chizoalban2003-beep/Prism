# PRISM Architecture

## Three entry points
- `ksa_cli.py` → developer agent (KSA)
- `kde_cli.py` → sports platform + REST API
- `prism_chat.py` served at localhost:8742 → personal assistant chat

## Capability layers (prism_* modules)
- **Decision**: decision_spectrum.py — Gaussian kernel, AdaptiveFulcrum
- **Planning**: prism_planner.py — LLM extracts task structure, engine ranks strategies
- **Memory**: prism_memory.py — semantic search, BM25 fallback, SQLite storage
- **Identity**: digital_identity.py, identity_bus.py, artifact_store.py
- **Perception**: prism_perception.py — system, typing, biometric, voice, screen channels
- **Execution**: prism_device_agent.py, prism_executor_agent.py, prism_browser_agent.py
- **Communication**: prism_email.py (IMAP/SMTP), prism_calendar.py (CalDAV/iCal)
- **Hardware**: prism_smart_home.py (Home Assistant REST API)
- **Policy**: prism_policy.py — CEO→Manager delegation model
- **Discovery**: prism_service_discovery.py — researches and integrates any unknown service
- **Instructions**: prism_instructions.py — standing rules taught once, applied always
- **LLM routing**: prism_llm_router.py — Claude → Ollama → stdlib fallback chain
- **Output**: prism_tts.py (local TTS), prism_proactive.py (scheduled triggers)
- **Tasks**: prism_task_queue.py — background execution with progress tracking

## What PRISM does not use
No numpy, no torch, no langchain, no OpenAI SDK, no Flask/FastAPI, no cloud services.
All decision math is pure Python arithmetic.
