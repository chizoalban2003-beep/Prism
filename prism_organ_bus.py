"""
prism_organ_bus.py
==================
LLM-mediated publish/subscribe bus connecting PRISM logic engines.

Biological analogy
------------------
PRISM's logic engines (physics engine, policy engine, organ_loader,
calendar, etc.) are like organs in a body.  They do not — and should not
— know each other's internal data formats.  They communicate via a shared
medium that carries meaning across vocabularies.

In a body that medium is hormones in the bloodstream.  In PRISM it is the
LLM.

Flow::

    Physics engine emits:
        OrganSignal(source="physics", signal_type="injury_risk_elevated",
                    payload={"risk": 0.78, "muscle_group": "hamstring"})

    OrganBus routes to subscribers (policy, calendar, horizon):
        LLM translates for policy:  {"adjustment": "reduce_load", "factor": 0.6}
        LLM translates for calendar:{"message": "Consider rest day — hamstring risk 78%"}
        LLM translates for horizon: {"intent": "Monitor hamstring recovery"}

    Each subscriber receives the translation, not the raw payload.
    No organ needs to know any other organ's schema.

Routing logic
-------------
1. **Direct route** (no LLM): signal_type exactly matches a subscription
   AND the payload keys overlap ≥ 50% → pass payload through unchanged.
   Cost: zero.
2. **LLM translation**: vocabularies differ → one LLM call per (signal, subscriber)
   pair.  Result is cached for identical (signal_type, receiver) pairs.
3. **Priority gating**: priority 1 (LOW) signals are batched; priority 2 (NORMAL)
   translate immediately; priority 3 (HIGH) bypass cache and always translate.

Persistence
-----------
Recent signals are stored in ~/.prism/organ_bus.db so you can replay them
for debugging or audit.
"""
from __future__ import annotations

import collections
import json
import logging
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Domain types
# ---------------------------------------------------------------------------

LOW    = 1
NORMAL = 2
HIGH   = 3


@dataclass
class OrganSignal:
    """A signal emitted by one logic engine destined for the bus."""
    source:      str
    signal_type: str
    payload:     dict[str, Any]
    priority:    int   = NORMAL
    timestamp:   float = field(default_factory=time.time)
    signal_id:   str   = field(default_factory=lambda: str(uuid.uuid4())[:8])

    def __post_init__(self):
        if self.priority not in (LOW, NORMAL, HIGH):
            raise ValueError(f"priority must be 1/2/3, got {self.priority!r}")


@dataclass
class OrganSubscription:
    """A subscriber registration — one engine listening for signal types."""
    organ_name:   str
    signal_types: list[str]          # which signal types to receive
    handler:      Callable           # handler(payload: dict) -> None
    vocabulary:   str                # prose: what keys/values this organ understands


@dataclass
class DeliveryRecord:
    """Result of one signal delivery to one subscriber."""
    signal_id:   str
    receiver:    str
    translated:  dict[str, Any]      # what the receiver actually got
    via_llm:     bool                # True = LLM translated; False = direct pass
    duration_ms: float
    success:     bool
    error:       str = ""


# ---------------------------------------------------------------------------
# Translation prompt
# ---------------------------------------------------------------------------

_TRANSLATE_PROMPT = """\
You are translating a signal between two PRISM logic engines.

Emitting engine:  {source}
Signal type:      {signal_type}
Raw payload:      {payload}

Receiving engine: {receiver}
This engine understands: {vocabulary}

Translate the payload into what {receiver} would understand.
Return ONLY valid JSON with at most 6 keys, no extra commentary.
The translated payload should preserve the important facts but use
terminology appropriate for the receiving engine.
"""


# ---------------------------------------------------------------------------
# Anomaly detection
# ---------------------------------------------------------------------------


class SignalAnomalyDetector:
    """
    Sliding-window frequency counter per signal type.
    Fires registered callbacks when a type exceeds baseline × multiplier
    within the window, indicating a compound anomaly (e.g. 3 health_alerts
    in 10 minutes when the normal rate is one per day).
    """

    def __init__(
        self,
        window_seconds: float = 600.0,
        baseline_per_window: float = 1.0,
        spike_multiplier: float = 3.0,
    ) -> None:
        self._window     = window_seconds
        self._baseline   = baseline_per_window
        self._multiplier = spike_multiplier
        self._timestamps: dict[str, collections.deque] = collections.defaultdict(collections.deque)
        self._callbacks:  list[Callable] = []
        self._lock        = threading.Lock()

    def on_anomaly(self, callback: Callable) -> None:
        """Register callback(signal_type, count, window_seconds) for anomaly events."""
        self._callbacks.append(callback)

    def record(self, signal_type: str) -> bool:
        """Record one occurrence. Returns True if an anomaly threshold was crossed."""
        now = time.time()
        with self._lock:
            dq = self._timestamps[signal_type]
            dq.append(now)
            cutoff = now - self._window
            while dq and dq[0] < cutoff:
                dq.popleft()
            count = len(dq)
        threshold = self._baseline * self._multiplier
        if count >= threshold:
            for cb in self._callbacks:
                try:
                    cb(signal_type, count, self._window)
                except Exception:
                    pass
            return True
        return False

    def counts(self) -> dict[str, int]:
        """Return current per-type counts within the active window."""
        now = time.time()
        cutoff = now - self._window
        with self._lock:
            return {k: sum(1 for ts in dq if ts >= cutoff)
                    for k, dq in self._timestamps.items()}


# ---------------------------------------------------------------------------
# OrganBus
# ---------------------------------------------------------------------------


class OrganBus:
    """
    LLM-mediated publish/subscribe bus for PRISM logic engines.

    Usage
    -----
    ::

        bus = OrganBus(llm_router=router)

        # Policy engine subscribes to injury signals
        bus.register(
            organ_name   = "policy_engine",
            signal_types = ["injury_risk_elevated", "performance_plateau"],
            handler      = policy.on_signal,
            vocabulary   = "Understands: adjustment (str), factor (0-1 float), "
                           "duration_days (int), reason (str)",
        )

        # Physics engine emits a signal
        bus.emit(OrganSignal(
            source      = "physics_engine",
            signal_type = "injury_risk_elevated",
            payload     = {"risk": 0.78, "muscle_group": "hamstring",
                           "model_confidence": 0.92},
            priority    = HIGH,
        ))
    """

    def __init__(
        self,
        llm_router: Optional[Any] = None,
        db_path: str    = "~/.prism/organ_bus.db",
    ) -> None:
        self._router          = llm_router
        self._db              = Path(db_path).expanduser()
        self._db.parent.mkdir(parents=True, exist_ok=True)
        self._subscribers:    list[OrganSubscription] = []
        self._cache:          dict[tuple[str, str, str], dict] = {}
        self._cache_maxsize   = 512  # evict oldest 25% when limit reached
        self._batch:          list[OrganSignal] = []
        self._lock            = threading.Lock()
        self.anomaly_detector = SignalAnomalyDetector()
        self._init_db()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(
        self,
        organ_name:   str,
        signal_types: list[str],
        handler:      Callable,
        vocabulary:   str = "",
    ) -> None:
        """Subscribe an organ to one or more signal types.

        Parameters
        ----------
        organ_name :
            Unique name for the subscribing engine.
        signal_types :
            List of signal type strings this organ wants to receive.
        handler :
            ``handler(payload: dict) -> None`` — called with the translated payload.
        vocabulary :
            Prose description of what data format/keys this engine expects.
            The richer this description, the better the LLM translation.
        """
        sub = OrganSubscription(
            organ_name   = organ_name,
            signal_types = list(signal_types),
            handler      = handler,
            vocabulary   = vocabulary,
        )
        with self._lock:
            # Replace existing subscription for the same organ
            self._subscribers = [s for s in self._subscribers
                                  if s.organ_name != organ_name]
            self._subscribers.append(sub)
        logger.info("[organ_bus] %s registered for %s", organ_name, signal_types)

    def unregister(self, organ_name: str) -> None:
        """Remove an organ's subscription."""
        with self._lock:
            self._subscribers = [s for s in self._subscribers
                                  if s.organ_name != organ_name]

    def emit(self, signal: OrganSignal) -> list[DeliveryRecord]:
        """
        Emit a signal onto the bus.  Delivers synchronously to all matching
        subscribers (except LOW priority — those queue up for flush_batch()).

        Returns list of DeliveryRecords (one per subscriber that received it).
        """
        self.anomaly_detector.record(signal.signal_type)

        if signal.priority == LOW:
            with self._lock:
                self._batch.append(signal)
            logger.debug("[organ_bus] Queued LOW signal %s from %s",
                         signal.signal_type, signal.source)
            return []

        return self._route(signal)

    def flush_batch(self) -> list[DeliveryRecord]:
        """Deliver all queued LOW-priority signals now."""
        with self._lock:
            batch, self._batch = list(self._batch), []
        records = []
        for sig in batch:
            records.extend(self._route(sig))
        return records

    def history(self, n: int = 20) -> list[dict]:
        """Return the n most recent signal records from the DB."""
        try:
            con = self._connect()
            rows = con.execute(
                "SELECT * FROM signals ORDER BY timestamp DESC LIMIT ?", (n,)
            ).fetchall()
            con.close()
            return [dict(zip(
                ["signal_id","source","signal_type","payload","priority",
                 "timestamp","receivers","status"],
                r)) for r in rows]
        except Exception as exc:
            logger.debug("[organ_bus] history() failed: %s", exc)
            return []

    def subscribers_for(self, signal_type: str) -> list[str]:
        """Return organ names subscribed to a given signal type."""
        with self._lock:
            return [s.organ_name for s in self._subscribers
                    if signal_type in s.signal_types]

    # ------------------------------------------------------------------
    # Internal routing
    # ------------------------------------------------------------------

    def _route(self, signal: OrganSignal) -> list[DeliveryRecord]:
        """Deliver signal to all matching subscribers."""
        with self._lock:
            matching = [s for s in self._subscribers
                        if signal.signal_type in s.signal_types
                        and s.organ_name != signal.source]

        records: list[DeliveryRecord] = []
        for sub in matching:
            rec = self._deliver(signal, sub)
            records.append(rec)

        self._persist_signal(signal, [r.receiver for r in records])
        return records

    def _deliver(self, signal: OrganSignal, sub: OrganSubscription) -> DeliveryRecord:
        """Translate and deliver one signal to one subscriber."""
        t0 = time.time()
        via_llm = False

        # ── Choose translation strategy ───────────────────────────────────────
        if self._can_direct_route(signal, sub):
            translated = dict(signal.payload)   # direct pass — no LLM
        else:
            translated, via_llm = self._translate(signal, sub)

        elapsed = (time.time() - t0) * 1000
        success = False
        error   = ""

        try:
            sub.handler(translated)
            success = True
        except Exception as exc:
            error = str(exc)
            logger.warning("[organ_bus] Handler %s failed for %s: %s",
                           sub.organ_name, signal.signal_type, exc)

        rec = DeliveryRecord(
            signal_id   = signal.signal_id,
            receiver    = sub.organ_name,
            translated  = translated,
            via_llm     = via_llm,
            duration_ms = elapsed,
            success     = success,
            error       = error,
        )
        logger.debug(
            "[organ_bus] %s→%s (%s) %s in %.0fms",
            signal.source, sub.organ_name, signal.signal_type,
            "LLM" if via_llm else "direct", elapsed,
        )
        return rec

    def _can_direct_route(
        self, signal: OrganSignal, sub: OrganSubscription
    ) -> bool:
        """
        True if payload can be passed without LLM translation.

        Conditions (all must hold):
        1. No LLM router available (forced fallback)
        2. OR: signal priority is HIGH — always translate for clarity
           Actually HIGH means always translate (more important, richer context)
           So: use direct only when priority=NORMAL/LOW AND overlap is sufficient
        3. Payload key overlap with vocabulary keywords ≥ 50%
           (heuristic: vocabulary prose mentions ≥ half the payload keys)
        """
        if not self._router:
            return True   # no LLM available — always direct

        if signal.priority == HIGH:
            return False  # high-priority signals always get LLM translation

        if not sub.vocabulary:
            return True   # no vocabulary declared — can't guide LLM anyway

        # Check key overlap: how many payload keys appear in the vocab string
        vocab_lower = sub.vocabulary.lower()
        payload_keys = [k.lower() for k in signal.payload]
        if not payload_keys:
            return True
        overlap = sum(1 for k in payload_keys if k in vocab_lower)
        return overlap / len(payload_keys) >= 0.5

    def _translate(
        self, signal: OrganSignal, sub: OrganSubscription
    ) -> tuple[dict, bool]:
        """
        LLM-translate signal payload into subscriber's vocabulary.

        Returns (translated_dict, via_llm=True).
        Falls back to direct pass if LLM unavailable or call fails.
        """
        if not self._router:
            return dict(signal.payload), False

        # Cache key: (signal_type, source, receiver) — HIGH bypasses cache
        cache_key = (signal.signal_type, signal.source, sub.organ_name)
        if signal.priority != HIGH and cache_key in self._cache:
            cached = dict(self._cache[cache_key])
            # Merge current payload values into the cached structure
            cached.update({k: v for k, v in signal.payload.items()
                           if k in cached})
            return cached, True

        prompt = _TRANSLATE_PROMPT.format(
            source      = signal.source,
            signal_type = signal.signal_type,
            payload     = json.dumps(signal.payload, default=str)[:400],
            receiver    = sub.organ_name,
            vocabulary  = sub.vocabulary[:400],
        )
        try:
            raw, _ = self._router.call(
                prompt, min_capability=1, max_tokens=200, json_mode=True)
            clean = (raw.strip()
                     .lstrip("```json").lstrip("```")
                     .rstrip("```").strip())
            translated = json.loads(clean)
            if not isinstance(translated, dict):
                raise ValueError("LLM returned non-dict")
        except Exception as exc:
            logger.debug("[organ_bus] LLM translate failed (%s→%s): %s",
                         signal.source, sub.organ_name, exc)
            return dict(signal.payload), False

        if signal.priority != HIGH:
            with self._lock:
                if len(self._cache) >= self._cache_maxsize:
                    # Evict oldest quarter of entries
                    evict_keys = list(self._cache.keys())[:self._cache_maxsize // 4]
                    for k in evict_keys:
                        del self._cache[k]
                self._cache[cache_key] = dict(translated)

        return translated, True

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        con = self._connect()
        con.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                signal_id   TEXT PRIMARY KEY,
                source      TEXT NOT NULL,
                signal_type TEXT NOT NULL,
                payload     TEXT NOT NULL,
                priority    INTEGER NOT NULL,
                timestamp   REAL NOT NULL,
                receivers   TEXT NOT NULL,
                status      TEXT NOT NULL DEFAULT 'delivered'
            )
        """)
        con.commit()
        con.close()

    def _persist_signal(
        self, signal: OrganSignal, receivers: list[str]
    ) -> None:
        try:
            con = self._connect()
            con.execute(
                """INSERT OR REPLACE INTO signals
                   (signal_id, source, signal_type, payload, priority, timestamp,
                    receivers, status)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (
                    signal.signal_id,
                    signal.source,
                    signal.signal_type,
                    json.dumps(signal.payload, default=str),
                    signal.priority,
                    signal.timestamp,
                    json.dumps(receivers),
                    "delivered" if receivers else "no_subscribers",
                ),
            )
            con.commit()
            con.close()
        except Exception as exc:
            logger.debug("[organ_bus] Persist failed: %s", exc)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self._db))
