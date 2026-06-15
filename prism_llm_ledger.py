"""
prism_llm_ledger.py
===================
Lightweight SQLite ledger for every LLM call made by PRISM.

Records provider, model, token counts, latency, USD cost, and caller
source for every call routed through LLMRouter.call().

Token counts
------------
  Claude / OpenAI responses include actual usage in the API response body.
  LLMRouter captures those when available and passes them here.
  When not available (Ollama, stdlib), tokens are estimated as len(text)//4
  — accurate to ~±15 %, sufficient for cost dashboards.

Pricing table
-------------
  Hard-coded 2026 market rates (USD per million tokens, input/output).
  Override via PRISM_LLM_PRICES env var as JSON or pass custom_prices to
  get_ledger(). Ollama/stdlib cost $0 (local compute).

Thread-safe singleton — call get_ledger() from anywhere.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# USD per million tokens (input_price, output_price)
_DEFAULT_PRICES: dict[str, tuple[float, float]] = {
    "claude-opus":        (15.00, 75.00),
    "claude-sonnet":      (3.00,  15.00),
    "claude-haiku":       (0.25,   1.25),
    "gpt-4o-mini":        (0.15,   0.60),
    "gpt-4o":             (2.50,  10.00),
    "gpt-4":              (30.00, 60.00),
    "ollama":             (0.00,   0.00),
    "stdlib":             (0.00,   0.00),
}


def _price_for(provider: str, model: str, prices: dict) -> tuple[float, float]:
    """Return (input_price, output_price) per million tokens for this model."""
    key = model.lower()
    for pattern, rate in prices.items():
        if pattern in key:
            return rate
    # Fall back to provider-level key
    if provider in prices:
        return prices[provider]
    return (0.00, 0.00)


def _cost_usd(input_tokens: int, output_tokens: int,
              in_price: float, out_price: float) -> float:
    return (input_tokens * in_price + output_tokens * out_price) / 1_000_000


@dataclass
class CallRecord:
    provider:      str
    model:         str
    input_tokens:  int
    output_tokens: int
    latency_ms:    float
    cost_usd:      float
    source:        str   = "unknown"
    session_id:    str   = ""
    call_id:       str   = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp:     float = field(default_factory=time.time)


class LLMLedger:
    """
    SQLite-backed ledger for LLM call accounting.

    Usage
    -----
        ledger = get_ledger()
        ledger.record_call(provider="claude", model="claude-sonnet-4",
                           input_tokens=512, output_tokens=256,
                           latency_ms=1200, source="chain")
        summary = ledger.summary()
    """

    def __init__(
        self,
        db_path: str = "~/.prism/llm_ledger.db",
        custom_prices: Optional[dict] = None,
    ) -> None:
        self._db = Path(db_path).expanduser()
        self._db.parent.mkdir(parents=True, exist_ok=True)
        self._prices = dict(_DEFAULT_PRICES)
        # Allow override via env var
        env_prices = os.environ.get("PRISM_LLM_PRICES", "")
        if env_prices:
            try:
                self._prices.update(json.loads(env_prices))
            except Exception:
                pass
        if custom_prices:
            self._prices.update(custom_prices)
        self._lock = threading.Lock()
        self._init_db()

    # ------------------------------------------------------------------
    # DB helpers
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self._db), timeout=30.0)

    def _init_db(self) -> None:
        with self._lock:
            con = self._connect()
            con.execute("""
                CREATE TABLE IF NOT EXISTS calls (
                    call_id       TEXT PRIMARY KEY,
                    timestamp     REAL NOT NULL,
                    provider      TEXT NOT NULL,
                    model         TEXT NOT NULL,
                    input_tokens  INTEGER NOT NULL DEFAULT 0,
                    output_tokens INTEGER NOT NULL DEFAULT 0,
                    latency_ms    REAL NOT NULL DEFAULT 0.0,
                    cost_usd      REAL NOT NULL DEFAULT 0.0,
                    source        TEXT NOT NULL DEFAULT 'unknown',
                    session_id    TEXT NOT NULL DEFAULT ''
                )
            """)
            con.execute("CREATE INDEX IF NOT EXISTS idx_ts ON calls(timestamp)")
            con.execute("CREATE INDEX IF NOT EXISTS idx_model ON calls(provider, model)")
            con.commit()
            con.close()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def record_call(
        self,
        provider:      str,
        model:         str,
        input_tokens:  int,
        output_tokens: int,
        latency_ms:    float,
        source:        str = "unknown",
        session_id:    str = "",
    ) -> CallRecord:
        in_p, out_p = _price_for(provider, model, self._prices)
        cost = _cost_usd(input_tokens, output_tokens, in_p, out_p)
        rec = CallRecord(
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            cost_usd=cost,
            source=source,
            session_id=session_id,
        )
        with self._lock:
            con = self._connect()
            con.execute(
                """INSERT INTO calls
                   (call_id, timestamp, provider, model,
                    input_tokens, output_tokens, latency_ms, cost_usd, source, session_id)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (rec.call_id, rec.timestamp, rec.provider, rec.model,
                 rec.input_tokens, rec.output_tokens, rec.latency_ms,
                 rec.cost_usd, rec.source, rec.session_id),
            )
            con.commit()
            con.close()
        return rec

    def clear(self) -> int:
        """Delete all records. Returns count deleted."""
        with self._lock:
            con = self._connect()
            count = con.execute("SELECT COUNT(*) FROM calls").fetchone()[0]
            con.execute("DELETE FROM calls")
            con.commit()
            con.close()
        return count

    # ------------------------------------------------------------------
    # Read / aggregate
    # ------------------------------------------------------------------

    def summary(self, since_ts: float = 0.0) -> dict:
        """Overall totals since since_ts (default: all time)."""
        con = self._connect()
        row = con.execute(
            """SELECT COUNT(*), SUM(input_tokens), SUM(output_tokens),
                      SUM(cost_usd), AVG(latency_ms)
               FROM calls WHERE timestamp >= ?""",
            (since_ts,),
        ).fetchone()
        con.close()
        calls, inp, out, cost, lat = row
        return {
            "total_calls":         calls or 0,
            "total_input_tokens":  inp  or 0,
            "total_output_tokens": out  or 0,
            "total_tokens":        (inp or 0) + (out or 0),
            "total_cost_usd":      round(cost or 0.0, 6),
            "avg_latency_ms":      round(lat  or 0.0, 1),
        }

    def by_model(self, days: int = 30) -> list[dict]:
        """Per-model breakdown for the last N days."""
        since = time.time() - days * 86400
        con = self._connect()
        rows = con.execute(
            """SELECT provider, model,
                      COUNT(*) as calls,
                      SUM(input_tokens), SUM(output_tokens),
                      SUM(cost_usd), AVG(latency_ms)
               FROM calls WHERE timestamp >= ?
               GROUP BY provider, model
               ORDER BY SUM(cost_usd) DESC""",
            (since,),
        ).fetchall()
        con.close()
        return [
            {
                "provider":      r[0],
                "model":         r[1],
                "calls":         r[2],
                "input_tokens":  r[3] or 0,
                "output_tokens": r[4] or 0,
                "total_tokens":  (r[3] or 0) + (r[4] or 0),
                "cost_usd":      round(r[5] or 0.0, 6),
                "avg_latency_ms": round(r[6] or 0.0, 1),
            }
            for r in rows
        ]

    def by_day(self, days: int = 30) -> list[dict]:
        """Daily totals for the last N days, newest first."""
        since = time.time() - days * 86400
        con = self._connect()
        rows = con.execute(
            """SELECT date(timestamp, 'unixepoch') as day,
                      COUNT(*), SUM(input_tokens), SUM(output_tokens), SUM(cost_usd)
               FROM calls WHERE timestamp >= ?
               GROUP BY day
               ORDER BY day DESC""",
            (since,),
        ).fetchall()
        con.close()
        return [
            {
                "date":          r[0],
                "calls":         r[1],
                "input_tokens":  r[2] or 0,
                "output_tokens": r[3] or 0,
                "total_tokens":  (r[2] or 0) + (r[3] or 0),
                "cost_usd":      round(r[4] or 0.0, 6),
            }
            for r in rows
        ]

    def by_source(self, days: int = 30) -> list[dict]:
        """Breakdown by caller source (chain, agent, organ, api…)."""
        since = time.time() - days * 86400
        con = self._connect()
        rows = con.execute(
            """SELECT source, COUNT(*), SUM(input_tokens), SUM(output_tokens), SUM(cost_usd)
               FROM calls WHERE timestamp >= ?
               GROUP BY source ORDER BY SUM(cost_usd) DESC""",
            (since,),
        ).fetchall()
        con.close()
        return [
            {
                "source":        r[0],
                "calls":         r[1],
                "input_tokens":  r[2] or 0,
                "output_tokens": r[3] or 0,
                "total_tokens":  (r[2] or 0) + (r[3] or 0),
                "cost_usd":      round(r[4] or 0.0, 6),
            }
            for r in rows
        ]

    def recent(self, n: int = 20) -> list[dict]:
        """Return the n most recent call records."""
        con = self._connect()
        rows = con.execute(
            """SELECT call_id, timestamp, provider, model,
                      input_tokens, output_tokens, latency_ms, cost_usd, source, session_id
               FROM calls ORDER BY timestamp DESC LIMIT ?""",
            (n,),
        ).fetchall()
        con.close()
        return [
            {
                "call_id":       r[0],
                "timestamp":     r[1],
                "provider":      r[2],
                "model":         r[3],
                "input_tokens":  r[4],
                "output_tokens": r[5],
                "latency_ms":    round(r[6], 1),
                "cost_usd":      round(r[7], 6),
                "source":        r[8],
                "session_id":    r[9],
            }
            for r in rows
        ]

    def price_table(self) -> dict:
        """Return the active price table (pattern → [input_usd_per_m, output_usd_per_m])."""
        return {k: list(v) for k, v in self._prices.items()}


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_ledger: Optional[LLMLedger] = None
_ledger_lock = threading.Lock()


def get_ledger(
    db_path: str = "~/.prism/llm_ledger.db",
    custom_prices: Optional[dict] = None,
) -> LLMLedger:
    """Return the process-wide LLMLedger singleton, creating it if needed."""
    global _ledger
    if _ledger is None:
        with _ledger_lock:
            if _ledger is None:
                _ledger = LLMLedger(db_path=db_path, custom_prices=custom_prices)
    return _ledger


def reset_ledger(db_path: str = "~/.prism/llm_ledger.db") -> LLMLedger:
    """Replace the singleton — used in tests."""
    global _ledger
    with _ledger_lock:
        _ledger = LLMLedger(db_path=db_path)
    return _ledger
