from __future__ import annotations
import json, logging, sqlite3, time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

@dataclass
class CalibrationEvent:
    event_id:   str
    domain:     str
    message:    str
    direction:  str
    factor_id:  str
    adjustment: float
    timestamp:  float = field(default_factory=time.time)

class PrismCalibration:
    """
    Conversational calibration — updates the decision model from feedback.
    """

    FEEDBACK_PATTERNS = {
        "too_aggressive": [
            "too aggressive","too risky","too much","overdid",
            "too bold","shouldn't have","that was too far",
            "dial it back","more conservative","play it safe",
        ],
        "too_conservative": [
            "too cautious","too safe","too slow","not enough",
            "too timid","should have gone","more aggressive",
            "bolder","take more risk","push harder",
        ],
        "wrong": [
            "that was wrong","disagree","wouldn't do that",
            "bad recommendation","incorrect","missed the point",
            "not what i wanted","off the mark",
        ],
        "correct": [
            "that was right","good call","exactly","perfect",
            "well done","correct","spot on","that worked",
            "agree","that's what i wanted",
        ],
    }

    def __init__(self, db_path="~/.prism/calibration.db"):
        self._db = Path(db_path).expanduser()
        self._db.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def detect(self, message: str) -> Optional[str]:
        msg_lower = message.lower()
        for direction, patterns in self.FEEDBACK_PATTERNS.items():
            if any(p in msg_lower for p in patterns):
                return direction
        return None

    def process(
        self,
        message:      str,
        direction:    str,
        last_decision:dict,
        beam,
        llm_router=None,
    ) -> CalibrationEvent:
        domain   = last_decision.get("domain", "general")
        position = last_decision.get("fulcrum_position", 0.5)
        factors  = last_decision.get("factors", {})

        adjustment_map = {
            "too_aggressive":  -0.08,
            "too_conservative": 0.08,
            "wrong":           -0.05,
            "correct":          0.03,
        }
        base_adj = adjustment_map.get(direction, 0.0)

        factor_id = "general_context"
        if llm_router and factors:
            prompt = (
                f"A decision system recommended an action that was "
                f"'{direction}'. The decision had these factors:\n"
                f"{json.dumps(factors, indent=2)}\n"
                f"User feedback: '{message}'\n"
                f"Which ONE factor id from the list above most needs "
                f"adjusting? Return ONLY the factor_id string, nothing else."
            )
            factor_raw, _ = llm_router.call(
                prompt, min_capability=1, max_tokens=30)
            candidate = factor_raw.strip().strip('"\'')
            if candidate in factors:
                factor_id = candidate

        if beam and hasattr(beam, 'fulcrum'):
            try:
                actual    = max(0.0, min(1.0, position + base_adj))
                predicted = position
                beam.fulcrum.observe(actual, predicted, position)
                logger.info("Calibrated %s factor '%s' by %+.3f",
                            domain, factor_id, base_adj)
            except Exception as e:
                logger.debug("Calibration update failed: %s", e)

        event = CalibrationEvent(
            event_id   = f"{time.time():.6f}",
            domain     = domain,
            message    = message,
            direction  = direction,
            factor_id  = factor_id,
            adjustment = base_adj,
        )
        self._store(event)
        return event

    def history(self, domain: str = None,
                 n: int = 20) -> list[CalibrationEvent]:
        with sqlite3.connect(self._db) as c:
            if domain:
                rows = c.execute(
                    "SELECT * FROM calibration WHERE domain=? "
                    "ORDER BY ts DESC LIMIT ?",
                    (domain, n)).fetchall()
            else:
                rows = c.execute(
                    "SELECT * FROM calibration ORDER BY ts DESC LIMIT ?",
                    (n,)).fetchall()
        return [CalibrationEvent(*r) for r in rows]

    def summary(self) -> str:
        events = self.history(n=20)
        if not events:
            return "No calibration history yet."
        directions = [e.direction for e in events]
        from collections import Counter
        counts = Counter(directions)
        return (f"{len(events)} calibration events: "
                + ", ".join(f"{v} {k}" for k, v in counts.most_common()))

    def _store(self, event: CalibrationEvent) -> None:
        with sqlite3.connect(self._db) as c:
            c.execute("INSERT INTO calibration VALUES(?,?,?,?,?,?,?)",
                      (event.event_id, event.domain, event.message,
                       event.direction, event.factor_id,
                       event.adjustment, event.timestamp))

    def _init_db(self) -> None:
        with sqlite3.connect(self._db) as c:
            c.execute("""CREATE TABLE IF NOT EXISTS calibration(
                id TEXT PRIMARY KEY, domain TEXT, message TEXT,
                direction TEXT, factor_id TEXT,
                adjustment REAL, ts REAL)""")
