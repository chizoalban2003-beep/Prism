"""
prism_persona.py
================
Crystallised behavioural profile — HOW the user operates.
Stores patterns, style preferences, and adaptive behaviour PRISM has inferred.
Distinct from PrismSoul (which stores beliefs/values).
"""
from __future__ import annotations

import json
import logging
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class PersonaTrait:
    name: str
    value: str
    confidence: float          # 0.0–1.0, grows with observations
    source: str                # "inferred" | "explicit"
    last_updated: float
    observation_count: int


@dataclass
class BehaviorPattern:
    pattern_id: str
    description: str
    frequency: int
    first_seen: float
    last_seen: float
    examples: list[str] = field(default_factory=list)


class PrismPersona:
    """
    SQLite-backed behavioural profile.
    Tables: traits, patterns, active_hours
    DB at ~/.prism/persona.db — schema versioned with PRAGMA user_version.
    """

    _DB_VERSION = 1

    def __init__(self, db_path: str = "~/.prism/persona.db"):
        self._db = Path(db_path).expanduser()
        self._db.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ── Traits ────────────────────────────────────────────────────────────────

    def update_trait(
        self,
        name: str,
        value: str,
        confidence: float,
        source: str = "inferred",
        delta: int = 1,
    ) -> None:
        now = time.time()
        with sqlite3.connect(self._db, timeout=30.0) as con:
            row = con.execute(
                "SELECT confidence, observation_count FROM traits WHERE name=?", (name,)
            ).fetchone()
            if row is None:
                con.execute(
                    "INSERT INTO traits VALUES (?,?,?,?,?,?)",
                    (name, value, min(1.0, confidence), source, now, delta),
                )
            else:
                obs = row[1] + delta
                # Weighted merge — explicit overrides inferred values
                new_conf = min(1.0, confidence) if source == "explicit" else min(
                    1.0, (row[0] * row[1] + confidence * delta) / (row[1] + delta)
                )
                con.execute(
                    "UPDATE traits SET value=?, confidence=?, source=?, "
                    "last_updated=?, observation_count=? WHERE name=?",
                    (value, round(new_conf, 4), source, now, obs, name),
                )

    def get_trait(self, name: str) -> PersonaTrait | None:
        with sqlite3.connect(self._db, timeout=30.0) as con:
            row = con.execute("SELECT * FROM traits WHERE name=?", (name,)).fetchone()
        return self._trait_from_row(row) if row else None

    def list_traits(self) -> list[PersonaTrait]:
        with sqlite3.connect(self._db, timeout=30.0) as con:
            rows = con.execute(
                "SELECT * FROM traits ORDER BY confidence DESC"
            ).fetchall()
        return [self._trait_from_row(r) for r in rows]

    # ── Patterns ──────────────────────────────────────────────────────────────

    def add_pattern(self, description: str, example: str = "") -> str:
        pid = str(uuid.uuid4())[:8]
        now = time.time()
        examples_json = json.dumps([example] if example else [])
        with sqlite3.connect(self._db, timeout=30.0) as con:
            con.execute(
                "INSERT INTO patterns VALUES (?,?,?,?,?,?)",
                (pid, description, 1, now, now, examples_json),
            )
        return pid

    def bump_pattern(self, description: str, example: str = "") -> None:
        """Find a pattern by description similarity and increment its frequency."""
        desc_lower = description.lower()
        with sqlite3.connect(self._db, timeout=30.0) as con:
            rows = con.execute(
                "SELECT pattern_id, description, frequency, examples FROM patterns"
            ).fetchall()
            match = None
            for pid, desc, freq, ex_json in rows:
                # Accept match if descriptions share ≥3 significant words
                words_a = set(desc_lower.split())
                words_b = set(desc.lower().split())
                shared = words_a & words_b - {"the", "a", "an", "of", "in", "on", "to"}
                if len(shared) >= 3 or desc.lower() == desc_lower:
                    match = (pid, freq, ex_json)
                    break

            now = time.time()
            if match is None:
                self.add_pattern(description, example)
                return

            pid, freq, ex_json = match
            examples = json.loads(ex_json or "[]")
            if example and example not in examples:
                examples = (examples + [example])[-3:]
            con.execute(
                "UPDATE patterns SET frequency=?, last_seen=?, examples=? WHERE pattern_id=?",
                (freq + 1, now, json.dumps(examples), pid),
            )

    # ── Active hours ──────────────────────────────────────────────────────────

    def record_active_hour(self, hour: int) -> None:
        """Record an interaction occurring at `hour` (0–23)."""
        with sqlite3.connect(self._db, timeout=30.0) as con:
            row = con.execute(
                "SELECT count FROM active_hours WHERE hour=?", (hour,)
            ).fetchone()
            if row is None:
                con.execute("INSERT INTO active_hours VALUES (?,?)", (hour, 1))
            else:
                con.execute(
                    "UPDATE active_hours SET count=? WHERE hour=?",
                    (row[0] + 1, hour),
                )

    def peak_hours(self) -> list[int]:
        """Return the 3 most active hours."""
        with sqlite3.connect(self._db, timeout=30.0) as con:
            rows = con.execute(
                "SELECT hour FROM active_hours ORDER BY count DESC LIMIT 3"
            ).fetchall()
        return [r[0] for r in rows]

    # ── Context / summary ─────────────────────────────────────────────────────

    def build_context(self, max_chars: int = 500) -> str:
        traits = {t.name: t for t in self.list_traits()}
        patterns = self._top_patterns(5)
        peaks = self.peak_hours()

        style = traits.get("communication_style")
        length = traits.get("response_length_preference")
        tech = traits.get("technical_depth")
        risk = traits.get("risk_tolerance")

        lines: list[str] = []

        style_parts: list[str] = []
        if style:
            style_parts.append(style.value)
        if length:
            style_parts.append(f"{length.value} responses preferred")
        if tech:
            style_parts.append(f"{tech.value} technical depth")
        if style_parts:
            lines.append(f"Style: {' · '.join(style_parts)}")

        if peaks:
            peak_str = ", ".join(f"{h}:00" for h in peaks[:2])
            lines.append(f"Active hours: peak at {peak_str}")

        if patterns:
            patt_str = " · ".join(p.description[:60] for p in patterns[:3])
            lines.append(f"Patterns: {patt_str}")

        if risk:
            lines.append(f"Risk tolerance: {risk.value}")

        total_obs = sum(t.observation_count for t in traits.values())
        lines.append(
            f"Confidence: {total_obs} observations · "
            f"{len(patterns)} patterns · {len(traits)} traits"
        )

        header = "[Crystallised user profile]"
        body = "\n".join(lines)
        result = f"{header}\n{body}"
        if len(result) > max_chars:
            result = result[:max_chars - 3] + "..."
        return result

    def summary(self) -> str:
        traits = self.list_traits()
        patterns = self._top_patterns(10)
        peaks = self.peak_hours()

        lines = ["**Behavioural Profile (Crystallised)**\n"]

        if traits:
            lines.append("**Traits:**")
            for t in traits:
                bar = "█" * int(t.confidence * 10) + "░" * (10 - int(t.confidence * 10))
                lines.append(
                    f"  {t.name}: {t.value}  [{bar}] "
                    f"{int(t.confidence * 100)}% ({t.observation_count} obs, {t.source})"
                )

        if patterns:
            lines.append("\n**Patterns:**")
            for p in patterns:
                lines.append(f"  [{p.frequency}×] {p.description}")
                if p.examples:
                    lines.append(f"      e.g. {p.examples[-1]}")

        if peaks:
            lines.append(f"\n**Peak hours:** {', '.join(f'{h}:00' for h in peaks)}")

        return "\n".join(lines)

    def growth_since(self, days: int = 7) -> dict:
        cutoff = time.time() - days * 86400
        with sqlite3.connect(self._db, timeout=30.0) as con:
            new_traits = con.execute(
                "SELECT COUNT(*) FROM traits WHERE last_updated >= ?", (cutoff,)
            ).fetchone()[0]
            new_patterns = con.execute(
                "SELECT COUNT(*) FROM patterns WHERE first_seen >= ?", (cutoff,)
            ).fetchone()[0]
            conf_rows = con.execute(
                "SELECT confidence FROM traits WHERE last_updated >= ?", (cutoff,)
            ).fetchall()
        conf_gain = round(sum(r[0] for r in conf_rows) / max(1, len(conf_rows)), 3)
        return {
            "new_traits": new_traits,
            "new_patterns": new_patterns,
            "confidence_avg": conf_gain,
            "days": days,
        }

    # ── Internal ──────────────────────────────────────────────────────────────

    def _top_patterns(self, n: int = 5) -> list[BehaviorPattern]:
        with sqlite3.connect(self._db, timeout=30.0) as con:
            rows = con.execute(
                "SELECT * FROM patterns ORDER BY frequency DESC LIMIT ?", (n,)
            ).fetchall()
        return [self._pattern_from_row(r) for r in rows]

    @staticmethod
    def _trait_from_row(row: tuple) -> PersonaTrait:
        return PersonaTrait(
            name=row[0],
            value=row[1],
            confidence=row[2],
            source=row[3],
            last_updated=row[4],
            observation_count=row[5],
        )

    @staticmethod
    def _pattern_from_row(row: tuple) -> BehaviorPattern:
        return BehaviorPattern(
            pattern_id=row[0],
            description=row[1],
            frequency=row[2],
            first_seen=row[3],
            last_seen=row[4],
            examples=json.loads(row[5] or "[]"),
        )

    def _init_db(self) -> None:
        with sqlite3.connect(self._db, timeout=30.0) as con:
            con.execute("""
                CREATE TABLE IF NOT EXISTS traits (
                    name              TEXT PRIMARY KEY,
                    value             TEXT NOT NULL,
                    confidence        REAL NOT NULL DEFAULT 0.5,
                    source            TEXT NOT NULL DEFAULT 'inferred',
                    last_updated      REAL NOT NULL,
                    observation_count INTEGER NOT NULL DEFAULT 0
                )
            """)
            con.execute("""
                CREATE TABLE IF NOT EXISTS patterns (
                    pattern_id  TEXT PRIMARY KEY,
                    description TEXT NOT NULL,
                    frequency   INTEGER NOT NULL DEFAULT 1,
                    first_seen  REAL NOT NULL,
                    last_seen   REAL NOT NULL,
                    examples    TEXT NOT NULL DEFAULT '[]'
                )
            """)
            con.execute("""
                CREATE TABLE IF NOT EXISTS active_hours (
                    hour  INTEGER PRIMARY KEY,
                    count INTEGER NOT NULL DEFAULT 0
                )
            """)
            self._migrate(con)

    def _migrate(self, con: sqlite3.Connection) -> None:
        ver = con.execute("PRAGMA user_version").fetchone()[0]
        if ver < self._DB_VERSION:
            con.execute(f"PRAGMA user_version = {self._DB_VERSION}")
