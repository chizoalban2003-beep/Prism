from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from identity_bus import IdentityBus, IdentitySignal


@dataclass
class DomainProfile:
    """Crystallised profile for one domain."""

    domain: str
    fixed_fulcrum: float
    variance: float
    n_observations: int
    crystallised: bool
    last_updated: float

    @property
    def confidence(self) -> float:
        return min(1.0, self.n_observations / 50.0) * (1.0 - max(0.0, min(1.0, self.variance)))


@dataclass
class DigitalIdentity:
    """
    The crystallised decision fingerprint of one user.
    Emerges from accumulated decisions across all sub-agents.
    Not a consciousness. A decision profile.
    """

    user_name: str
    domains: dict[str, DomainProfile]
    cross_signals: dict[str, float]
    overall_risk: float
    consistency: float
    n_total: int
    created_at: float
    last_updated: float
    version: int = 1

    CRYSTALLISE_THRESHOLD = 0.02

    @property
    def crystallised_domains(self) -> list[str]:
        return [d for d, p in self.domains.items() if p.crystallised]

    @property
    def confidence(self) -> float:
        if not self.domains:
            return 0.0
        return sum(p.confidence for p in self.domains.values()) / len(self.domains)

    def emergent_insight(self) -> str:
        vals = [p.fixed_fulcrum for p in self.domains.values() if p.crystallised]
        if not vals:
            return "Insufficient data for insight."
        if max(vals) - min(vals) > 0.40:
            return "Compartmentaliser — aggressive in some domains, conservative in others."
        if self.overall_risk > 0.65:
            return "Risk-seeking across contexts."
        if self.overall_risk < 0.35:
            return "Consistently conservative decision-maker."
        if self.cross_signals.get("time_pressure_response", 0.5) > 0.75:
            return "Thrives under pressure — decisions improve with urgency."
        return "Balanced and consistent decision-maker."

    def to_dict(self) -> dict:
        return {
            "user_name": self.user_name,
            "domains": {
                domain: {
                    "domain": profile.domain,
                    "fixed_fulcrum": profile.fixed_fulcrum,
                    "variance": profile.variance,
                    "n_observations": profile.n_observations,
                    "crystallised": profile.crystallised,
                    "last_updated": profile.last_updated,
                    "confidence": profile.confidence,
                }
                for domain, profile in sorted(self.domains.items())
            },
            "cross_signals": dict(sorted(self.cross_signals.items())),
            "overall_risk": self.overall_risk,
            "consistency": self.consistency,
            "n_total": self.n_total,
            "created_at": self.created_at,
            "last_updated": self.last_updated,
            "version": self.version,
            "crystallised_domains": self.crystallised_domains,
            "confidence": self.confidence,
            "insight": self.emergent_insight(),
        }

    def to_card_data(self) -> dict:
        """Returns data dict for identity_card() in prism_responses.py."""
        return {
            "domains": [
                {
                    "label": d,
                    "value": round(p.fixed_fulcrum, 3),
                    "crystallised": p.crystallised,
                }
                for d, p in sorted(self.domains.items())
            ],
            "insight": self.emergent_insight(),
            "confidence": round(self.confidence, 3),
            "n_decisions": self.n_total,
        }


class CrystallisationEngine:
    """
    Builds and updates a DigitalIdentity from observed decisions.
    """

    EMA_ALPHA = 0.08
    CRYSTALLISE_THR = 0.02
    MIN_OBS = 20
    SCHEMA = """
    CREATE TABLE IF NOT EXISTS domain_profiles (
        user_name TEXT NOT NULL,
        domain TEXT NOT NULL,
        fixed_fulcrum REAL NOT NULL,
        variance REAL NOT NULL,
        n_observations INTEGER NOT NULL,
        crystallised INTEGER NOT NULL,
        last_updated REAL NOT NULL,
        PRIMARY KEY (user_name, domain)
    );
    CREATE TABLE IF NOT EXISTS identity_meta (
        user_name TEXT PRIMARY KEY,
        created_at REAL NOT NULL,
        last_updated REAL NOT NULL,
        version INTEGER NOT NULL DEFAULT 1
    );
    """

    def __init__(
        self,
        user_name: str,
        bus: IdentityBus,
        db_path: str = "~/.prism/identity.db",
    ):
        self.user_name = user_name
        self.bus = bus
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        now = time.time()
        with self._connect() as conn:
            conn.executescript(self.SCHEMA)
            conn.execute(
                """
                INSERT INTO identity_meta (user_name, created_at, last_updated, version)
                VALUES (?, ?, ?, 1)
                ON CONFLICT(user_name) DO NOTHING
                """,
                (self.user_name, now, now),
            )

    def _get_domain_row(self, conn: sqlite3.Connection, domain: str) -> Optional[sqlite3.Row]:
        return conn.execute(
            """
            SELECT domain, fixed_fulcrum, variance, n_observations, crystallised, last_updated
            FROM domain_profiles
            WHERE user_name = ? AND domain = ?
            """,
            (self.user_name, domain),
        ).fetchone()

    def observe(
        self,
        domain: str,
        fulcrum_used: float,
        outcome_rating: float,
        context: dict = None,
    ) -> None:
        """
        Update the domain profile with one new observation.
        Uses exponential moving average to update fixed_fulcrum.
        Updates running variance estimate.
        Publishes aggression_index signal if variance stable enough.
        """
        now = time.time()
        fulcrum = max(0.0, min(1.0, float(fulcrum_used)))
        rating = max(0.0, min(1.0, float(outcome_rating)))
        context = context or {}
        with self._connect() as conn:
            row = self._get_domain_row(conn, domain)
            if row is None:
                previous_fixed = fulcrum
                previous_variance = 0.0
                previous_count = 0
            else:
                previous_fixed = float(row["fixed_fulcrum"])
                previous_variance = float(row["variance"])
                previous_count = int(row["n_observations"])
            fixed = (self.EMA_ALPHA * fulcrum) + ((1.0 - self.EMA_ALPHA) * previous_fixed)
            variance = (self.EMA_ALPHA * ((fulcrum - fixed) ** 2)) + (
                (1.0 - self.EMA_ALPHA) * previous_variance
            )
            n_observations = previous_count + 1
            variance = max(0.0, min(1.0, variance))
            crystallised = int(n_observations >= self.MIN_OBS and variance < self.CRYSTALLISE_THR)
            conn.execute(
                """
                INSERT INTO domain_profiles (
                    user_name, domain, fixed_fulcrum, variance, n_observations, crystallised, last_updated
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_name, domain) DO UPDATE SET
                    fixed_fulcrum = excluded.fixed_fulcrum,
                    variance = excluded.variance,
                    n_observations = excluded.n_observations,
                    crystallised = excluded.crystallised,
                    last_updated = excluded.last_updated
                """,
                (self.user_name, domain, fixed, variance, n_observations, crystallised, now),
            )
            conn.execute(
                """
                UPDATE identity_meta
                SET last_updated = ?, version = version + 1
                WHERE user_name = ?
                """,
                (now, self.user_name),
            )
        confidence = min(1.0, max(0.2, n_observations / 50.0))
        self.bus.publish(
            IdentitySignal(
                source=domain,
                signal_id="aggression_index",
                value=fixed,
                confidence=confidence,
                timestamp=now,
            )
        )
        self.bus.publish(
            IdentitySignal(
                source=domain,
                signal_id="consistency_score",
                value=max(0.0, min(1.0, 1.0 - variance)),
                confidence=confidence,
                timestamp=now,
            )
        )
        if "time_pressure_response" in context:
            self.bus.publish(
                IdentitySignal(
                    source=domain,
                    signal_id="time_pressure_response",
                    value=max(0.0, min(1.0, float(context["time_pressure_response"]))),
                    confidence=confidence,
                    timestamp=now,
                )
            )
        if "data_reliance" in context:
            self.bus.publish(
                IdentitySignal(
                    source=domain,
                    signal_id="data_reliance",
                    value=max(0.0, min(1.0, float(context["data_reliance"]))),
                    confidence=confidence,
                    timestamp=now,
                )
            )
        self.bus.publish(
            IdentitySignal(
                source=domain,
                signal_id="risk_override_tendency",
                value=max(0.0, min(1.0, fixed if rating >= 0.5 else (1.0 - fixed))),
                confidence=confidence,
                timestamp=now,
            )
        )

    def get_identity(self) -> Optional[DigitalIdentity]:
        """Return current crystallised identity, or None if insufficient data."""
        with self._connect() as conn:
            meta = conn.execute(
                """
                SELECT created_at, last_updated, version
                FROM identity_meta
                WHERE user_name = ?
                """,
                (self.user_name,),
            ).fetchone()
            rows = conn.execute(
                """
                SELECT domain, fixed_fulcrum, variance, n_observations, crystallised, last_updated
                FROM domain_profiles
                WHERE user_name = ?
                ORDER BY domain
                """,
                (self.user_name,),
            ).fetchall()
        if meta is None and not rows:
            return None
        domains = {
            row["domain"]: DomainProfile(
                domain=row["domain"],
                fixed_fulcrum=float(row["fixed_fulcrum"]),
                variance=float(row["variance"]),
                n_observations=int(row["n_observations"]),
                crystallised=bool(row["crystallised"]),
                last_updated=float(row["last_updated"]),
            )
            for row in rows
        }
        cross_signals = self.bus.cross_domain_profile()
        observed = [profile for profile in domains.values() if profile.n_observations > 0]
        crystallised = [profile for profile in observed if profile.crystallised]
        risk_pool = crystallised or observed
        overall_risk = (
            sum(profile.fixed_fulcrum for profile in risk_pool) / len(risk_pool)
            if risk_pool
            else 0.5
        )
        consistency = (
            sum(max(0.0, min(1.0, 1.0 - profile.variance)) for profile in observed) / len(observed)
            if observed
            else cross_signals.get("consistency_score", 0.5)
        )
        n_total = sum(profile.n_observations for profile in domains.values())
        created_at = float(meta["created_at"]) if meta else time.time()
        last_updated = float(meta["last_updated"]) if meta else created_at
        version = int(meta["version"]) if meta else 1
        return DigitalIdentity(
            user_name=self.user_name,
            domains=domains,
            cross_signals=cross_signals,
            overall_risk=overall_risk,
            consistency=consistency,
            n_total=n_total,
            created_at=created_at,
            last_updated=last_updated,
            version=version,
        )

    def snapshot(self) -> str:
        """Serialise identity to JSON string for storage."""
        identity = self.get_identity()
        return json.dumps(identity.to_dict() if identity else {})

    def reset_domain(self, domain: str) -> None:
        """Clear a domain's profile. User-initiated — does not affect other domains."""
        now = time.time()
        with self._connect() as conn:
            conn.execute(
                """
                DELETE FROM domain_profiles
                WHERE user_name = ? AND domain = ?
                """,
                (self.user_name, domain),
            )
            conn.execute(
                """
                UPDATE identity_meta
                SET last_updated = ?, version = version + 1
                WHERE user_name = ?
                """,
                (now, self.user_name),
            )
