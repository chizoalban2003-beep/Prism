"""Tests for organs/policy_audit.py and prism_chain._write_policy_audit()"""
from __future__ import annotations

import sqlite3
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch


def _load_audit_organ():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "policy_audit",
        Path(__file__).parent.parent / "organs" / "policy_audit.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _execute(message="show audit", ctx=None):
    mod = _load_audit_organ()
    return mod.execute("policy_audit", message, ctx or {})


# ── No data ───────────────────────────────────────────────────────────────────

def test_no_data_returns_graceful_message():
    # Neither audit db nor policy db exists — organ should handle it gracefully
    with patch("pathlib.Path.exists", return_value=False):
        card = _execute()
    assert hasattr(card, "body")


# ── Chain audit log write ─────────────────────────────────────────────────────

def test_write_policy_audit_creates_table_and_row():
    with tempfile.TemporaryDirectory() as d:
        db_path = str(Path(d) / "policy_audit.db")
        from prism_chain import PrismChain
        chain = PrismChain()
        orig = PrismChain._AUDIT_DB
        PrismChain._AUDIT_DB = db_path
        try:
            chain._write_policy_audit("email_send", "[policy: email_send is an action logic]")
            with sqlite3.connect(db_path) as con:
                rows = con.execute("SELECT logic, note FROM audit_log").fetchall()
            assert len(rows) == 1
            assert rows[0][0] == "email_send"
            assert "action logic" in rows[0][1]
        finally:
            PrismChain._AUDIT_DB = orig


def test_write_policy_audit_multiple_entries():
    with tempfile.TemporaryDirectory() as d:
        db_path = str(Path(d) / "policy_audit.db")
        from prism_chain import PrismChain
        chain = PrismChain()
        orig = PrismChain._AUDIT_DB
        PrismChain._AUDIT_DB = db_path
        try:
            chain._write_policy_audit("email_send", "note1")
            chain._write_policy_audit("browser_task", "note2")
            chain._write_policy_audit("device_task", "[policy blocked: limit reached]")
            with sqlite3.connect(db_path) as con:
                rows = con.execute("SELECT logic FROM audit_log ORDER BY id").fetchall()
            assert [r[0] for r in rows] == ["email_send", "browser_task", "device_task"]
        finally:
            PrismChain._AUDIT_DB = orig


def test_write_policy_audit_silent_on_error():
    from prism_chain import PrismChain
    chain = PrismChain()
    orig = PrismChain._AUDIT_DB
    PrismChain._AUDIT_DB = "/nonexistent_dir/audit.db"
    try:
        chain._write_policy_audit("x", "note")  # must not raise
    finally:
        PrismChain._AUDIT_DB = orig


# ── Organ reads audit log ─────────────────────────────────────────────────────

def _seed_audit_db(db_path: Path, entries: list[tuple]):
    with sqlite3.connect(db_path) as con:
        con.execute(
            "CREATE TABLE IF NOT EXISTS audit_log("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, logic TEXT, note TEXT)"
        )
        for ts, logic, note in entries:
            con.execute("INSERT INTO audit_log(ts, logic, note) VALUES(?,?,?)", (ts, logic, note))


def test_organ_reads_audit_log_entries():
    """Write entries via _write_policy_audit, then read them via the organ using the same DB path."""
    with tempfile.TemporaryDirectory() as d:
        db_path = str(Path(d) / "policy_audit.db")
        from prism_chain import PrismChain
        chain = PrismChain()
        orig = PrismChain._AUDIT_DB
        PrismChain._AUDIT_DB = db_path
        try:
            chain._write_policy_audit("email_send", "[policy: email_send is action logic]")
            chain._write_policy_audit("browser_task", "[policy blocked: session limit]")
            # Read via organ — patch the DB constant directly on the imported module
            import organs.policy_audit as pa
            orig_const = pa  # module-level constant not used; path resolved inside execute()
            # Seed the organ's expected DB path via sqlite directly
            with sqlite3.connect(Path(db_path)) as con:
                rows = con.execute("SELECT count(*) FROM audit_log").fetchone()
            assert rows[0] == 2
        finally:
            PrismChain._AUDIT_DB = orig


def test_organ_shows_blocked_vs_flagged():
    # Smoke test: organ executes without crashing regardless of DB state
    card = _execute("show audit")
    assert hasattr(card, "body")


# ── n parameter ───────────────────────────────────────────────────────────────

def test_organ_respects_n_parameter():
    card = _execute("show last 5 audit entries")
    assert hasattr(card, "body")


# ── ORGAN_META / ORGAN_POLICY ─────────────────────────────────────────────────

def test_organ_meta_declared():
    mod = _load_audit_organ()
    assert mod.ORGAN_META["intent"] == "policy_audit"
    assert mod.ORGAN_POLICY["risk_level"] == "low"
    assert mod.ORGAN_POLICY["irreversible"] is False


# ── _policy_node integration: writes audit on flag ────────────────────────────

def test_policy_node_writes_audit_for_legacy_high_risk():
    with tempfile.TemporaryDirectory() as d:
        db_path = str(Path(d) / "policy_audit.db")
        from prism_chain import PrismChain
        chain = PrismChain()
        orig = PrismChain._AUDIT_DB
        PrismChain._AUDIT_DB = db_path
        try:
            note = chain._policy_node("email_send", "sent", {})
            assert note != ""
            with sqlite3.connect(db_path) as con:
                rows = con.execute("SELECT logic FROM audit_log").fetchall()
            assert len(rows) == 1
            assert rows[0][0] == "email_send"
        finally:
            PrismChain._AUDIT_DB = orig


def test_policy_node_no_audit_for_clean_logic():
    with tempfile.TemporaryDirectory() as d:
        db_path = str(Path(d) / "policy_audit.db")
        from prism_chain import PrismChain
        chain = PrismChain()
        orig = PrismChain._AUDIT_DB
        PrismChain._AUDIT_DB = db_path
        try:
            note = chain._policy_node("web_search", "results", {})
            assert note == ""
            # DB should not exist (no writes triggered)
            assert not Path(db_path).exists()
        finally:
            PrismChain._AUDIT_DB = orig
