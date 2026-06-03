"""Bundled organ: policy_audit — audit log of every policy flag and spend decision."""
ORGAN_META = {
    "intent": "policy_audit",
    "description": "Show a time-ordered audit log of all policy flags, blocked actions, and spend approvals",
    "version": "1.0",
}

ORGAN_POLICY = {
    "risk_level": "low",
    "requires_approval": False,
    "irreversible": False,
    "max_per_session": None,
}


def execute(intent: str, message: str, ctx: dict):
    import re
    import sqlite3
    from datetime import datetime
    from pathlib import Path

    from prism_responses import text_card

    # Parse how many rows the user wants
    n_match = re.search(r"\b(\d+)\b", message)
    n = min(int(n_match.group(1)), 100) if n_match else 30

    lines: list[str] = []

    # ── Section 1: chain-level policy flags ──────────────────────────────────
    audit_db = Path("~/.prism/policy_audit.db").expanduser()
    if audit_db.exists():
        try:
            with sqlite3.connect(audit_db) as con:
                rows = con.execute(
                    "SELECT ts, logic, note FROM audit_log ORDER BY ts DESC LIMIT ?",
                    (n,),
                ).fetchall()
            if rows:
                lines.append(f"Chain policy flags (last {len(rows)}):")
                for ts, logic, note in rows:
                    dt = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")
                    flag = "BLOCKED" if "blocked" in note.lower() else "flagged"
                    lines.append(f"  {dt}  [{flag}]  {logic}  —  {note[:120]}")
        except Exception as exc:
            lines.append(f"  (could not read chain audit log: {exc})")
    else:
        lines.append("Chain policy flags: none recorded yet.")

    # ── Section 2: spend log from PolicyEngine ────────────────────────────────
    policy_db = Path("~/.prism/policy.db").expanduser()
    if policy_db.exists():
        try:
            with sqlite3.connect(policy_db) as con:
                # spend_log may not exist in all installs
                tables = {r[0] for r in con.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()}
                if "spend_log" in tables:
                    rows = con.execute(
                        "SELECT ts, user, category, provider, amount, approved "
                        "FROM spend_log ORDER BY ts DESC LIMIT ?",
                        (n,),
                    ).fetchall()
                    if rows:
                        lines.append(f"\nSpend decisions (last {len(rows)}):")
                        for ts, user, cat, prov, amt, approved in rows:
                            dt = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")
                            verdict = "approved" if approved else "DENIED"
                            lines.append(
                                f"  {dt}  [{verdict}]  {user}/{cat}  "
                                f"{prov}  £{amt:.2f}"
                            )
                    else:
                        lines.append("\nSpend decisions: none recorded.")
        except Exception as exc:
            lines.append(f"\n(could not read spend log: {exc})")

    # ── Section 3: summary stats ──────────────────────────────────────────────
    if audit_db.exists():
        try:
            with sqlite3.connect(audit_db) as con:
                total, blocked = con.execute(
                    "SELECT COUNT(*), SUM(CASE WHEN lower(note) LIKE '%blocked%' THEN 1 ELSE 0 END) "
                    "FROM audit_log"
                ).fetchone()
            blocked = blocked or 0
            lines.append(f"\nSummary: {total} policy event(s) total, {blocked} blocked.")
        except Exception:
            pass

    if not lines:
        return text_card("No policy audit data found.", intent)

    return text_card("\n".join(lines), intent)
