from __future__ import annotations
import json, logging, sqlite3, time, urllib.parse, urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

@dataclass
class Contact:
    contact_id:   str
    name:         str
    emails:       list[str] = field(default_factory=list)
    phones:       list[str] = field(default_factory=list)
    organisation: str = ""
    role:         str = ""
    notes:        str = ""
    tags:         list[str] = field(default_factory=list)
    last_contacted:str = ""
    source:       str = "local"

class PrismContacts:
    """
    Structured contact management.

    Config:
      [contacts]
      google_token = ""
      auto_extract = true
    """

    def __init__(self, db_path="~/.prism/contacts.db",
                  google_token="", auto_extract=True):
        self._db         = Path(db_path).expanduser()
        self._db.parent.mkdir(parents=True, exist_ok=True)
        self._google     = google_token
        self._auto       = auto_extract
        self._init_db()

    @classmethod
    def from_config(cls, config: dict) -> "PrismContacts":
        c = config.get("contacts", {})
        return cls(
            google_token = c.get("google_token",""),
            auto_extract = c.get("auto_extract", True),
        )

    def add(self, contact: Contact) -> str:
        if not contact.contact_id:
            import hashlib
            contact.contact_id = hashlib.sha256(
                contact.name.encode()).hexdigest()[:10]
        with sqlite3.connect(self._db) as c:
            c.execute("""INSERT OR REPLACE INTO contacts
                VALUES(?,?,?,?,?,?,?,?,?,?)""", (
                contact.contact_id, contact.name,
                json.dumps(contact.emails),
                json.dumps(contact.phones),
                contact.organisation, contact.role,
                contact.notes, json.dumps(contact.tags),
                contact.last_contacted, contact.source))
        return contact.contact_id

    def search(self, query: str) -> list[Contact]:
        q = f"%{query.lower()}%"
        with sqlite3.connect(self._db) as c:
            rows = c.execute("""
                SELECT * FROM contacts
                WHERE lower(name) LIKE ?
                   OR lower(organisation) LIKE ?
                   OR lower(role) LIKE ?
                   OR lower(notes) LIKE ?
                ORDER BY name""",
                (q, q, q, q)).fetchall()
        return [self._row_to_contact(r) for r in rows]

    def get(self, name_or_id: str) -> Optional[Contact]:
        results = self.search(name_or_id)
        return results[0] if results else None

    def all_contacts(self) -> list[Contact]:
        with sqlite3.connect(self._db) as c:
            rows = c.execute(
                "SELECT * FROM contacts ORDER BY name").fetchall()
        return [self._row_to_contact(r) for r in rows]

    def update_last_contacted(self, contact_id: str) -> None:
        with sqlite3.connect(self._db) as c:
            c.execute("UPDATE contacts SET last_contacted=? WHERE id=?",
                      (time.strftime("%Y-%m-%d"), contact_id))

    def sync_google(self) -> int:
        if not self._google:
            return 0
        url = ("https://people.googleapis.com/v1/people/me/connections"
               "?personFields=names,emailAddresses,phoneNumbers,"
               "organizations&pageSize=200")
        req = urllib.request.Request(url, headers={
            "Authorization": f"Bearer {self._google}"})
        try:
            resp = urllib.request.urlopen(req, timeout=10)
            data = json.loads(resp.read())
            count = 0
            for person in data.get("connections",[]):
                names  = person.get("names",[{}])
                name   = names[0].get("displayName","") if names else ""
                if not name: continue
                emails = [e["value"] for e in
                          person.get("emailAddresses",[]) if "value" in e]
                phones = [p["value"] for p in
                          person.get("phoneNumbers",[]) if "value" in p]
                orgs   = person.get("organizations",[{}])
                org    = orgs[0].get("name","") if orgs else ""
                role   = orgs[0].get("title","") if orgs else ""
                cid    = person.get("resourceName","").replace("/","_")
                self.add(Contact(cid, name, emails, phones,
                                  org, role, source="google"))
                count += 1
            return count
        except Exception as e:
            logger.warning("Google Contacts sync failed: %s", e)
            return 0

    def extract_from_memory(self, memory, llm_router=None) -> int:
        if not memory or not llm_router:
            return 0
        results = memory.search("email from meeting with", top_n=20)
        count   = 0
        for r in results:
            prompt = (
                f"Extract people mentioned in this text as structured contacts. "
                f"Text: {r.excerpt[:500]}\n"
                f"Return JSON array: "
                f'[{{"name":"","email":"","organisation":"","role":""}}]'
                f" Return empty array if no people found.")
            raw, _ = llm_router.call(
                prompt, min_capability=1, max_tokens=300, json_mode=True)
            try:
                import re as _re
                clean = raw.strip().lstrip("```json").rstrip("```").strip()
                people = json.loads(clean)
                for p in (people or []):
                    if p.get("name") and len(p["name"]) > 2:
                        existing = self.get(p["name"])
                        if not existing:
                            self.add(Contact(
                                contact_id   = "",
                                name         = p["name"],
                                emails       = [p["email"]] if p.get("email") else [],
                                organisation = p.get("organisation",""),
                                role         = p.get("role",""),
                                source       = "prism_memory",
                            ))
                            count += 1
            except Exception:
                pass
        return count

    def _row_to_contact(self, row) -> Contact:
        return Contact(
            contact_id    = row[0], name=row[1],
            emails        = json.loads(row[2]),
            phones        = json.loads(row[3]),
            organisation  = row[4], role=row[5],
            notes         = row[6],
            tags          = json.loads(row[7]),
            last_contacted= row[8], source=row[9])

    def _init_db(self) -> None:
        with sqlite3.connect(self._db) as c:
            c.execute("""CREATE TABLE IF NOT EXISTS contacts(
                id TEXT PRIMARY KEY, name TEXT, emails_json TEXT,
                phones_json TEXT, organisation TEXT, role TEXT,
                notes TEXT, tags_json TEXT, last_contacted TEXT,
                source TEXT)""")
            c.execute("CREATE INDEX IF NOT EXISTS ix_name "
                      "ON contacts(name)")
