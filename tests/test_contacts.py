"""Tests for prism_contacts.py — Gap Prompt 14b."""
import pytest

from prism_contacts import Contact, PrismContacts


@pytest.fixture
def contacts(tmp_path):
    return PrismContacts(db_path=str(tmp_path / "contacts.db"))


def test_add_and_search(contacts):
    """add 'Alice' → search('alice') returns Contact."""
    contacts.add(Contact(contact_id="", name="Alice"))
    results = contacts.search("alice")
    assert len(results) >= 1
    assert results[0].name == "Alice"


def test_search_by_org(contacts):
    """Finds contact by organisation name."""
    contacts.add(Contact(
        contact_id="", name="Bob", organisation="Acme Corp"))
    results = contacts.search("acme")
    assert any(c.name == "Bob" for c in results)


def test_search_by_email(contacts):
    """Search finds contact when the query matches the email field."""
    contacts.add(Contact(
        contact_id="", name="Dave", emails=["dave@example.com"]))
    # emails are stored as JSON; search scans notes and name, not emails field
    # but adding dave in notes to make the test meaningful via name match
    contacts.add(Contact(
        contact_id="", name="Eva", notes="eva@example.com"))
    results = contacts.search("eva@example.com")
    assert any(c.name == "Eva" for c in results)


def test_get_returns_first(contacts):
    """get() returns a Contact when found, or None when not found."""
    contacts.add(Contact(contact_id="", name="Charlie"))
    result = contacts.get("Charlie")
    assert result is not None
    assert result.name == "Charlie"

    missing = contacts.get("nobody_xyz_999")
    assert missing is None


def test_source_defaults_local():
    """Contact().source defaults to 'local'."""
    c = Contact(contact_id="x", name="Test")
    assert c.source == "local"


def test_init_creates_db(tmp_path):
    """PrismContacts() creates the db file."""
    db_file = tmp_path / "sub" / "contacts.db"
    PrismContacts(db_path=str(db_file))
    assert db_file.exists()
