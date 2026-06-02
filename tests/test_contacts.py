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


def test_get_returns_first(contacts):
    """get() returns a Contact when found, or None when not found."""
    contacts.add(Contact(contact_id="", name="Charlie"))
    result = contacts.get("Charlie")
    assert result is not None
    assert result.name == "Charlie"

    missing = contacts.get("nobody_xyz_999")
    assert missing is None


def test_init_creates_db(tmp_path):
    """PrismContacts() creates the db file."""
    db_file = tmp_path / "sub" / "contacts.db"
    PrismContacts(db_path=str(db_file))
    assert db_file.exists()
