import tempfile, os
from prism_contacts import PrismContacts, Contact

def _tmp_contacts():
    tmp = tempfile.mktemp(suffix=".db")
    return PrismContacts(db_path=tmp)

def test_add_and_search():
    c = _tmp_contacts()
    c.add(Contact("", "Alice Smith", emails=["alice@example.com"]))
    results = c.search("alice")
    assert results and results[0].name == "Alice Smith"

def test_search_by_org():
    c = _tmp_contacts()
    c.add(Contact("", "Bob Jones", organisation="Acme Corp"))
    results = c.search("acme")
    assert results and results[0].organisation == "Acme Corp"

def test_get_returns_first():
    c = _tmp_contacts()
    c.add(Contact("", "Carol White"))
    result = c.get("carol")
    assert result is not None and result.name == "Carol White"

def test_source_defaults_local():
    contact = Contact("", "Dave")
    assert contact.source == "local"

def test_init_creates_db():
    import tempfile
    tmp = tempfile.mktemp(suffix=".db")
    PrismContacts(db_path=tmp)
    assert os.path.exists(tmp)
    os.unlink(tmp)
