"""Tests for prism_documents.py — Gap Prompt 13a."""
from prism_documents import Document, PrismDocuments


def test_not_configured_when_empty():
    """PrismDocuments() with no args should have no configured providers."""
    assert PrismDocuments().configured_providers == []


def test_configured_gdrive():
    """PrismDocuments with a gdrive_token should report gdrive configured."""
    assert PrismDocuments(gdrive_token="tok").configured_providers == ["gdrive"]


def test_configured_notion():
    """PrismDocuments with a notion_token should report notion configured."""
    assert PrismDocuments(notion_token="secret_tok").configured_providers == ["notion"]


def test_configured_dropbox():
    """PrismDocuments with a dropbox_token should report dropbox configured."""
    assert PrismDocuments(dropbox_token="dbx_tok").configured_providers == ["dropbox"]


def test_configured_all_providers():
    """PrismDocuments with all tokens should list all three providers."""
    pd = PrismDocuments(gdrive_token="g", notion_token="n", dropbox_token="d")
    assert pd.configured_providers == ["gdrive", "notion", "dropbox"]


def test_notion_to_doc_extracts_title():
    """_notion_to_doc should extract title from properties and return a Document."""
    pd = PrismDocuments()
    raw = {
        "id": "page-123",
        "url": "https://notion.so/page-123",
        "last_edited_time": "2024-01-15T10:00:00.000Z",
        "object": "page",
        "properties": {
            "title": {
                "title": [
                    {"plain_text": "My Notion Page"}
                ]
            }
        },
    }
    doc = pd._notion_to_doc(raw)
    assert isinstance(doc, Document)
    assert doc.title == "My Notion Page"
    assert doc.provider == "notion"
    assert doc.doc_id == "page-123"


def test_notion_to_doc_untitled_fallback():
    """_notion_to_doc should fall back to '(untitled)' when title is missing."""
    pd = PrismDocuments()
    raw = {
        "id": "page-456",
        "url": "",
        "last_edited_time": "",
        "object": "page",
        "properties": {},
    }
    doc = pd._notion_to_doc(raw)
    assert doc.title == "(untitled)"


def test_dropbox_to_doc_extracts_name():
    """_dropbox_to_doc should map entry fields to a Document."""
    pd = PrismDocuments()
    entry = {
        ".tag": "file",
        "name": "report.pdf",
        "path_lower": "/report.pdf",
        "server_modified": "2024-01-10T09:30:00Z",
        "size": 204800,
    }
    doc = pd._dropbox_to_doc(entry)
    assert isinstance(doc, Document)
    assert doc.title == "report.pdf"
    assert doc.provider == "dropbox"
    assert doc.doc_id == "/report.pdf"
    assert doc.size_bytes == 204800
    assert doc.url.startswith("https://dropbox.com/home/")


def test_status_summary():
    """status_summary() should return a dict with 'configured' key."""
    summary = PrismDocuments().status_summary()
    assert "configured" in summary
    assert summary["configured"] == []
    assert summary["providers_available"] == 0


def test_status_summary_with_providers():
    """status_summary() with tokens should reflect configured providers."""
    pd = PrismDocuments(gdrive_token="tok", notion_token="n")
    summary = pd.status_summary()
    assert summary["providers_available"] == 2
    assert "gdrive" in summary["configured"]
    assert "notion" in summary["configured"]


def test_search_returns_list():
    """search() returns a list (empty when unconfigured)."""
    result = PrismDocuments().search("test")
    assert isinstance(result, list)
    assert result == []


def test_recent_returns_list():
    """recent() returns a list (empty when unconfigured)."""
    result = PrismDocuments().recent()
    assert isinstance(result, list)
    assert result == []


def test_create_note_returns_none_when_unconfigured():
    """create_note() returns None when no providers are configured."""
    doc = PrismDocuments().create_note("Title", "Content")
    assert doc is None


def test_from_config_reads_tokens():
    """from_config() should read tokens from the 'documents' config dict."""
    config = {
        "documents": {
            "gdrive_token": "gdrive-abc",
            "notion_token": "notion-xyz",
            "dropbox_token": "",
        }
    }
    pd = PrismDocuments.from_config(config)
    assert "gdrive" in pd.configured_providers
    assert "notion" in pd.configured_providers
    assert "dropbox" not in pd.configured_providers


def test_from_config_empty():
    """from_config() with empty config should produce no configured providers."""
    pd = PrismDocuments.from_config({})
    assert pd.configured_providers == []


def test_read_unknown_provider():
    """read() with an unknown provider returns empty string."""
    pd = PrismDocuments()
    doc = Document(doc_id="x", title="x", url="", provider="unknown")
    assert pd.read(doc) == ""
