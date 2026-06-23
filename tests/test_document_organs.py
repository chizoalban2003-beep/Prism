"""
tests/test_document_organs.py
=============================
Tests for the three document organs:
  - gdrive_search   (Google Drive v3 files API)
  - notion_query    (Notion v1 search API)
  - dropbox_fetch   (Dropbox files/search_v2 API)

All external HTTP is mocked via patching ``urllib.request.urlopen`` inside
each organ module. Covered for every organ:
  * missing token → useful setup card, no exception
  * happy path    → hits rendered with title/url/modified/snippet
  * empty result  → "no … match" message
  * malformed JSON / non-200 → "<provider> search failed: <exc>" message
  * ORGAN_META / ORGAN_POLICY shape
"""
from __future__ import annotations

import importlib.util
import io
import json
from unittest.mock import patch


def _load(organ_name: str):
    spec = importlib.util.spec_from_file_location(
        organ_name,
        f"organs/{organ_name}.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _FakeResponse:
    """Minimal context-manager wrapper that mimics urllib's HTTPResponse."""

    def __init__(self, body_bytes: bytes):
        self._buf = io.BytesIO(body_bytes)

    def read(self):
        return self._buf.read()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def _ok(payload: dict) -> _FakeResponse:
    return _FakeResponse(json.dumps(payload).encode("utf-8"))


# ── gdrive_search ─────────────────────────────────────────────────────────────

class TestGDriveSearch:
    organ = _load("gdrive_search")

    def test_no_token(self):
        card = self.organ.execute("gdrive_search", "search my drive for Q3 plan", {})
        assert "google drive token" in card.body.lower()
        assert "gdrive_token" in card.body.lower()

    def test_happy_path(self):
        ctx = {"documents_config": {"gdrive_token": "ya29.test"}}
        payload = {
            "files": [
                {
                    "id":           "1abc",
                    "name":         "Q3 plan.docx",
                    "modifiedTime": "2026-06-20T10:00:00Z",
                    "webViewLink":  "https://drive.google.com/file/d/1abc/view",
                    "mimeType":     "application/vnd.google-apps.document",
                },
                {
                    "id":           "2def",
                    "name":         "Q3 plan v2.docx",
                    "modifiedTime": "2026-06-21T11:00:00Z",
                    "webViewLink":  "https://drive.google.com/file/d/2def/view",
                    "mimeType":     "application/vnd.google-apps.document",
                },
            ]
        }
        with patch("urllib.request.urlopen", return_value=_ok(payload)):
            card = self.organ.execute("gdrive_search", "search my drive for Q3 plan", ctx)
        assert "q3 plan" in card.body.lower()
        assert "drive.google.com/file/d/1abc" in card.body
        assert "drive.google.com/file/d/2def" in card.body
        assert "2026-06-20" in card.body

    def test_empty_result(self):
        ctx = {"documents_config": {"gdrive_token": "ya29.test"}}
        with patch("urllib.request.urlopen", return_value=_ok({"files": []})):
            card = self.organ.execute("gdrive_search", "drive: nonexistent thing", ctx)
        assert "no drive files match" in card.body.lower()

    def test_network_failure(self):
        ctx = {"documents_config": {"gdrive_token": "ya29.test"}}
        with patch("urllib.request.urlopen", side_effect=Exception("boom")):
            card = self.organ.execute("gdrive_search", "drive: anything", ctx)
        assert "drive search failed" in card.body.lower()
        assert "boom" in card.body.lower()

    def test_malformed_json(self):
        ctx = {"documents_config": {"gdrive_token": "ya29.test"}}
        bad = _FakeResponse(b"not json {{{")
        with patch("urllib.request.urlopen", return_value=bad):
            card = self.organ.execute("gdrive_search", "drive: foo", ctx)
        assert "failed" in card.body.lower()

    def test_organ_meta(self):
        assert self.organ.ORGAN_META["intent"] == "gdrive_search"
        assert self.organ.ORGAN_META["capabilities"] == ["internet_read"]
        assert "query" in self.organ.ORGAN_META["inputs"]
        assert self.organ.ORGAN_POLICY["risk_level"] == "low"
        assert self.organ.ORGAN_POLICY["requires_approval"] is False
        assert self.organ.ORGAN_POLICY["irreversible"] is False


# ── notion_query ──────────────────────────────────────────────────────────────

class TestNotionQuery:
    organ = _load("notion_query")

    def test_no_token(self):
        card = self.organ.execute("notion_query", "notion: meeting notes", {})
        assert "notion token" in card.body.lower()
        assert "notion_token" in card.body.lower()

    def test_happy_path(self):
        ctx = {"documents_config": {"notion_token": "secret_test"}}
        payload = {
            "results": [
                {
                    "object":           "page",
                    "id":               "page-1",
                    "url":              "https://www.notion.so/page-1",
                    "last_edited_time": "2026-06-22T09:00:00.000Z",
                    "properties": {
                        "Name": {
                            "type":  "title",
                            "title": [{"plain_text": "Q3 Planning Notes"}],
                        }
                    },
                },
                {
                    "object":           "database",
                    "id":               "db-2",
                    "url":              "https://www.notion.so/db-2",
                    "last_edited_time": "2026-06-23T09:00:00.000Z",
                    "title":            [{"plain_text": "Meetings DB"}],
                },
            ]
        }
        with patch("urllib.request.urlopen", return_value=_ok(payload)):
            card = self.organ.execute("notion_query", "notion: Q3 planning", ctx)
        assert "q3 planning notes" in card.body.lower()
        assert "meetings db" in card.body.lower()
        assert "notion.so/page-1" in card.body
        assert "notion.so/db-2" in card.body

    def test_empty_result(self):
        ctx = {"documents_config": {"notion_token": "secret_test"}}
        with patch("urllib.request.urlopen", return_value=_ok({"results": []})):
            card = self.organ.execute("notion_query", "notion: nothing here", ctx)
        assert "no notion pages match" in card.body.lower()

    def test_network_failure(self):
        ctx = {"documents_config": {"notion_token": "secret_test"}}
        with patch("urllib.request.urlopen", side_effect=Exception("timeout")):
            card = self.organ.execute("notion_query", "notion: anything", ctx)
        assert "notion search failed" in card.body.lower()
        assert "timeout" in card.body.lower()

    def test_untitled_page(self):
        ctx = {"documents_config": {"notion_token": "secret_test"}}
        payload = {
            "results": [
                {
                    "object":           "page",
                    "id":               "page-x",
                    "url":              "https://www.notion.so/page-x",
                    "last_edited_time": "2026-06-22T09:00:00.000Z",
                    "properties":       {},
                }
            ]
        }
        with patch("urllib.request.urlopen", return_value=_ok(payload)):
            card = self.organ.execute("notion_query", "notion: x", ctx)
        assert "(untitled)" in card.body

    def test_organ_meta(self):
        assert self.organ.ORGAN_META["intent"] == "notion_query"
        assert self.organ.ORGAN_META["capabilities"] == ["internet_read"]
        assert "query" in self.organ.ORGAN_META["inputs"]
        assert self.organ.ORGAN_POLICY["risk_level"] == "low"
        assert self.organ.ORGAN_POLICY["requires_approval"] is False


# ── dropbox_fetch ─────────────────────────────────────────────────────────────

class TestDropboxFetch:
    organ = _load("dropbox_fetch")

    def test_no_token(self):
        card = self.organ.execute("dropbox_fetch", "dropbox: budget", {})
        assert "dropbox token" in card.body.lower()
        assert "dropbox_token" in card.body.lower()

    def test_happy_path(self):
        ctx = {"documents_config": {"dropbox_token": "sl.test"}}
        payload = {
            "matches": [
                {
                    "metadata": {
                        ".tag":     "metadata",
                        "metadata": {
                            ".tag":            "file",
                            "name":            "budget_2026.xlsx",
                            "path_display":    "/Finance/budget_2026.xlsx",
                            "server_modified": "2026-06-15T08:00:00Z",
                            "size":            48210,
                        },
                    }
                }
            ]
        }
        with patch("urllib.request.urlopen", return_value=_ok(payload)):
            card = self.organ.execute("dropbox_fetch", "dropbox: budget", ctx)
        assert "budget_2026.xlsx" in card.body
        assert "/Finance/budget_2026.xlsx" in card.body
        assert "48210" in card.body

    def test_empty_result(self):
        ctx = {"documents_config": {"dropbox_token": "sl.test"}}
        with patch("urllib.request.urlopen", return_value=_ok({"matches": []})):
            card = self.organ.execute("dropbox_fetch", "dropbox: nada", ctx)
        assert "no dropbox files match" in card.body.lower()

    def test_network_failure(self):
        ctx = {"documents_config": {"dropbox_token": "sl.test"}}
        with patch("urllib.request.urlopen", side_effect=Exception("401")):
            card = self.organ.execute("dropbox_fetch", "dropbox: x", ctx)
        assert "dropbox search failed" in card.body.lower()
        assert "401" in card.body

    def test_organ_meta(self):
        assert self.organ.ORGAN_META["intent"] == "dropbox_fetch"
        assert self.organ.ORGAN_META["capabilities"] == ["internet_read"]
        assert "query" in self.organ.ORGAN_META["inputs"]
        assert self.organ.ORGAN_POLICY["risk_level"] == "low"
        assert self.organ.ORGAN_POLICY["requires_approval"] is False
