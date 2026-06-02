from __future__ import annotations
import json, logging, os, urllib.request, urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

@dataclass
class Document:
    doc_id:    str
    title:     str
    url:       str
    provider:  str      # "gdrive"|"notion"|"dropbox"
    mime_type: str = ""
    modified:  str = ""
    content:   str = ""   # plain text excerpt
    size_bytes:int = 0

class PrismDocuments:
    """
    Unified document management across Google Drive, Notion, and Dropbox.
    Each provider is optional — configure only what you use.

    Config in prism_config.toml:
      [documents]
      # Google Drive — create service account at console.cloud.google.com
      # OR use OAuth2 token from https://developers.google.com/drive/api
      gdrive_token       = ""   # OAuth2 access token
      gdrive_credentials = ""   # path to credentials.json

      # Notion — create integration at notion.so/my-integrations
      notion_token       = ""   # "secret_..."

      # Dropbox — create app at dropbox.com/developers/apps
      dropbox_token      = ""   # long-lived access token
    """

    def __init__(self, gdrive_token="", notion_token="", dropbox_token=""):
        self._gdrive   = gdrive_token
        self._notion   = notion_token
        self._dropbox  = dropbox_token

    @classmethod
    def from_config(cls, config: dict) -> "PrismDocuments":
        d = config.get("documents", {})
        return cls(
            gdrive_token  = d.get("gdrive_token",""),
            notion_token  = d.get("notion_token",""),
            dropbox_token = d.get("dropbox_token",""),
        )

    @property
    def configured_providers(self) -> list[str]:
        providers = []
        if self._gdrive:   providers.append("gdrive")
        if self._notion:   providers.append("notion")
        if self._dropbox:  providers.append("dropbox")
        return providers

    # ── Search across all configured providers ────────────────────────────

    def search(self, query: str, n: int = 10) -> list[Document]:
        """Search documents across all configured providers."""
        results = []
        if self._gdrive:
            results.extend(self._gdrive_search(query, n))
        if self._notion:
            results.extend(self._notion_search(query, n))
        if self._dropbox:
            results.extend(self._dropbox_search(query, n))
        return results[:n]

    def recent(self, n: int = 10) -> list[Document]:
        """Get recently modified documents."""
        results = []
        if self._gdrive:   results.extend(self._gdrive_recent(n))
        if self._notion:   results.extend(self._notion_recent(n))
        if self._dropbox:  results.extend(self._dropbox_recent(n))
        results.sort(key=lambda d: d.modified, reverse=True)
        return results[:n]

    def read(self, doc: Document) -> str:
        """Read the content of a document."""
        if doc.provider == "gdrive":   return self._gdrive_read(doc)
        if doc.provider == "notion":   return self._notion_read(doc)
        if doc.provider == "dropbox":  return self._dropbox_read(doc)
        return ""

    def create_note(self, title: str, content: str,
                     provider: str = None) -> Optional[Document]:
        """Create a new note/page in the preferred provider."""
        target = provider or (self.configured_providers[0]
                               if self.configured_providers else None)
        if not target: return None
        if target == "notion":  return self._notion_create(title, content)
        if target == "gdrive":  return self._gdrive_create(title, content)
        return None

    # ── Google Drive ─────────────────────────────────────────────────────

    def _gdrive_search(self, query: str, n: int) -> list[Document]:
        q   = urllib.parse.quote(f"fullText contains '{query}' and trashed=false")
        url = (f"https://www.googleapis.com/drive/v3/files"
               f"?q={q}&pageSize={n}"
               f"&fields=files(id,name,webViewLink,mimeType,modifiedTime,size)")
        data = self._gdrive_get(url)
        return [Document(
            doc_id   = f["id"],
            title    = f["name"],
            url      = f.get("webViewLink",""),
            provider = "gdrive",
            mime_type= f.get("mimeType",""),
            modified = f.get("modifiedTime",""),
            size_bytes= int(f.get("size",0)),
        ) for f in data.get("files",[])]

    def _gdrive_recent(self, n: int) -> list[Document]:
        url = (f"https://www.googleapis.com/drive/v3/files"
               f"?orderBy=modifiedTime+desc&pageSize={n}"
               f"&q=trashed%3Dfalse"
               f"&fields=files(id,name,webViewLink,mimeType,modifiedTime)")
        data = self._gdrive_get(url)
        return [Document(
            doc_id=f["id"], title=f["name"],
            url=f.get("webViewLink",""), provider="gdrive",
            modified=f.get("modifiedTime","")
        ) for f in data.get("files",[])]

    def _gdrive_read(self, doc: Document) -> str:
        if "document" in doc.mime_type or "text" in doc.mime_type:
            url  = (f"https://www.googleapis.com/drive/v3/files"
                    f"/{doc.doc_id}/export?mimeType=text%2Fplain")
            data = self._gdrive_get(url, raw=True)
            return data[:5000] if isinstance(data, str) else ""
        return f"[Binary file: {doc.title}]"

    def _gdrive_create(self, title: str, content: str) -> Optional[Document]:
        metadata = json.dumps({"name": title,
                                "mimeType": "application/vnd.google-apps.document"}).encode()
        req = urllib.request.Request(
            "https://www.googleapis.com/drive/v3/files",
            data=metadata,
            headers={"Authorization": f"Bearer {self._gdrive}",
                     "Content-Type": "application/json"})
        try:
            resp = urllib.request.urlopen(req, timeout=10)
            d    = json.loads(resp.read())
            return Document(d["id"], title, d.get("webViewLink",""),
                             "gdrive")
        except Exception as e:
            logger.warning("GDrive create failed: %s", e)
            return None

    def _gdrive_get(self, url: str, raw: bool = False):
        req = urllib.request.Request(url, headers={
            "Authorization": f"Bearer {self._gdrive}"})
        try:
            resp = urllib.request.urlopen(req, timeout=10)
            data = resp.read()
            return data.decode(errors="replace") if raw else json.loads(data)
        except Exception as e:
            logger.debug("GDrive GET failed: %s", e)
            return {} if not raw else ""

    # ── Notion ────────────────────────────────────────────────────────────

    def _notion_search(self, query: str, n: int) -> list[Document]:
        payload = json.dumps({"query": query,
                               "page_size": n}).encode()
        data    = self._notion_post("https://api.notion.com/v1/search", payload)
        return [self._notion_to_doc(r)
                for r in data.get("results",[])
                if r.get("object") in ("page","database")][:n]

    def _notion_recent(self, n: int) -> list[Document]:
        payload = json.dumps({"sort":{"direction":"descending",
                                       "timestamp":"last_edited_time"},
                               "page_size": n}).encode()
        data    = self._notion_post("https://api.notion.com/v1/search", payload)
        return [self._notion_to_doc(r)
                for r in data.get("results",[])
                if r.get("object")=="page"][:n]

    def _notion_read(self, doc: Document) -> str:
        data  = self._notion_get(
            f"https://api.notion.com/v1/blocks/{doc.doc_id}/children")
        lines = []
        for block in data.get("results",[])[:50]:
            bt    = block.get("type","")
            inner = block.get(bt, {})
            texts = inner.get("rich_text",[]) if isinstance(inner, dict) else []
            line  = "".join(t.get("plain_text","") for t in texts)
            if line: lines.append(line)
        return "\n".join(lines)[:5000]

    def _notion_create(self, title: str, content: str) -> Optional[Document]:
        payload = json.dumps({
            "parent": {"type":"page_id","page_id":"root"},
            "properties": {"title":{"title":[{"text":{"content":title}}]}},
            "children": [{"object":"block","type":"paragraph",
                           "paragraph":{"rich_text":[{"text":{"content":content[:2000]}}]}}]
        }).encode()
        data = self._notion_post("https://api.notion.com/v1/pages", payload)
        if data.get("id"):
            return Document(data["id"], title,
                             data.get("url",""), "notion")
        return None

    def _notion_to_doc(self, r: dict) -> Document:
        props = r.get("properties",{})
        title = ""
        for p in props.values():
            texts = p.get("title",[])
            if texts:
                title = "".join(t.get("plain_text","") for t in texts)
                break
        return Document(doc_id=r["id"], title=title or "(untitled)",
                         url=r.get("url",""), provider="notion",
                         modified=r.get("last_edited_time",""))

    def _notion_get(self, url: str) -> dict:
        req = urllib.request.Request(url, headers={
            "Authorization": f"Bearer {self._notion}",
            "Notion-Version": "2022-06-28"})
        try:
            resp = urllib.request.urlopen(req, timeout=10)
            return json.loads(resp.read())
        except Exception as e:
            logger.debug("Notion GET failed: %s", e)
            return {}

    def _notion_post(self, url: str, payload: bytes) -> dict:
        req = urllib.request.Request(url, data=payload, headers={
            "Authorization": f"Bearer {self._notion}",
            "Notion-Version": "2022-06-28",
            "Content-Type": "application/json"})
        try:
            resp = urllib.request.urlopen(req, timeout=10)
            return json.loads(resp.read())
        except Exception as e:
            logger.debug("Notion POST failed: %s", e)
            return {}

    # ── Dropbox ───────────────────────────────────────────────────────────

    def _dropbox_search(self, query: str, n: int) -> list[Document]:
        payload = json.dumps({"query": query,
                               "options": {"max_results": n}}).encode()
        req = urllib.request.Request(
            "https://api.dropboxapi.com/2/files/search_v2",
            data=payload,
            headers={"Authorization": f"Bearer {self._dropbox}",
                     "Content-Type": "application/json"})
        try:
            resp = urllib.request.urlopen(req, timeout=10)
            data = json.loads(resp.read())
            return [self._dropbox_to_doc(m["metadata"]["metadata"])
                    for m in data.get("matches",[])
                    if m.get("metadata",{}).get("metadata",{}).get(".tag")=="file"]
        except Exception as e:
            logger.debug("Dropbox search failed: %s", e)
            return []

    def _dropbox_recent(self, n: int) -> list[Document]:
        payload = json.dumps({"path":"","recursive":False,
                               "limit":n}).encode()
        req = urllib.request.Request(
            "https://api.dropboxapi.com/2/files/list_folder",
            data=payload,
            headers={"Authorization": f"Bearer {self._dropbox}",
                     "Content-Type": "application/json"})
        try:
            resp = urllib.request.urlopen(req, timeout=10)
            data = json.loads(resp.read())
            return [self._dropbox_to_doc(e)
                    for e in data.get("entries",[])
                    if e.get(".tag")=="file"][:n]
        except Exception as e:
            logger.debug("Dropbox list failed: %s", e)
            return []

    def _dropbox_read(self, doc: Document) -> str:
        arg = json.dumps({"path": doc.doc_id})
        req = urllib.request.Request(
            "https://content.dropboxapi.com/2/files/download",
            headers={"Authorization": f"Bearer {self._dropbox}",
                     "Dropbox-API-Arg": arg})
        try:
            resp = urllib.request.urlopen(req, timeout=10)
            return resp.read().decode(errors="replace")[:5000]
        except Exception as e:
            logger.debug("Dropbox read failed: %s", e)
            return ""

    def _dropbox_to_doc(self, e: dict) -> Document:
        return Document(doc_id=e.get("path_lower",""),
                         title=e.get("name",""),
                         url=f"https://dropbox.com/home{e.get('path_lower','')}",
                         provider="dropbox",
                         modified=e.get("server_modified",""),
                         size_bytes=e.get("size",0))

    def status_summary(self) -> dict:
        return {"configured": self.configured_providers,
                "providers_available": len(self.configured_providers)}
