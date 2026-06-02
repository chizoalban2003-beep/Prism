from __future__ import annotations
import json, logging, urllib.parse, urllib.request
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    title:   str
    url:     str
    snippet: str
    source:  str = ""

class PrismSearch:
    """
    Web search integration. Three providers in order of preference:

    1. Brave Search API — $0/month free tier (2000 queries/month)
       Get key: api.search.brave.com
    2. SerpAPI — 100 free searches/month
       Get key: serpapi.com
    3. DuckDuckGo Instant Answer — completely free, no key needed
       Limited to instant answers, not full web results

    Config in prism_config.toml:
      [search]
      provider       = "auto"   # "brave"|"serp"|"ddg"|"auto"
      brave_api_key  = ""       # api.search.brave.com/app/keys
      serp_api_key   = ""       # serpapi.com/manage-api-key
    """

    def __init__(self, provider="auto",
                  brave_key="", serp_key=""):
        self._provider  = provider
        self._brave_key = brave_key
        self._serp_key  = serp_key

    @classmethod
    def from_config(cls, config: dict) -> "PrismSearch":
        s = config.get("search", {})
        return cls(
            provider  = s.get("provider", "auto"),
            brave_key = s.get("brave_api_key", ""),
            serp_key  = s.get("serp_api_key", ""),
        )

    @property
    def configured(self) -> bool:
        return True   # DuckDuckGo requires no key

    def search(self, query: str,
                n: int = 5) -> list[SearchResult]:
        """
        Search the web. Returns results ranked by relevance.
        Falls back through providers automatically.
        """
        provider = self._resolve_provider()
        if provider == "brave":
            return self._brave_search(query, n)
        if provider == "serp":
            return self._serp_search(query, n)
        return self._ddg_search(query, n)

    def quick_answer(self, query: str) -> str:
        """
        Get a quick factual answer using DuckDuckGo Instant Answer.
        Best for: current weather, definitions, conversions, facts.
        Returns plain text answer or empty string.
        """
        url = (f"https://api.duckduckgo.com/?q="
               f"{urllib.parse.quote(query)}&format=json&no_html=1")
        try:
            resp = urllib.request.urlopen(url, timeout=5)
            data = json.loads(resp.read())
            answer = (data.get("Answer")
                      or data.get("AbstractText")
                      or data.get("Definition", ""))
            return answer[:1000] if answer else ""
        except Exception as e:
            logger.debug("DDG instant answer failed: %s", e)
            return ""

    def _resolve_provider(self) -> str:
        if self._provider == "brave" and self._brave_key:
            return "brave"
        if self._provider == "serp" and self._serp_key:
            return "serp"
        if self._provider == "auto":
            if self._brave_key: return "brave"
            if self._serp_key:  return "serp"
        return "ddg"

    def _brave_search(self, query: str, n: int) -> list[SearchResult]:
        url = (f"https://api.search.brave.com/res/v1/web/search"
               f"?q={urllib.parse.quote(query)}&count={n}")
        req = urllib.request.Request(url, headers={
            "Accept":            "application/json",
            "Accept-Encoding":   "gzip",
            "X-Subscription-Token": self._brave_key,
        })
        try:
            resp = urllib.request.urlopen(req, timeout=8)
            data = json.loads(resp.read())
            return [SearchResult(
                title   = r.get("title",""),
                url     = r.get("url",""),
                snippet = r.get("description",""),
                source  = "brave",
            ) for r in data.get("web",{}).get("results",[])][:n]
        except Exception as e:
            logger.warning("Brave search failed: %s", e)
            return self._ddg_search(query, n)

    def _serp_search(self, query: str, n: int) -> list[SearchResult]:
        url = (f"https://serpapi.com/search.json"
               f"?q={urllib.parse.quote(query)}"
               f"&num={n}&api_key={self._serp_key}")
        try:
            resp = urllib.request.urlopen(url, timeout=8)
            data = json.loads(resp.read())
            return [SearchResult(
                title   = r.get("title",""),
                url     = r.get("link",""),
                snippet = r.get("snippet",""),
                source  = "serp",
            ) for r in data.get("organic_results",[])][:n]
        except Exception as e:
            logger.warning("SerpAPI failed: %s", e)
            return self._ddg_search(query, n)

    def _ddg_search(self, query: str, n: int) -> list[SearchResult]:
        """
        DuckDuckGo HTML scraping — free, no key, limited results.
        Uses the lite endpoint which is more scraping-friendly.
        """
        url = (f"https://html.duckduckgo.com/html/?q="
               f"{urllib.parse.quote(query)}")
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (compatible; PRISM/1.0)"})
        try:
            resp = urllib.request.urlopen(req, timeout=8)
            html = resp.read().decode(errors="replace")
            import re
            results = []
            # Extract result snippets from DDG HTML
            pattern = r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>([^<]+)</a>'
            for i, (url_raw, title) in enumerate(
                    re.findall(pattern, html)):
                if i >= n: break
                results.append(SearchResult(
                    title   = title.strip(),
                    url     = url_raw,
                    snippet = "",
                    source  = "ddg",
                ))
            if not results:
                # Fallback: return instant answer as single result
                answer = self.quick_answer(query)
                if answer:
                    results.append(SearchResult(
                        title   = query,
                        url     = "",
                        snippet = answer,
                        source  = "ddg_instant",
                    ))
            return results
        except Exception as e:
            logger.debug("DDG search failed: %s", e)
            return []

    def status_summary(self) -> dict:
        return {
            "provider":   self._resolve_provider(),
            "configured": self.configured,
            "free_tier":  self._resolve_provider() == "ddg",
        }
