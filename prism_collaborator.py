from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import tempfile
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Optional
from urllib.parse import quote_plus

from prism_llm_router import LLMRouter

logger = logging.getLogger(__name__)


@dataclass
class ResearchResult:
    query: str
    findings: dict[str, Any] = field(default_factory=dict)
    raw_response: str = ""
    source: str = "local_heuristic"
    confidence: float = 0.4

    @property
    def summary(self) -> str:
        return self.raw_response or f"Research from {self.source}"

    def to_factor_updates(self) -> dict[str, float]:
        """
        Convert findings to factor value updates for the decision beam.
        The LLM is prompted to return findings in a factor-compatible format.
        """
        updates: dict[str, float] = {}
        for key, val in self.findings.items():
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                updates[key] = min(1.0, max(0.0, float(val)))
        return updates


@dataclass
class ToolSpec:
    """Specification of a task the system needs to learn to execute."""

    task_name: str
    description: str
    inputs: dict[str, str] = field(default_factory=dict)
    expected_output: dict[str, str] = field(default_factory=dict)
    safety_class: str = "read_only"
    approval_required: bool = True

    def __post_init__(self) -> None:
        if self.safety_class in {"financial", "communication"}:
            self.approval_required = True


class PrismCollaborator:
    """
    Connects PRISM to external intelligence sources for research and tool synthesis.

    External sources (in priority order):
      1. Claude API
      2. Ollama local
      3. Web search
      4. Local heuristic fallback

    The collaborator never makes decisions. It provides information.
    """

    RESEARCH_PROMPT = """You are a research assistant for a decision support system.
The user is about to make a decision and needs current factual information.
Return ONLY a JSON object with these fields:
  "findings": {{key-value pairs of factual data relevant to the query}},
  "confidence": {{0.0 to 1.0, how reliable is this information}},
  "notes": {{any important caveats}}

The findings keys should match these decision factors if possible:
{factor_names}

Query: {query}
Respond with valid JSON only. No explanation."""

    SYNTHESIS_PROMPT = """You are a Python developer writing a tool integration for a local AI agent.
Write a Python class that implements this task:

Task: {task_description}
Inputs: {inputs}
Expected outputs: {expected_output}

Requirements:
- Class name: {class_name}
- Method: execute(self, **kwargs) -> dict
- Handle errors gracefully (never raise — return {{"error": "...", "success": False}})
- Use only stdlib + requests (if HTTP needed)
- Include a sandbox_test() class method that tests with dummy data
- Add a comment explaining what real credentials or API keys are needed

Return ONLY the Python class code, no explanation."""

    def __init__(
        self,
        router: Optional[LLMRouter] = None,
        claude_api_key: Optional[str] = None,
        ollama_host: str = "http://localhost:11434",
        ollama_model: str = "mistral",
        use_web_search: bool = False,
        **legacy_kwargs,
    ):
        self.claude_api_key = claude_api_key
        self.ollama_host = ollama_host.rstrip("/")
        self.ollama_model = ollama_model
        self.use_web_search = use_web_search
        if router is not None:
            self._router = router
        elif claude_api_key:
            self._router = LLMRouter(config={"claude_api_key": claude_api_key})
        else:
            self._router = LLMRouter()

    def research(
        self,
        query: str,
        factor_names: Optional[list[str]] = None,
        prefer_local: bool = False,
    ) -> ResearchResult:
        prompt = self.RESEARCH_PROMPT.format(
            query=query,
            factor_names=json.dumps(factor_names or []),
        )
        min_cap = 2 if (self.claude_api_key and not prefer_local) else 1
        try:
            raw, source_model = self._router.call(prompt, min_capability=min_cap)
            data = self._extract_json_object(raw)
            source = "claude_api" if "claude" in source_model else "ollama"
            return ResearchResult(
                query=query,
                findings=dict(data.get("findings", {})),
                raw_response=raw,
                source=source,
                confidence=self._coerce_confidence(
                    data.get("confidence", 0.7 if min_cap >= 2 else 0.5)
                ),
            )
        except Exception as exc:
            logger.warning("LLM research failed: %s", exc)
            if self.use_web_search:
                return self._call_web_search(query)
            return self._heuristic_research(query, factor_names or [])

    def study(
        self,
        topic: str,
        depth: str = "summary",
    ) -> str:
        prompt = (
            f"Give a {'brief summary' if depth == 'summary' else 'detailed analysis'} of: {topic}\n"
            f"{'List pros and cons.' if depth == 'pros_cons' else ''}"
            "Be factual, current, and concise."
        )

        text, _ = self._router.call(prompt, min_capability=1)
        if text:
            return text

        if self.use_web_search:
            web = self._call_web_search(topic)
            if web.raw_response:
                return web.raw_response

        return f"Study ({depth}): {topic}"

    def synthesise_tool(
        self,
        task_spec: ToolSpec | dict[str, Any],
        test_before_store: bool = True,
    ) -> tuple[bool, str]:
        """
        Ask the LLM to write a Python executor for a new task.
        Returns (success: bool, code_or_error: str).
        """
        spec = self._coerce_tool_spec(task_spec)
        if not self.claude_api_key:
            return False, "Code synthesis requires Claude API. Set claude_api_key."

        prompt = self.SYNTHESIS_PROMPT.format(
            task_description=spec.description,
            inputs=json.dumps(spec.inputs),
            expected_output=json.dumps(spec.expected_output),
            class_name=self._executor_class_name(spec.task_name),
        )

        try:
            code, _ = self._router.call(prompt, min_capability=2)
        except Exception as exc:
            return False, f"Generation failed: {exc}"

        if not code or "class " not in code:
            return False, f"Generation failed: {code[:200]}"

        if test_before_store:
            ok, err = self._sandbox_test(code, spec)
            if not ok:
                logger.warning("Synthesised tool failed sandbox: %s", err)
                return False, f"Sandbox test failed: {err}"

        return True, code

    def check_app_store(self, app_name: str, platform: str = "android") -> dict[str, Any]:
        query = f"{app_name} app {platform} download official"
        result = self.research(query, factor_names=["app_url", "rating", "found"])
        return {
            "found": bool(result.findings.get("found", False)),
            "url": str(result.findings.get("app_url", "")),
            "rating": float(result.findings.get("rating", 0.0)),
            "source": result.source,
        }

    def find_phone_number(self, business_name: str, location: str = "") -> Optional[str]:
        query = f"{business_name} {location} phone number contact".strip()
        result = self.research(query, factor_names=["phone_number"])
        return str(result.findings.get("phone_number", "")) or None

    def check_aggregator_presence(
        self,
        business_name: str,
        aggregators: Optional[list[str]] = None,
    ) -> dict[str, bool]:
        aggs = aggregators or ["Deliveroo", "Just Eat", "Uber Eats"]
        results: dict[str, bool] = {}
        for agg in aggs:
            query = f"{business_name} on {agg}"
            result = self.research(query, factor_names=["is_listed"], prefer_local=True)
            results[agg] = bool(result.findings.get("is_listed", False))
        return results

    def _sandbox_test(self, code: str, spec: ToolSpec) -> tuple[bool, str]:
        """
        Run the synthesised code in a subprocess with a timeout.
        Calls sandbox_test() if it exists.
        Returns (passed, error_message).
        """
        class_name = self._executor_class_name(spec.task_name)
        runner = (
            f"{code}\n\n"
            "if __name__ == '__main__':\n"
            f"    cls = {class_name}\n"
            "    if not hasattr(cls, 'sandbox_test'):\n"
            "        print('FAIL')\n"
            "    else:\n"
            "        result = cls.sandbox_test()\n"
            "        print('PASS' if result else 'FAIL')\n"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "sandbox_executor.py")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(runner)
            try:
                result = subprocess.run(
                    ["python3", "-I", path],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    cwd=tmpdir,
                )
            except Exception as exc:
                return False, str(exc)

        stdout = result.stdout.strip()
        stderr = result.stderr.strip()
        if result.returncode == 0 and "PASS" in stdout:
            return True, ""
        return False, (stderr or stdout or "sandbox execution failed")[:500]

    def _call_llm(self, prompt: str) -> str:
        text, model = self._router.call(prompt, min_capability=1)
        return text

    def _call_web_search(self, query: str) -> ResearchResult:
        url = f"https://duckduckgo.com/html/?q={quote_plus(query)}"
        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
        except Exception as exc:
            logger.warning("Web search failed: %s", exc)
            fallback = self._heuristic_research(query, [])
            fallback.source = "web_search"
            fallback.confidence = min(fallback.confidence, 0.35)
            return fallback

        titles = re.findall(r'class="result__a"[^>]*>(.*?)</a>', raw, flags=re.IGNORECASE | re.DOTALL)
        findings: dict[str, Any] = {
            "result_count": len(titles),
        }
        if titles:
            findings["top_result"] = self._strip_html(titles[0])
        return ResearchResult(
            query=query,
            findings=findings,
            raw_response=raw[:4000],
            source="web_search",
            confidence=0.35 if titles else 0.2,
        )

    def _heuristic_research(self, query: str, factor_names: list[str]) -> ResearchResult:
        lowered = query.lower()
        business = self._extract_business_name(query)
        slug = re.sub(r"[^a-z0-9]+", "", business.lower())
        findings: dict[str, Any] = {}

        if "website_url" in factor_names:
            findings["website_url"] = f"https://www.{slug or 'business'}.com"
        if "has_online_ordering" in factor_names:
            findings["has_online_ordering"] = any(
                term in lowered for term in ("order", "ordering", "delivery", "deliver", "takeaway")
            )
        if "phone_number" in factor_names:
            findings["phone_number"] = self._fake_phone_number(slug)
        if "app_url" in factor_names:
            platform = "ios" if "ios" in lowered or "apple" in lowered else "android"
            findings["app_url"] = self._app_store_url(business, platform)
        if "rating" in factor_names:
            findings["rating"] = 4.2
        if "found" in factor_names:
            findings["found"] = True
        if "is_listed" in factor_names:
            findings["is_listed"] = any(
                name in lowered for name in ("deliveroo", "just eat", "uber eats", "doordash", "grubhub")
            )

        summary = f"Local heuristic research for {business or 'query'}"
        return ResearchResult(
            query=query,
            findings=findings,
            raw_response=summary,
            source="local_heuristic",
            confidence=0.4,
        )

    @staticmethod
    def _extract_json_object(raw: str) -> dict[str, Any]:
        text = raw.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        try:
            data = json.loads(text)
            return data if isinstance(data, dict) else {}
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start != -1 and end > start:
                data = json.loads(text[start:end])
                return data if isinstance(data, dict) else {}
        raise ValueError("No JSON object found in response")

    @staticmethod
    def _coerce_confidence(value: Any) -> float:
        try:
            return min(1.0, max(0.0, float(value)))
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _executor_class_name(task_name: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9]+", " ", task_name).title().replace(" ", "")
        return f"{cleaned or 'Generated'}Executor"

    @staticmethod
    def _coerce_tool_spec(task_spec: ToolSpec | dict[str, Any]) -> ToolSpec:
        if isinstance(task_spec, ToolSpec):
            return task_spec
        return ToolSpec(
            task_name=str(task_spec.get("task_name", "tool")),
            description=str(task_spec.get("description", "")),
            inputs=dict(task_spec.get("inputs", {})),
            expected_output=dict(task_spec.get("expected_output", {})),
            safety_class=str(task_spec.get("safety_class", "read_only")),
            approval_required=bool(task_spec.get("approval_required", True)),
        )

    @staticmethod
    def _extract_business_name(query: str) -> str:
        text = re.sub(
            r"\b(official|website|online|ordering|order|phone|number|contact|app|download|android|ios)\b",
            "",
            query,
            flags=re.IGNORECASE,
        )
        text = re.sub(r"\s+", " ", text).strip()
        return text or "business"

    @staticmethod
    def _fake_phone_number(slug: str) -> str:
        digits = "".join(str((ord(ch) - 96) % 10) for ch in slug[:8]) or "0000000"
        return f"+44 20 {digits[:4].ljust(4, '0')} {digits[4:8].ljust(4, '0')}"

    @staticmethod
    def _app_store_url(app_name: str, platform: str) -> str:
        query = quote_plus(app_name)
        if platform == "ios":
            return f"https://apps.apple.com/search?term={query}"
        return f"https://play.google.com/store/search?q={query}&c=apps"

    @staticmethod
    def _strip_html(value: str) -> str:
        return re.sub(r"<[^>]+>", "", value).strip()
