from __future__ import annotations

import base64
import json
import logging
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class BrowserStep:
    action:      str    # "navigate"|"click"|"type"|"scroll"|"extract"|"wait"
    target:      str    # CSS selector, URL, or text to type
    description: str    # human-readable description of this step
    success:     bool   = True
    result:      str    = ""


@dataclass
class BrowserTaskResult:
    goal:        str
    success:     bool
    steps:       list[BrowserStep]
    extracted:   str    = ""    # any data extracted from the page
    final_url:   str    = ""
    error:       str    = ""
    screenshot:  str    = ""    # base64 PNG of final state (optional)


class PersistentBrowserContext:
    """
    Manages a persistent browser context with saved cookies/sessions.
    Saves login state so the user only logs into sites once.
    """
    SESSION_DIR = Path.home() / ".prism" / "browser_sessions"

    def __init__(self):
        self.SESSION_DIR.mkdir(parents=True, exist_ok=True)

    def session_path(self, domain: str) -> str:
        """Path to saved session for a domain."""
        safe = domain.replace("://", "_").replace("/", "_").replace(".", "_")
        return str(self.SESSION_DIR / f"{safe}_session")

    def has_session(self, domain: str) -> bool:
        return Path(self.session_path(domain)).exists()

    def launch_with_session(self, p, domain: str, headless: bool = True):
        """Launch browser with saved session if available."""
        session_path = self.session_path(domain)
        if self.has_session(domain):
            context = p.chromium.launch_persistent_context(
                session_path, headless=headless,
                args=["--no-sandbox", "--disable-dev-shm-usage"])
        else:
            browser = p.chromium.launch(headless=headless)
            context = browser.new_context()
        return context

    def save_session(self, context, domain: str) -> None:
        """Save current session state."""
        try:
            context.storage_state(path=self.session_path(domain))
        except Exception:
            pass


class PrismBrowserAgent:
    """
    LLM-guided browser automation via Playwright.

    The agent receives a goal in natural language, uses the LLM to
    decide what to do on the current page, executes the action,
    observes the result, and repeats until the goal is achieved
    or the max steps limit is reached.

    Requires:
      pip install playwright
      playwright install chromium

    Usage:
        agent = PrismBrowserAgent.setup(llm_router=router)
        result = agent.execute("find the cheapest flight from London to Paris next Friday")
    """

    MAX_STEPS              = 15
    PAGE_TIMEOUT           = 10000   # ms
    LOGIN_DETECTION_THRESHOLD = 3    # min matching indicators to flag login wall

    def __init__(
        self,
        llm_router       = None,
        headless:    bool = True,
        max_steps:   int  = MAX_STEPS,
        screenshot:  bool = False,
    ):
        self._router     = llm_router
        self._llm_router = llm_router
        self._headless   = headless
        self._max_steps  = max_steps
        self._screenshot = screenshot
        self._browser    = None
        self._page       = None

    @classmethod
    def setup(cls, **kwargs) -> "PrismBrowserAgent":
        return cls(**kwargs)

    @property
    def available(self) -> bool:
        try:
            import playwright  # noqa
            return True
        except ImportError:
            return False

    def execute(self, goal: str,
                start_url: str = "https://www.google.com") -> BrowserTaskResult:
        """
        Execute a browser task to achieve a goal.
        Returns BrowserTaskResult with all steps and extracted data.
        """
        if not self.available:
            return BrowserTaskResult(
                goal=goal, success=False, steps=[],
                error="Playwright not installed. Run: "
                      "pip install playwright && playwright install chromium")

        steps: list[BrowserStep] = []
        try:
            from playwright.sync_api import sync_playwright
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=self._headless)
                page    = browser.new_page()
                page.set_default_timeout(self.PAGE_TIMEOUT)

                # Navigate to start URL
                page.goto(start_url)
                steps.append(BrowserStep(
                    "navigate", start_url,
                    f"Opened {start_url}", True))

                for step_num in range(self._max_steps):
                    # Get current page state
                    page_text = self._extract_page_text(page)
                    current_url = page.url

                    # Ask LLM what to do next
                    action_json = self._decide_action(
                        goal, current_url, page_text, steps)

                    if action_json is None:
                        break

                    done   = action_json.get("done", False)
                    reason = action_json.get("reason", "")

                    if done:
                        extracted = action_json.get("result", page_text[:2000])
                        steps.append(BrowserStep(
                            "extract", "", f"Goal achieved: {reason}",
                            True, extracted))
                        browser.close()
                        return BrowserTaskResult(
                            goal=goal, success=True, steps=steps,
                            extracted=extracted, final_url=current_url)

                    # Execute the action
                    step = self._execute_action(page, action_json)
                    steps.append(step)

                    if not step.success:
                        # Try to recover or give up
                        logger.debug("Step failed: %s", step.result)
                        if step_num >= 3:
                            break

                    time.sleep(0.5)   # brief pause between actions

                # Max steps reached — extract whatever we have
                final_text = self._extract_page_text(page)
                browser.close()
                return BrowserTaskResult(
                    goal=goal, success=False, steps=steps,
                    extracted=final_text[:2000],
                    final_url=page.url,
                    error="Reached maximum steps without completing goal")

        except Exception as e:
            logger.warning("Browser agent error: %s", e)
            return BrowserTaskResult(
                goal=goal, success=False, steps=steps, error=str(e)[:300])

    def _decide_action(
        self,
        goal:        str,
        current_url: str,
        page_text:   str,
        steps:       list[BrowserStep],
    ) -> Optional[dict]:
        """Ask LLM what to do on the current page."""
        if self._router is None:
            return None

        history = "\n".join(
            f"Step {i+1}: {s.action} — {s.description}"
            for i, s in enumerate(steps[-5:]))

        prompt = (
            f"You are controlling a web browser to achieve this goal:\n"
            f"GOAL: {goal}\n\n"
            f"Current URL: {current_url}\n"
            f"Recent steps:\n{history}\n\n"
            f"Current page content (truncated):\n{page_text[:1500]}\n\n"
            f"What should the browser do next? Return ONLY valid JSON:\n"
            f'{{"action":"click|type|navigate|scroll|extract",'
            f'"target":"CSS selector, URL, or text to type",'
            f'"description":"what this does",'
            f'"done":false,'
            f'"result":"if done=true, the answer/result extracted"}}\n'
            f'If the goal is achieved, set done:true and result to the answer.\n'
            f'If you need to search, navigate to https://www.google.com?q=your+query\n'
            f'For click: use a descriptive CSS selector or visible text.\n'
            f'Keep target under 100 characters.'
        )

        raw, _ = self._router.call(
            prompt, min_capability=2, max_tokens=300, json_mode=True)
        try:
            clean = raw.strip().lstrip("```json").rstrip("```").strip()
            return json.loads(clean)
        except Exception:
            return None

    def _execute_action(self, page, action_json: dict) -> BrowserStep:
        action      = action_json.get("action", "")
        target      = action_json.get("target", "")
        description = action_json.get("description", "")

        try:
            if action == "navigate":
                url = target if target.startswith("http") else f"https://{target}"
                page.goto(url, wait_until="domcontentloaded",
                          timeout=self.PAGE_TIMEOUT)

            elif action == "click":
                # Try multiple strategies to find and click the element
                try:
                    page.click(target, timeout=3000)
                except Exception:
                    try:
                        page.get_by_text(target).first.click(timeout=3000)
                    except Exception:
                        page.locator(target).first.click(timeout=3000)
                page.wait_for_load_state("networkidle", timeout=5000)

            elif action == "type":
                # Find the focused or first visible input and type
                page.keyboard.type(target)
                page.keyboard.press("Enter")
                page.wait_for_load_state("networkidle", timeout=5000)
                target = target[:50]

            elif action == "scroll":
                page.evaluate("window.scrollBy(0, 500)")

            elif action == "extract":
                text = self._extract_page_text(page)
                return BrowserStep(action, target, description, True, text[:2000])

            else:
                return BrowserStep(action, target,
                                   f"Unknown action: {action}", False)

            # Check if we hit a login wall
            page_text = self._extract_page_text(page)
            login_indicators = ["sign in", "log in", "create account",
                                 "password", "email address", "username"]
            if sum(1 for w in login_indicators if w in page_text.lower()) >= self.LOGIN_DETECTION_THRESHOLD:
                # Login wall detected
                logger.info("Login wall detected at %s", page.url)
                from urllib.parse import urlparse
                domain = urlparse(page.url).netloc
                self._last_login_domain = domain
                return BrowserStep(
                    action="login_required",
                    target=domain,
                    description=f"Login required for {domain}",
                    success=False,
                    result=f"Login wall detected. If you have credentials for {domain}, "
                           f"add them to prism_config.toml [browser_credentials] "
                           f"or complete the login manually."
                )

            return BrowserStep(action, target, description, True)

        except Exception as e:
            return BrowserStep(action, target, description, False, str(e)[:200])

    @staticmethod
    def _extract_page_text(page) -> str:
        """Extract visible text from the current page."""
        try:
            return page.evaluate("""
                () => {
                    const walker = document.createTreeWalker(
                        document.body,
                        NodeFilter.SHOW_TEXT,
                        {acceptNode: n => {
                            const p = n.parentElement;
                            if (!p) return NodeFilter.FILTER_REJECT;
                            const s = window.getComputedStyle(p);
                            return (s.display !== 'none' && s.visibility !== 'hidden')
                                ? NodeFilter.FILTER_ACCEPT
                                : NodeFilter.FILTER_REJECT;
                        }}
                    );
                    const texts = [];
                    let node;
                    while ((node = walker.nextNode()) && texts.length < 300) {
                        const t = node.textContent.trim();
                        if (t.length > 2) texts.push(t);
                    }
                    return texts.join(' ');
                }
            """)[:3000]
        except Exception:
            return ""

    def _understand_page_vision(self, page,
                                ollama_host: str = "http://localhost:11434") -> str:
        """
        Take a screenshot and send to Ollama LLaVA for visual understanding.
        More reliable than text extraction for complex pages.
        Returns plain text description of what's on the page.
        """
        try:
            screenshot = page.screenshot(type="jpeg", quality=50)
            b64        = base64.b64encode(screenshot).decode()
            prompt     = (
                "Describe what is on this web page in 3-5 sentences. "
                "What is the main content? What actions are available? "
                "Is there a login form, error message, or CAPTCHA?"
            )
            payload = json.dumps({
                "model": "llava",
                "prompt": prompt,
                "images": [b64],
                "stream": False,
            }).encode()
            req  = urllib.request.Request(
                f"{ollama_host}/api/generate", data=payload,
                headers={"Content-Type": "application/json"})
            resp = urllib.request.urlopen(req, timeout=20)
            return json.loads(resp.read()).get("response", "")
        except Exception as e:
            logger.debug("Vision page understanding failed: %s", e)
            return self._extract_page_text(page)

    def execute_with_session(
        self,
        goal:       str,
        start_url:  str = "https://www.google.com",
        domain:     str = None,
    ) -> BrowserTaskResult:
        """
        Execute with persistent session management.
        Reuses saved cookies/sessions from previous logins.
        Saves new sessions after successful completion.
        """
        if not self.available:
            return BrowserTaskResult(goal=goal, success=False, steps=[],
                error="Playwright not installed.")

        ctx_manager = PersistentBrowserContext()
        inferred_domain = domain or (start_url.split("/")[2] if "/" in start_url else start_url)

        steps = []
        try:
            from playwright.sync_api import sync_playwright
            with sync_playwright() as p:
                if ctx_manager.has_session(inferred_domain):
                    browser = p.chromium.launch(headless=self._headless)
                    context = browser.new_context(storage_state=
                        ctx_manager.session_path(inferred_domain))
                    logger.info("Using saved session for %s", inferred_domain)
                else:
                    browser = p.chromium.launch(headless=self._headless)
                    context = browser.new_context()

                page = context.new_page()
                page.set_default_timeout(self.PAGE_TIMEOUT)
                page.goto(start_url)
                steps.append(BrowserStep("navigate", start_url,
                                          f"Opened {start_url}", True))

                for step_num in range(self._max_steps):
                    # Try vision understanding for complex pages
                    if step_num % 3 == 0 and self._llm_router:
                        try:
                            page_desc = self._understand_page_vision(page)
                        except Exception:
                            page_desc = self._extract_page_text(page)
                    else:
                        page_desc = self._extract_page_text(page)

                    action_json = self._decide_action(
                        goal, page.url, page_desc, steps)
                    if action_json is None:
                        break

                    if action_json.get("done"):
                        extracted = action_json.get("result", page_desc[:2000])
                        ctx_manager.save_session(context, inferred_domain)
                        browser.close()
                        return BrowserTaskResult(goal=goal, success=True,
                            steps=steps, extracted=extracted, final_url=page.url)

                    step = self._execute_action(page, action_json)
                    steps.append(step)
                    if step.action == "login_required":
                        browser.close()
                        return BrowserTaskResult(goal=goal, success=False,
                            steps=steps, error=step.result)

                    time.sleep(0.5)

                ctx_manager.save_session(context, inferred_domain)
                final = self._extract_page_text(page)
                browser.close()
                return BrowserTaskResult(goal=goal, success=False, steps=steps,
                    extracted=final[:2000], final_url=page.url,
                    error="Max steps reached")

        except Exception as e:
            return BrowserTaskResult(goal=goal, success=False,
                steps=steps, error=str(e)[:300])

    def status(self) -> dict:
        return {
            "available":  self.available,
            "headless":   self._headless,
            "max_steps":  self._max_steps,
            "requires":   "pip install playwright && playwright install chromium"
                          if not self.available else "",
        }
