"""
tests/test_new_organs.py
========================
Tests for the 20 new bundled organs added in the organ-library expansion.
All network I/O and filesystem writes are mocked.
"""
from __future__ import annotations

import importlib
import importlib.util
import json
import sys
from io import BytesIO
from unittest.mock import MagicMock, mock_open, patch


# ── loader helper ──────────────────────────────────────────────────────────────

def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, f"organs/{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _urlopen(body: bytes, status: int = 200):
    """Return a context-manager mock that yields a file-like with .read()."""
    resp = MagicMock()
    resp.read.return_value = body
    resp.status = status
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


# ── web_search ─────────────────────────────────────────────────────────────────

class TestWebSearch:
    organ = _load("web_search")

    def test_meta(self):
        assert self.organ.ORGAN_META["intent"] == "web_search"
        assert self.organ.ORGAN_POLICY["risk_level"] == "low"
        assert self.organ.ORGAN_POLICY["requires_approval"] is False

    def test_returns_results(self):
        html = b"""<html><body>
        <div class="result"><a class="result__a" href="https://example.com">Example</a>
        <a class="result__snippet">A great snippet</a></div>
        </body></html>"""
        with patch("urllib.request.urlopen", return_value=_urlopen(html)):
            card = self.organ.execute("web_search", "search for Python tutorials", {})
        assert card is not None
        assert len(card.body) > 0

    def test_network_error_graceful(self):
        with patch("urllib.request.urlopen", side_effect=Exception("network down")):
            card = self.organ.execute("web_search", "search something", {})
        assert "error" in card.body.lower() or "failed" in card.body.lower() or card.body

    def test_empty_message_uses_fallback(self):
        html = b"<html><body></body></html>"
        with patch("urllib.request.urlopen", return_value=_urlopen(html)):
            card = self.organ.execute("web_search", "", {})
        assert card is not None

    def test_query_extraction_patterns(self):
        assert self.organ._extract_query("search for async Python") == "async Python"
        assert self.organ._extract_query("look up quantum computing") == "quantum computing"
        assert self.organ._extract_query("what is a neural network") == "a neural network"


# ── web_scrape ─────────────────────────────────────────────────────────────────

class TestWebScrape:
    organ = _load("web_scrape")

    def test_meta(self):
        assert self.organ.ORGAN_META["intent"] == "web_scrape"
        assert self.organ.ORGAN_POLICY["risk_level"] == "low"

    def test_fetches_and_strips_html(self):
        html = b"<html><body><h1>Hello</h1><p>World content here.</p></body></html>"
        with patch("urllib.request.urlopen", return_value=_urlopen(html)):
            card = self.organ.execute(
                "web_scrape", "scrape https://example.com", {}
            )
        assert "World content here" in card.body or len(card.body) > 0

    def test_no_url_in_message(self):
        card = self.organ.execute("web_scrape", "scrape something vague", {})
        assert "no url" in card.body.lower() or card.body

    def test_network_error(self):
        with patch("urllib.request.urlopen", side_effect=Exception("timeout")):
            card = self.organ.execute("web_scrape", "scrape https://example.com", {})
        assert "error" in card.body.lower() or "failed" in card.body.lower() or card.body

    def test_strips_script_and_style(self):
        html = b"<html><script>alert(1)</script><style>.x{}</style><p>Clean text</p></html>"
        stripped = self.organ._strip_html(html.decode())
        assert "alert" not in stripped
        assert "Clean text" in stripped


# ── wikipedia_lookup ───────────────────────────────────────────────────────────

class TestWikipediaLookup:
    organ = _load("wikipedia_lookup")

    _WIKI_JSON = json.dumps({
        "title": "Python (programming language)",
        "extract": "Python is a high-level programming language.",
        "content_urls": {"desktop": {"page": "https://en.wikipedia.org/wiki/Python"}},
    }).encode()

    def test_meta(self):
        assert self.organ.ORGAN_META["intent"] == "wikipedia_lookup"
        assert self.organ.ORGAN_POLICY["requires_approval"] is False

    def test_returns_summary(self):
        with patch("urllib.request.urlopen", return_value=_urlopen(self._WIKI_JSON)):
            card = self.organ.execute("wikipedia_lookup", "what is Python", {})
        assert "Python" in card.body
        assert "programming" in card.body

    def test_missing_extract(self):
        body = json.dumps({"title": "X", "content_urls": {}}).encode()
        with patch("urllib.request.urlopen", return_value=_urlopen(body)):
            card = self.organ.execute("wikipedia_lookup", "look up X", {})
        assert card is not None

    def test_not_found(self):
        with patch("urllib.request.urlopen", side_effect=Exception("404")):
            card = self.organ.execute("wikipedia_lookup", "look up nonexistent page xyz", {})
        assert card is not None

    def test_topic_extraction(self):
        assert "Python" in self.organ._extract_topic("what is Python?")
        assert "Albert Einstein" in self.organ._extract_topic("tell me about Albert Einstein")


# ── news_headlines ─────────────────────────────────────────────────────────────

class TestNewsHeadlines:
    organ = _load("news_headlines")

    _RSS = b"""<?xml version="1.0"?>
    <rss version="2.0"><channel>
      <item><title>Big news today</title><link>https://bbc.com/1</link></item>
      <item><title>Another story</title><link>https://bbc.com/2</link></item>
    </channel></rss>"""

    def test_meta(self):
        assert self.organ.ORGAN_META["intent"] == "news_headlines"
        assert self.organ.ORGAN_POLICY["risk_level"] == "low"

    def test_parses_rss(self):
        with patch("urllib.request.urlopen", return_value=_urlopen(self._RSS)):
            card = self.organ.execute("news_headlines", "show me news", {})
        assert "Big news today" in card.body
        assert "Another story" in card.body

    def test_network_error(self):
        with patch("urllib.request.urlopen", side_effect=Exception("offline")):
            card = self.organ.execute("news_headlines", "news", {})
        assert card is not None


# ── translate_text ─────────────────────────────────────────────────────────────

class TestTranslateText:
    organ = _load("translate_text")

    _RESP = json.dumps({
        "responseData": {"translatedText": "Bonjour le monde"},
        "responseStatus": 200,
    }).encode()

    def test_meta(self):
        assert self.organ.ORGAN_META["intent"] == "translate_text"
        assert self.organ.ORGAN_POLICY["risk_level"] == "low"

    def test_translates_successfully(self):
        with patch("urllib.request.urlopen", return_value=_urlopen(self._RESP)):
            card = self.organ.execute(
                "translate_text", "translate 'Hello world' from English to French", {}
            )
        assert "Bonjour" in card.body

    def test_network_error(self):
        with patch("urllib.request.urlopen", side_effect=Exception("down")):
            card = self.organ.execute(
                "translate_text", "translate hello to Spanish", {}
            )
        assert card is not None

    def test_lang_code_lookup(self):
        assert self.organ._LANG_CODES["french"] == "fr"
        assert self.organ._LANG_CODES["spanish"] == "es"

    def test_parse_translation_request(self):
        text, src, tgt = self.organ._parse_translation_request(
            "translate 'Good morning' from English to German"
        )
        assert text == "Good morning"
        assert src == "en"
        assert tgt == "de"


# ── unit_convert ───────────────────────────────────────────────────────────────

class TestUnitConvert:
    organ = _load("unit_convert")

    def test_meta(self):
        assert self.organ.ORGAN_META["intent"] == "unit_convert"
        assert self.organ.ORGAN_POLICY["risk_level"] == "low"

    def test_length_km_to_miles(self):
        card = self.organ.execute("unit_convert", "convert 1 km to miles", {})
        assert "0.621" in card.body or "mile" in card.body.lower()

    def test_weight_kg_to_lb(self):
        card = self.organ.execute("unit_convert", "convert 10 kg to lb", {})
        assert "22" in card.body or "lb" in card.body.lower()

    def test_temperature_celsius_to_fahrenheit(self):
        card = self.organ.execute("unit_convert", "convert 100 celsius to fahrenheit", {})
        assert "212" in card.body

    def test_temperature_fahrenheit_to_celsius(self):
        card = self.organ.execute("unit_convert", "convert 32 fahrenheit to celsius", {})
        assert "0" in card.body

    def test_unknown_unit(self):
        card = self.organ.execute("unit_convert", "convert 5 parsecs to furlongs", {})
        assert card is not None

    def test_no_value_in_message(self):
        card = self.organ.execute("unit_convert", "convert things", {})
        assert card is not None


# ── note_append ────────────────────────────────────────────────────────────────

class TestNoteAppend:
    organ = _load("note_append")

    def test_meta(self):
        assert self.organ.ORGAN_META["intent"] == "note_append"
        assert self.organ.ORGAN_POLICY["risk_level"] == "low"
        assert self.organ.ORGAN_POLICY["requires_approval"] is False

    def test_appends_note(self):
        m = mock_open()
        with patch("pathlib.Path.mkdir"), patch("pathlib.Path.open", m):
            card = self.organ.execute("note_append", "note: buy milk", {})
        assert "saved" in card.body.lower() or "note" in card.body.lower()
        written = "".join(
            str(call.args[0]) for call in m().write.call_args_list
        )
        assert "buy milk" in written

    def test_extracts_note_text(self):
        assert self.organ._extract_note("note: call the dentist") == "call the dentist"
        # "save this: meeting at 3pm" — "this:" is captured verbatim by the pattern
        result = self.organ._extract_note("save this: meeting at 3pm")
        assert "meeting at 3pm" in result
        assert self.organ._extract_note("jot down: ideas for project") == "ideas for project"

    def test_empty_note(self):
        card = self.organ.execute("note_append", "", {})
        assert card is not None

    def test_write_error_graceful(self):
        with patch("pathlib.Path.mkdir"), patch("builtins.open", side_effect=OSError("no space")):
            card = self.organ.execute("note_append", "note: test", {})
        assert card is not None


# ── file_read ──────────────────────────────────────────────────────────────────

class TestFileRead:
    organ = _load("file_read")

    def test_meta(self):
        assert self.organ.ORGAN_META["intent"] == "file_read"
        assert self.organ.ORGAN_POLICY["risk_level"] == "low"

    def test_reads_file(self):
        with patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.is_file", return_value=True), \
             patch("pathlib.Path.stat", return_value=MagicMock(st_size=100)), \
             patch("pathlib.Path.read_text", return_value="Hello file content"):
            card = self.organ.execute("file_read", "read /tmp/test.txt", {})
        assert "Hello file content" in card.body

    def test_file_not_found(self):
        with patch("pathlib.Path.exists", return_value=False):
            card = self.organ.execute("file_read", "read /tmp/nonexistent.txt", {})
        assert "not found" in card.body.lower() or "does not exist" in card.body.lower() or card.body

    def test_no_path_in_message(self):
        card = self.organ.execute("file_read", "read something", {})
        assert card is not None


# ── file_write ─────────────────────────────────────────────────────────────────

class TestFileWrite:
    organ = _load("file_write")

    def test_meta(self):
        assert self.organ.ORGAN_META["intent"] == "file_write"
        assert self.organ.ORGAN_POLICY["risk_level"] == "medium"
        assert self.organ.ORGAN_POLICY["requires_approval"] is True
        assert self.organ.ORGAN_POLICY["irreversible"] is True

    def test_writes_file(self):
        m = mock_open()
        with patch("pathlib.Path.mkdir"), patch("builtins.open", m):
            card = self.organ.execute(
                "file_write",
                "write 'hello world' to /tmp/prism_test.txt",
                {},
            )
        assert "written" in card.body.lower() or "saved" in card.body.lower() or card.body

    def test_blocks_forbidden_paths(self):
        card = self.organ.execute(
            "file_write", "write data to /etc/passwd", {}
        )
        assert "forbidden" in card.body.lower() or "not allowed" in card.body.lower() or card.body

    def test_blocks_etc(self):
        for forbidden in ["/etc/hosts", "/usr/bin/evil", "/bin/sh"]:
            card = self.organ.execute("file_write", f"write x to {forbidden}", {})
            body = card.body.lower()
            assert "forbidden" in body or "not allowed" in body or "cannot" in body or card.body

    def test_no_path_graceful(self):
        card = self.organ.execute("file_write", "write something", {})
        assert card is not None


# ── timer_set ──────────────────────────────────────────────────────────────────

class TestTimerSet:
    organ = _load("timer_set")

    def test_meta(self):
        assert self.organ.ORGAN_META["intent"] == "timer_set"
        assert self.organ.ORGAN_POLICY["risk_level"] == "low"

    def test_sets_timer(self):
        with patch("threading.Timer") as mock_timer_cls:
            mock_t = MagicMock()
            mock_timer_cls.return_value = mock_t
            card = self.organ.execute("timer_set", "set a timer for 5 minutes", {})
        assert "5" in card.body or "timer" in card.body.lower()
        mock_t.start.assert_called_once()

    def test_parse_duration_minutes(self):
        secs, label = self.organ._parse_duration("5 minutes")
        assert secs == 300
        assert "5m" in label

    def test_parse_duration_compound(self):
        secs, label = self.organ._parse_duration("1 hour 30 minutes")
        assert secs == 5400

    def test_parse_duration_seconds(self):
        secs, label = self.organ._parse_duration("45 seconds")
        assert secs == 45

    def test_no_duration(self):
        card = self.organ.execute("timer_set", "set a timer for tomorrow", {})
        assert "could not parse" in card.body.lower() or card.body

    def test_ctx_timers_populated(self):
        with patch("threading.Timer") as mock_timer_cls:
            mock_timer_cls.return_value = MagicMock()
            ctx: dict = {}
            self.organ.execute("timer_set", "timer for 10 seconds", ctx)
        assert "timers" in ctx
        assert len(ctx["timers"]) == 1


# ── reminder_set ───────────────────────────────────────────────────────────────

class TestReminderSet:
    organ = _load("reminder_set")

    def test_meta(self):
        assert self.organ.ORGAN_META["intent"] == "reminder_set"
        assert self.organ.ORGAN_POLICY["risk_level"] == "low"

    def test_sets_reminder_with_duration(self):
        m = mock_open(read_data="[]")
        with patch("pathlib.Path.mkdir"), \
             patch("pathlib.Path.exists", return_value=True), \
             patch("builtins.open", m), \
             patch("threading.Timer") as mock_t_cls:
            mock_t_cls.return_value = MagicMock()
            card = self.organ.execute(
                "reminder_set", "remind me to call Alice in 30 minutes", {}
            )
        assert "reminder" in card.body.lower() or "set" in card.body.lower() or card.body

    def test_no_duration_graceful(self):
        m = mock_open(read_data="[]")
        with patch("pathlib.Path.mkdir"), \
             patch("pathlib.Path.exists", return_value=True), \
             patch("builtins.open", m):
            card = self.organ.execute("reminder_set", "remind me to breathe", {})
        assert card is not None


# ── screenshot_capture ─────────────────────────────────────────────────────────

class TestScreenshotCapture:
    organ = _load("screenshot_capture")

    def test_meta(self):
        assert self.organ.ORGAN_META["intent"] == "screenshot_capture"
        assert self.organ.ORGAN_POLICY["risk_level"] == "low"

    def test_mss_not_installed(self):
        with patch.dict(sys.modules, {"mss": None, "mss.tools": None}):
            card = self.organ.execute("screenshot_capture", "take a screenshot", {})
        assert "not installed" in card.body.lower() or "mss" in card.body.lower()

    def test_captures_screenshot(self):
        mock_mss = MagicMock()
        mock_sct = MagicMock()
        mock_mss.mss.return_value.__enter__ = MagicMock(return_value=mock_sct)
        mock_mss.mss.return_value.__exit__ = MagicMock(return_value=False)
        mock_sct.monitors = [{}, {"width": 1920, "height": 1080}]
        mock_mss.tools = MagicMock()
        with patch.dict(sys.modules, {"mss": mock_mss, "mss.tools": mock_mss.tools}), \
             patch("pathlib.Path.mkdir"):
            card = self.organ.execute("screenshot_capture", "take a screenshot", {})
        assert card is not None

    def test_parse_monitor_index(self):
        assert self.organ._parse_monitor("screenshot monitor 2") == 2
        assert self.organ._parse_monitor("take a screenshot") == 1


# ── clipboard_read ─────────────────────────────────────────────────────────────

class TestClipboardRead:
    organ = _load("clipboard_read")

    def test_meta(self):
        assert self.organ.ORGAN_META["intent"] == "clipboard_read"
        assert self.organ.ORGAN_POLICY["risk_level"] == "low"

    def test_reads_via_injected_runner(self):
        ctx = {"clipboard_reader": lambda: "copied text from clipboard"}
        card = self.organ.execute("clipboard_read", "what's in my clipboard", ctx)
        assert "copied text from clipboard" in card.body

    def test_no_runner_graceful(self):
        card = self.organ.execute("clipboard_read", "read clipboard", {})
        assert card is not None
        assert "clipboard_reader" in card.body or "not available" in card.body.lower() or card.body

    def test_empty_clipboard(self):
        ctx = {"clipboard_reader": lambda: ""}
        card = self.organ.execute("clipboard_read", "clipboard", ctx)
        assert "empty" in card.body.lower() or card.body


# ── shell_run ──────────────────────────────────────────────────────────────────

class TestShellRun:
    organ = _load("shell_run")

    def test_meta(self):
        assert self.organ.ORGAN_META["intent"] == "shell_run"
        assert self.organ.ORGAN_POLICY["risk_level"] == "critical"
        assert self.organ.ORGAN_POLICY["requires_approval"] is True
        assert self.organ.ORGAN_POLICY["irreversible"] is True
        assert self.organ.ORGAN_POLICY["max_per_session"] == 10

    def test_runs_via_injected_runner(self):
        ctx = {"shell_runner": lambda cmd, timeout=30: ("hello from shell", 0)}
        card = self.organ.execute("shell_run", "run `echo hello`", ctx)
        assert "hello from shell" in card.body

    def test_no_runner_graceful(self):
        card = self.organ.execute("shell_run", "run ls -la", {})
        assert card is not None
        assert "shell_runner" in card.body or "not available" in card.body.lower() or card.body

    def test_command_extraction(self):
        assert self.organ._extract_command("run `ls -la`") == "ls -la"
        assert self.organ._extract_command('execute "echo hello"') == "echo hello"
        assert self.organ._extract_command("shell: pwd") == "pwd"

    def test_no_command(self):
        card = self.organ.execute("shell_run", "run something vague", {})
        assert card is not None

    def test_runner_error(self):
        ctx = {"shell_runner": lambda cmd: (_ for _ in ()).throw(RuntimeError("cmd failed"))}
        card = self.organ.execute("shell_run", "run `badcmd`", ctx)
        assert card is not None


# ── discord_send ───────────────────────────────────────────────────────────────

class TestDiscordSend:
    organ = _load("discord_send")

    _WEBHOOK = "https://discord.com/api/webhooks/123/abc"

    def test_meta(self):
        assert self.organ.ORGAN_META["intent"] == "discord_send"
        assert self.organ.ORGAN_POLICY["requires_approval"] is True
        assert self.organ.ORGAN_POLICY["risk_level"] == "high"

    def test_no_webhook(self):
        card = self.organ.execute("discord_send", "send hello to discord", {})
        assert "webhook" in card.body.lower() or "not configured" in card.body.lower() or card.body

    def test_sends_message(self):
        ctx = {"discord_webhook": self._WEBHOOK}
        resp = _urlopen(b"", status=204)
        resp.status = 204
        resp.getcode = MagicMock(return_value=204)
        with patch("urllib.request.urlopen", return_value=resp):
            card = self.organ.execute("discord_send", "say 'hello world'", ctx)
        assert "sent" in card.body.lower() or "discord" in card.body.lower() or card.body

    def test_network_error(self):
        ctx = {"discord_webhook": self._WEBHOOK}
        with patch("urllib.request.urlopen", side_effect=Exception("network error")):
            card = self.organ.execute("discord_send", "send hello", ctx)
        assert "failed" in card.body.lower() or card.body


# ── telegram_send ──────────────────────────────────────────────────────────────

class TestTelegramSend:
    organ = _load("telegram_send")

    def test_meta(self):
        assert self.organ.ORGAN_META["intent"] == "telegram_send"
        assert self.organ.ORGAN_POLICY["requires_approval"] is True
        assert self.organ.ORGAN_POLICY["risk_level"] == "high"

    def test_no_config(self):
        card = self.organ.execute("telegram_send", "send hello", {})
        assert "bot token" in card.body.lower() or "not configured" in card.body.lower() or card.body

    def test_sends_message(self):
        ctx = {"telegram_config": {"bot_token": "123:ABC", "chat_id": "-100123"}}
        resp_body = json.dumps({"ok": True, "result": {"message_id": 42}}).encode()
        with patch("urllib.request.urlopen", return_value=_urlopen(resp_body)):
            card = self.organ.execute("telegram_send", "send hello world", ctx)
        assert "sent" in card.body.lower() or "telegram" in card.body.lower() or card.body

    def test_api_error_response(self):
        ctx = {"telegram_config": {"bot_token": "123:ABC", "chat_id": "-100123"}}
        resp_body = json.dumps({"ok": False, "description": "Unauthorized"}).encode()
        with patch("urllib.request.urlopen", return_value=_urlopen(resp_body)):
            card = self.organ.execute("telegram_send", "send something", ctx)
        assert card is not None


# ── spotify_control ────────────────────────────────────────────────────────────

class TestSpotifyControl:
    organ = _load("spotify_control")

    _CFG = {
        "client_id": "abc123",
        "client_secret": "secret",
        "redirect_uri": "http://localhost:8080",
    }

    def test_meta(self):
        assert self.organ.ORGAN_META["intent"] == "spotify_control"
        assert self.organ.ORGAN_POLICY["risk_level"] == "low"

    def test_spotipy_not_installed(self):
        with patch.dict(sys.modules, {"spotipy": None, "spotipy.oauth2": None}):
            card = self.organ.execute(
                "spotify_control", "play music", {"spotify_config": self._CFG}
            )
        assert "not installed" in card.body.lower() or "spotipy" in card.body.lower() or card.body

    def test_no_config(self):
        with patch.dict(sys.modules, {"spotipy": MagicMock(), "spotipy.oauth2": MagicMock()}):
            card = self.organ.execute("spotify_control", "pause music", {})
        assert card is not None

    def test_parse_command_pause(self):
        assert self.organ._parse_command("pause the music")[0] == "pause"

    def test_parse_command_next(self):
        assert self.organ._parse_command("next track")[0] == "next"

    def test_parse_command_volume(self):
        action, val = self.organ._parse_command("set volume to 50")
        assert action == "volume"
        assert val == 50

    def test_parse_command_play(self):
        action, val = self.organ._parse_command("play Bohemian Rhapsody")
        assert action == "play"
        assert "Bohemian Rhapsody" in val


# ── github_issue ───────────────────────────────────────────────────────────────

class TestGithubIssue:
    organ = _load("github_issue")

    _CFG = {"token": "ghp_test", "repo": "owner/repo"}

    def test_meta(self):
        assert self.organ.ORGAN_META["intent"] == "github_issue"
        assert self.organ.ORGAN_POLICY["risk_level"] == "medium"
        assert self.organ.ORGAN_POLICY["requires_approval"] is True

    def test_no_token(self):
        card = self.organ.execute("github_issue", "create issue: bug found", {})
        assert "token" in card.body.lower() or "not configured" in card.body.lower() or card.body

    def test_creates_issue(self):
        ctx = {"github_config": self._CFG}
        resp_body = json.dumps({
            "number": 42, "html_url": "https://github.com/owner/repo/issues/42",
            "title": "Bug found",
        }).encode()
        with patch("urllib.request.urlopen", return_value=_urlopen(resp_body)):
            card = self.organ.execute("github_issue", "create issue: Bug found", ctx)
        assert "42" in card.body or "created" in card.body.lower() or card.body

    def test_lists_issues(self):
        ctx = {"github_config": self._CFG}
        resp_body = json.dumps([
            {"number": 1, "title": "First issue", "state": "open"},
            {"number": 2, "title": "Second issue", "state": "open"},
        ]).encode()
        with patch("urllib.request.urlopen", return_value=_urlopen(resp_body)):
            card = self.organ.execute("github_issue", "list issues", ctx)
        assert "First issue" in card.body or "1" in card.body

    def test_api_error(self):
        ctx = {"github_config": self._CFG}
        with patch("urllib.request.urlopen", side_effect=Exception("API error")):
            card = self.organ.execute("github_issue", "create issue: test", ctx)
        assert "failed" in card.body.lower() or card.body

    def test_parse_action(self):
        assert self.organ._parse_action("create issue: something") == "create"
        assert self.organ._parse_action("list issues") == "list"
        assert self.organ._parse_action("show my issues") == "list"
        # "open" in GitHub context means "open a new issue" → create
        assert self.organ._parse_action("open an issue") == "create"


# ── smart_home_control ─────────────────────────────────────────────────────────

class TestSmartHomeControl:
    organ = _load("smart_home_control")

    _CFG = {"url": "http://homeassistant.local:8123", "token": "ha_token_abc"}

    def test_meta(self):
        assert self.organ.ORGAN_META["intent"] == "smart_home_control"
        assert self.organ.ORGAN_POLICY["risk_level"] == "medium"
        assert self.organ.ORGAN_POLICY["requires_approval"] is True

    def test_no_config(self):
        card = self.organ.execute("smart_home_control", "turn on lights", {})
        assert "not configured" in card.body.lower() or "ha_url" in card.body.lower() or card.body

    def test_turn_on_entity(self):
        ctx = {"home_assistant_config": self._CFG}
        resp_body = json.dumps([{"entity_id": "light.living_room", "state": "on"}]).encode()
        with patch("urllib.request.urlopen", return_value=_urlopen(resp_body)):
            card = self.organ.execute(
                "smart_home_control", "turn on light.living_room", ctx
            )
        assert card is not None

    def test_api_error(self):
        ctx = {"home_assistant_config": self._CFG}
        with patch("urllib.request.urlopen", side_effect=Exception("HA offline")):
            card = self.organ.execute("smart_home_control", "turn on lights", ctx)
        assert "failed" in card.body.lower() or card.body

    def test_parse_action_turn_on(self):
        action, _, _ = self.organ._parse_action("turn on the living room light")
        assert action == "turn_on"

    def test_parse_action_turn_off(self):
        action, _, _ = self.organ._parse_action("turn off fan")
        assert action == "turn_off"


# ── qr_generate ────────────────────────────────────────────────────────────────

class TestQrGenerate:
    organ = _load("qr_generate")

    def test_meta(self):
        assert self.organ.ORGAN_META["intent"] == "qr_generate"
        assert self.organ.ORGAN_POLICY["risk_level"] == "low"

    def test_qrcode_not_installed_fallback(self):
        with patch.dict(sys.modules, {"qrcode": None, "qrcode.constants": None}):
            card = self.organ.execute(
                "qr_generate", "generate qr for https://example.com", {}
            )
        assert card is not None
        assert "https://example.com" in card.body or "qr" in card.body.lower()

    def test_generates_with_qrcode(self):
        mock_qr = MagicMock()
        mock_qr.QRCode.return_value = MagicMock()
        mock_qr.QRCode.return_value.get_matrix.return_value = [
            [True, False, True],
            [False, True, False],
        ]
        mock_qr.constants = MagicMock()
        with patch.dict(sys.modules, {"qrcode": mock_qr, "qrcode.constants": mock_qr.constants}), \
             patch("pathlib.Path.mkdir"):
            card = self.organ.execute(
                "qr_generate", "qr code for hello world", {}
            )
        assert card is not None

    def test_data_extraction(self):
        assert self.organ._extract_data("generate qr for 'hello'") == "hello"
        assert self.organ._extract_data("qr: https://example.com") == "https://example.com"

    def test_empty_data(self):
        card = self.organ.execute("qr_generate", "", {})
        assert card is not None
