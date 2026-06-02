"""Tests for prism_pwa — PWA asset generation."""
import json

import pytest

from prism_pwa import get_icon_svg, get_manifest, get_mobile_html, get_service_worker


# ── manifest ──────────────────────────────────────────────────────────────────

class TestManifest:
    def setup_method(self):
        self.manifest = json.loads(get_manifest())

    def test_name(self):
        assert "PRISM" in self.manifest["name"]

    def test_short_name(self):
        assert self.manifest["short_name"] == "PRISM"

    def test_start_url(self):
        assert self.manifest["start_url"] == "/mobile"

    def test_display_standalone(self):
        assert self.manifest["display"] == "standalone"

    def test_has_icons(self):
        assert isinstance(self.manifest.get("icons"), list)
        assert len(self.manifest["icons"]) > 0

    def test_theme_color(self):
        assert "theme_color" in self.manifest

    def test_background_color(self):
        assert "background_color" in self.manifest

    def test_valid_json(self):
        raw = get_manifest()
        parsed = json.loads(raw)
        assert isinstance(parsed, dict)


# ── service worker ─────────────────────────────────────────────────────────────

class TestServiceWorker:
    def setup_method(self):
        self.sw = get_service_worker()

    def test_returns_string(self):
        assert isinstance(self.sw, str)
        assert len(self.sw) > 100

    def test_has_install_event(self):
        assert "install" in self.sw

    def test_has_fetch_event(self):
        assert "fetch" in self.sw

    def test_caches_mobile_shell(self):
        assert "/mobile" in self.sw

    def test_caches_manifest(self):
        assert "/manifest.json" in self.sw

    def test_caches_icon(self):
        assert "/icon.svg" in self.sw

    def test_has_cache_name(self):
        assert "prism" in self.sw.lower()


# ── icon SVG ──────────────────────────────────────────────────────────────────

class TestIconSVG:
    def setup_method(self):
        self.svg = get_icon_svg()

    def test_returns_string(self):
        assert isinstance(self.svg, str)

    def test_is_svg(self):
        assert self.svg.strip().startswith("<svg")

    def test_has_viewbox(self):
        assert "viewBox" in self.svg

    def test_has_prism_content(self):
        assert len(self.svg) > 200


# ── mobile HTML ───────────────────────────────────────────────────────────────

class TestMobileHTML:
    def setup_method(self):
        self.html = get_mobile_html()

    def test_returns_string(self):
        assert isinstance(self.html, str)
        assert len(self.html) > 1000

    def test_has_doctype(self):
        assert "<!DOCTYPE html>" in self.html or "<!doctype html>" in self.html.lower()

    def test_references_manifest(self):
        assert "/manifest.json" in self.html

    def test_references_sw(self):
        assert "/sw.js" in self.html

    def test_references_icon(self):
        assert "/icon.svg" in self.html

    def test_has_chat_tab(self):
        assert "chat" in self.html.lower()

    def test_has_goals_tab(self):
        assert "goal" in self.html.lower()

    def test_has_voice_tab(self):
        assert "voice" in self.html.lower()

    def test_has_status_tab(self):
        assert "status" in self.html.lower()

    def test_has_pwa_meta_tags(self):
        assert "apple-mobile-web-app" in self.html

    def test_has_viewport_meta(self):
        assert "viewport" in self.html

    def test_post_chat_endpoint(self):
        assert "/chat" in self.html

    def test_horizon_goals_endpoint(self):
        assert "/horizon/goals" in self.html

    def test_safe_area_css(self):
        assert "safe-area-inset" in self.html

    def test_speech_recognition(self):
        assert "SpeechRecognition" in self.html or "speechRecognition" in self.html

    def test_install_prompt_handler(self):
        assert "beforeinstallprompt" in self.html
