"""
tests/test_vision_analyzer.py
==============================
Tests for vision_analyzer.py

Covers:
  - is_available: True when Ollama lists the model, False when down
  - analyze_frame: parses JSON response into FrameAnalysis
  - degrades_when_ollama_down: returns empty default, no exception raised
  - analyze_technique: TechniqueReport with correct n_frames and aggregations
  - detect_tactical_situation: parses tactical JSON
  - summarize_session: aggregates frame analyses into SessionSummary
  - compare_sessions: returns comparison dict
  - timeout_handled: socket timeout doesn't propagate as exception
"""

from __future__ import annotations

import json
import urllib.error
from unittest.mock import MagicMock, patch

import pytest

from media_processor import Frame
from vision_analyzer import (
    FrameAnalysis,
    SessionSummary,
    TacticalContext,
    TechniqueReport,
    VisionAnalyzer,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_analyzer(model: str = "llava") -> VisionAnalyzer:
    return VisionAnalyzer(host="http://localhost:11434", model=model)


def _mock_response(data: dict | str) -> MagicMock:
    """Build a mock urllib HTTP response that returns the given data."""
    if isinstance(data, dict):
        body = json.dumps(data).encode()
    else:
        body = data.encode() if isinstance(data, str) else data
    mock = MagicMock()
    mock.__enter__ = lambda s: s
    mock.__exit__  = MagicMock(return_value=False)
    mock.read      = MagicMock(return_value=body)
    mock.status    = 200
    return mock


def _ollama_response(response_text: str) -> dict:
    """Wrap text in Ollama's response envelope."""
    return {"model": "llava", "response": response_text, "done": True}


# ---------------------------------------------------------------------------
# is_available
# ---------------------------------------------------------------------------

class TestIsAvailable:
    def test_available_when_model_in_list(self):
        va  = _make_analyzer("llava")
        tag_resp = {"models": [{"name": "llava:latest"}, {"name": "llava:13b"}]}
        with patch("urllib.request.urlopen", return_value=_mock_response(tag_resp)):
            assert va.is_available() is True

    def test_not_available_when_model_missing(self):
        va  = _make_analyzer("llava:34b")
        tag_resp = {"models": [{"name": "mistral:latest"}]}
        with patch("urllib.request.urlopen", return_value=_mock_response(tag_resp)):
            assert va.is_available() is False

    def test_not_available_when_ollama_down(self):
        va = _make_analyzer()
        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError("connection refused"),
        ):
            assert va.is_available() is False


# ---------------------------------------------------------------------------
# analyze_frame — normal path
# ---------------------------------------------------------------------------

class TestAnalyzeFrame:
    _FRAME_JSON = json.dumps({
        "quality":     8,
        "action":      "player sprinting towards goal",
        "positioning": {"notes": "wide left flank"},
        "improvement": "open hips for better turn",
        "tags":        ["sprint", "attack", "wide_play"],
    })

    def test_analyze_frame_returns_frame_analysis(self):
        va  = _make_analyzer()
        _env = _ollama_response(self._FRAME_JSON)
        with patch.object(va, "_call_ollama", return_value=json.loads(self._FRAME_JSON)):
            result = va.analyze_frame("base64img", "football", "athlete")
        assert isinstance(result, FrameAnalysis)

    def test_analyze_frame_parses_quality_score(self):
        va = _make_analyzer()
        with patch.object(va, "_call_ollama", return_value=json.loads(self._FRAME_JSON)):
            result = va.analyze_frame("base64img", "football", "athlete")
        assert result.quality_score == pytest.approx(0.8)

    def test_analyze_frame_parses_tags(self):
        va = _make_analyzer()
        with patch.object(va, "_call_ollama", return_value=json.loads(self._FRAME_JSON)):
            result = va.analyze_frame("base64img", "football", "athlete")
        assert "sprint" in result.tags
        assert "attack" in result.tags

    def test_analyze_frame_parses_positions(self):
        va = _make_analyzer()
        with patch.object(va, "_call_ollama", return_value=json.loads(self._FRAME_JSON)):
            result = va.analyze_frame("base64img", "football", "athlete")
        assert isinstance(result.positions, dict)

    def test_analyze_frame_parses_technique_notes(self):
        va = _make_analyzer()
        with patch.object(va, "_call_ollama", return_value=json.loads(self._FRAME_JSON)):
            result = va.analyze_frame("base64img", "football", "athlete")
        assert "hips" in result.technique_notes

    def test_analyze_frame_uses_custom_prompt(self):
        va       = _make_analyzer()
        captured: list[str] = []
        def fake_call(prompt, images=None, expect_json=True):
            captured.append(prompt)
            return {"quality": 5, "tags": []}
        with patch.object(va, "_call_ollama", side_effect=fake_call):
            va.analyze_frame("b64", "basketball", "coach", prompt="custom prompt")
        assert "custom prompt" in captured[0]

    def test_analyze_frame_uses_default_prompt_with_sport(self):
        va       = _make_analyzer()
        captured: list[str] = []
        def fake_call(prompt, images=None, expect_json=True):
            captured.append(prompt)
            return {"quality": 5, "tags": []}
        with patch.object(va, "_call_ollama", side_effect=fake_call):
            va.analyze_frame("b64", "tennis", "athlete")
        assert "tennis" in captured[0]


# ---------------------------------------------------------------------------
# Graceful degradation when Ollama is down
# ---------------------------------------------------------------------------

class TestDegradationWhenOllamaDown:
    def test_analyze_frame_returns_default_not_raises(self):
        va = _make_analyzer()
        with patch.object(
            va, "_call_ollama",
            side_effect=ConnectionError("Ollama down"),
        ):
            result = va.analyze_frame("b64", "football", "athlete")
        assert isinstance(result, FrameAnalysis)
        assert result.quality_score == 0.0
        assert result.tags == []

    def test_detect_tactical_situation_returns_default_not_raises(self):
        va = _make_analyzer()
        with patch.object(
            va, "_call_ollama",
            side_effect=ConnectionError("Ollama down"),
        ):
            ctx = va.detect_tactical_situation("b64", "football")
        assert isinstance(ctx, TacticalContext)
        assert ctx.phase == ""

    def test_summarize_session_returns_default_not_raises(self):
        va      = _make_analyzer()
        analyses = [
            FrameAnalysis(
                frame_id="f1", model="llava", prompt_used="p",
                raw_response="", tags=["run"], quality_score=0.7,
            )
        ]
        with patch.object(
            va, "_call_ollama",
            side_effect=ConnectionError("Ollama down"),
        ):
            summary = va.summarize_session(analyses, "football", "athlete")
        assert isinstance(summary, SessionSummary)

    def test_compare_sessions_returns_empty_dict_not_raises(self):
        va = _make_analyzer()
        s  = SessionSummary(
            session_id="s1", n_clips=5,
            highlights=[], load_estimate=0.5,
            technique_score=0.6, tactical_insights=[],
            recommendations=[],
        )
        with patch.object(
            va, "_call_ollama",
            side_effect=ConnectionError("down"),
        ):
            result = va.compare_sessions(s, s)
        assert isinstance(result, dict)
        assert "improvements" in result


# ---------------------------------------------------------------------------
# analyze_technique
# ---------------------------------------------------------------------------

class TestAnalyzeTechnique:
    def _make_frames(self, tmp_path, n: int) -> list[Frame]:
        frames = []
        for i in range(n):
            jpg = tmp_path / f"frame_{i:03d}.jpg"
            jpg.write_bytes(b"\xFF\xD8\xFF" + b"\xAA" * 50)
            frames.append(
                Frame(
                    frame_id=f"f{i}", video_id="vid001",
                    timestamp=float(i), path=str(jpg),
                )
            )
        return frames

    def test_technique_report_correct_n_frames_small(self, tmp_path):
        va      = _make_analyzer()
        frames  = self._make_frames(tmp_path, 5)
        def fake_analyze(b64, sport, role, prompt=None):
            return FrameAnalysis(
                frame_id="fx", model="llava", prompt_used="p",
                raw_response="", tags=["pass"], quality_score=0.7,
            )
        with patch.object(va, "analyze_frame", side_effect=fake_analyze):
            report = va.analyze_technique(frames, "football", "athlete")
        assert report.n_frames == 5
        assert isinstance(report, TechniqueReport)

    def test_technique_report_samples_at_most_20_frames(self, tmp_path):
        va     = _make_analyzer()
        frames = self._make_frames(tmp_path, 50)
        called = []
        def fake_analyze(b64, sport, role, prompt=None):
            called.append(1)
            return FrameAnalysis(
                frame_id="fx", model="llava", prompt_used="",
                raw_response="", tags=[], quality_score=0.5,
            )
        with patch.object(va, "analyze_frame", side_effect=fake_analyze):
            report = va.analyze_technique(frames, "football", "athlete")
        assert len(called) <= 20
        assert report.n_frames <= 20

    def test_technique_report_overall_score(self, tmp_path):
        va     = _make_analyzer()
        frames = self._make_frames(tmp_path, 3)
        scores = [0.8, 0.6, 0.7]
        call_idx = [0]
        def fake_analyze(b64, sport, role, prompt=None):
            s = scores[call_idx[0] % len(scores)]
            call_idx[0] += 1
            return FrameAnalysis(
                frame_id="fx", model="llava", prompt_used="",
                raw_response="", tags=[], quality_score=s,
            )
        with patch.object(va, "analyze_frame", side_effect=fake_analyze):
            report = va.analyze_technique(frames, "football", "athlete")
        assert report.overall_score == pytest.approx(0.7, abs=0.05)

    def test_technique_report_empty_frames(self):
        va     = _make_analyzer()
        report = va.analyze_technique([], "football", "athlete")
        assert report.n_frames == 0
        assert report.frame_analyses == []

    def test_technique_report_key_findings_from_tags(self, tmp_path):
        va     = _make_analyzer()
        frames = self._make_frames(tmp_path, 3)
        def fake_analyze(b64, sport, role, prompt=None):
            return FrameAnalysis(
                frame_id="fx", model="llava", prompt_used="",
                raw_response="", tags=["sprint", "attack"], quality_score=0.8,
            )
        with patch.object(va, "analyze_frame", side_effect=fake_analyze):
            report = va.analyze_technique(frames, "football", "athlete")
        assert "sprint" in report.key_findings


# ---------------------------------------------------------------------------
# detect_tactical_situation
# ---------------------------------------------------------------------------

class TestDetectTacticalSituation:
    _TACTICAL_JSON = {
        "phase":           "attack",
        "zone":            "final_third",
        "pressure_0_to_1": 0.8,
        "space_0_to_1":    0.3,
        "formation":       "4-3-3 high press",
        "tags":            ["high_press", "final_third"],
    }

    def test_parses_tactical_context(self):
        va = _make_analyzer()
        with patch.object(va, "_call_ollama", return_value=self._TACTICAL_JSON):
            ctx = va.detect_tactical_situation("b64", "football")
        assert ctx.phase           == "attack"
        assert ctx.ball_zone       == "final_third"
        assert ctx.pressure_level  == pytest.approx(0.8)
        assert ctx.space_available == pytest.approx(0.3)
        assert "high_press"        in ctx.tags

    def test_formation_extracted(self):
        va = _make_analyzer()
        with patch.object(va, "_call_ollama", return_value=self._TACTICAL_JSON):
            ctx = va.detect_tactical_situation("b64", "football")
        assert "4-3-3" in ctx.formation_guess

    def test_pressure_clamped_to_0_1(self):
        va = _make_analyzer()
        data = dict(self._TACTICAL_JSON)
        data["pressure_0_to_1"] = 5.0  # invalid — should be clamped
        with patch.object(va, "_call_ollama", return_value=data):
            ctx = va.detect_tactical_situation("b64", "football")
        assert ctx.pressure_level <= 1.0


# ---------------------------------------------------------------------------
# summarize_session
# ---------------------------------------------------------------------------

class TestSummarizeSession:
    _SUMMARY_JSON = {
        "highlights":        ["great sprint at 0:45", "key tackle"],
        "load_estimate":     0.75,
        "technique_score":   0.68,
        "tactical_insights": ["good pressing triggers"],
        "recommendations":   ["improve weak foot", "reduce turnovers"],
    }

    def _make_analyses(self, n: int = 5) -> list[FrameAnalysis]:
        return [
            FrameAnalysis(
                frame_id=f"f{i}", model="llava", prompt_used="",
                raw_response="", tags=["run"], quality_score=0.7,
            )
            for i in range(n)
        ]

    def test_summarize_session_returns_summary(self):
        va  = _make_analyzer()
        fas = self._make_analyses(5)
        with patch.object(va, "_call_ollama", return_value=self._SUMMARY_JSON):
            summary = va.summarize_session(fas, "football", "athlete")
        assert isinstance(summary, SessionSummary)

    def test_summarize_session_n_clips(self):
        va  = _make_analyzer()
        fas = self._make_analyses(7)
        with patch.object(va, "_call_ollama", return_value=self._SUMMARY_JSON):
            summary = va.summarize_session(fas, "football", "athlete")
        assert summary.n_clips == 7

    def test_summarize_session_highlights(self):
        va  = _make_analyzer()
        fas = self._make_analyses(3)
        with patch.object(va, "_call_ollama", return_value=self._SUMMARY_JSON):
            summary = va.summarize_session(fas, "football", "athlete")
        assert len(summary.highlights) == 2

    def test_summarize_session_scores_clamped(self):
        va  = _make_analyzer()
        fas = self._make_analyses(3)
        bad = {"load_estimate": 99, "technique_score": -5,
               "highlights": [], "tactical_insights": [], "recommendations": []}
        with patch.object(va, "_call_ollama", return_value=bad):
            summary = va.summarize_session(fas, "football", "athlete")
        assert 0.0 <= summary.load_estimate     <= 1.0
        assert 0.0 <= summary.technique_score   <= 1.0

    def test_summarize_empty_analyses(self):
        va      = _make_analyzer()
        summary = va.summarize_session([], "football", "athlete")
        assert summary.n_clips == 0
        assert summary.highlights == []


# ---------------------------------------------------------------------------
# Timeout handling
# ---------------------------------------------------------------------------

class TestTimeoutHandled:
    def test_timeout_in_analyze_frame_returns_default(self):
        """socket.timeout should be caught, not propagate."""
        va = _make_analyzer()
        with patch(
            "urllib.request.urlopen",
            side_effect=TimeoutError("timed out"),
        ):
            result = va.analyze_frame("b64", "football", "athlete")
        assert isinstance(result, FrameAnalysis)
        assert result.quality_score == 0.0

    def test_timeout_in_call_ollama_raises_connection_error(self):
        """_call_ollama converts URLError to ConnectionError."""
        va = _make_analyzer()
        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError("timeout"),
        ):
            with pytest.raises(ConnectionError):
                va._call_ollama("some prompt")
