"""
vision_analyzer.py
==================
KDE Sports Agent — Local Vision AI via Ollama LLaVA

Sends frames/images to a local Ollama instance running LLaVA.
Returns structured analysis used to update the decision model.

All HTTP calls use stdlib urllib.request — no third-party HTTP library.
If Ollama is not running, every method degrades gracefully: it returns
an empty/default object and logs a warning.

Requirements:
    Ollama running locally  → https://ollama.ai
    Model pulled            → ollama pull llava
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

OLLAMA_HOST   = "http://localhost:11434"
DEFAULT_MODEL = "llava"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class FrameAnalysis:
    frame_id:        str
    model:           str
    prompt_used:     str
    raw_response:    str
    tags:            list[str]
    quality_score:   float = 0.0
    positions:       dict  = field(default_factory=dict)
    technique_notes: str   = ""


@dataclass
class TechniqueReport:
    video_id:       str
    sport:          str
    profile:        str
    n_frames:       int
    key_findings:   list[str]
    improvements:   list[str]
    strengths:      list[str]
    overall_score:  float
    frame_analyses: list[FrameAnalysis] = field(default_factory=list)


@dataclass
class TacticalContext:
    """Tactical situation extracted from a single frame."""
    formation_guess: str
    ball_zone:       str
    pressure_level:  float
    space_available: float
    phase:           str
    tags:            list[str]


@dataclass
class SessionSummary:
    session_id:        str
    n_clips:           int
    highlights:        list[str]
    load_estimate:     float
    technique_score:   float
    tactical_insights: list[str]
    recommendations:   list[str]


# ---------------------------------------------------------------------------
# VisionAnalyzer
# ---------------------------------------------------------------------------

class VisionAnalyzer:
    """
    Local vision AI using Ollama LLaVA.

    All API calls use stdlib urllib.request.
    If Ollama is not running, methods return empty/default objects
    and log a warning.
    """

    _TIMEOUT = 60  # seconds

    def __init__(
        self,
        host:  str = OLLAMA_HOST,
        model: str = DEFAULT_MODEL,
    ) -> None:
        self._host  = host.rstrip("/")
        self._model = model

    # ── Availability check ───────────────────────────────────────────────────

    def is_available(self) -> bool:
        """Check if Ollama is running and the configured model is pulled."""
        try:
            with urllib.request.urlopen(
                f"{self._host}/api/tags", timeout=5
            ) as resp:
                data   = json.loads(resp.read().decode())
                models = [m.get("name", "") for m in data.get("models", [])]
                return any(
                    m.startswith(self._model) for m in models
                )
        except Exception:
            return False

    # ── Core API call ────────────────────────────────────────────────────────

    def _call_ollama(
        self,
        prompt:      str,
        images:      Optional[list[str]] = None,
        expect_json: bool                = True,
    ) -> dict | str:
        """
        POST {host}/api/generate.
        Raises ConnectionError if Ollama is unreachable.
        """
        payload = {
            "model":  self._model,
            "prompt": prompt,
            "stream": False,
        }
        if images:
            payload["images"] = images

        body = json.dumps(payload).encode()
        req  = urllib.request.Request(
            f"{self._host}/api/generate",
            data    = body,
            headers = {"Content-Type": "application/json"},
            method  = "POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self._TIMEOUT) as resp:
                raw_data = resp.read().decode()
        except urllib.error.URLError as exc:
            raise ConnectionError(
                f"Ollama unreachable at {self._host}: {exc}"
            ) from exc

        outer = json.loads(raw_data)
        text  = outer.get("response", "")

        if not expect_json:
            return text

        # Try to extract a JSON object from the response text
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Look for first {...} block in the text
        start = text.find("{")
        end   = text.rfind("}") + 1
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass

        # Return raw text wrapped in dict as last resort
        return {"raw": text}

    # ── Frame analysis ───────────────────────────────────────────────────────

    def analyze_frame(
        self,
        base64_image: str,
        sport:        str,
        role:         str,
        prompt:       Optional[str] = None,
    ) -> FrameAnalysis:
        """
        Analyse a single frame image (base64) with LLaVA.
        Returns a FrameAnalysis; returns empty default if Ollama is down.
        """
        default_prompt = (
            f"You are a sports performance analyst watching {sport}. "
            f"Describe the action in this frame, rate the quality 1-10, "
            f"describe player positioning, and give one improvement tip. "
            f"Respond ONLY with JSON: "
            f'{{\"quality\":0-10, \"action\":\"...\", '
            f'\"positioning\":{{\"notes\":\"...\"}}, '
            f'\"improvement\":\"...\", \"tags\":[...]}}'
        )
        used_prompt = prompt or default_prompt
        frame_id    = uuid.uuid4().hex

        try:
            data = self._call_ollama(
                used_prompt, images=[base64_image], expect_json=True
            )
        except ConnectionError:
            logger.warning("VisionAnalyzer: Ollama not available; returning empty FrameAnalysis")
            return FrameAnalysis(
                frame_id=frame_id, model=self._model,
                prompt_used=used_prompt, raw_response="",
                tags=[], quality_score=0.0,
            )
        except Exception:
            logger.exception("VisionAnalyzer: analyze_frame error")
            return FrameAnalysis(
                frame_id=frame_id, model=self._model,
                prompt_used=used_prompt, raw_response="",
                tags=[], quality_score=0.0,
            )

        raw_str = json.dumps(data) if isinstance(data, dict) else str(data)

        if isinstance(data, dict):
            raw_quality = data.get("quality", 0)
            try:
                quality = float(raw_quality) / 10.0
            except (ValueError, TypeError):
                quality = 0.0
            tags = data.get("tags", [])
            if not isinstance(tags, list):
                tags = []
            positions = data.get("positioning", {})
            if not isinstance(positions, dict):
                positions = {}
            technique = (
                data.get("improvement", "") or
                data.get("technique_notes", "")
            )
        else:
            quality = 0.0
            tags = []
            positions = {}
            technique = ""

        return FrameAnalysis(
            frame_id        = frame_id,
            model           = self._model,
            prompt_used     = used_prompt,
            raw_response    = raw_str,
            tags            = tags,
            quality_score   = round(quality, 3),
            positions       = positions,
            technique_notes = str(technique),
        )

    # ── Technique report ─────────────────────────────────────────────────────

    def analyze_technique(
        self,
        frames,               # list[Frame] from MediaProcessor
        sport:          str,
        role:           str,
        media_processor = None,
    ) -> TechniqueReport:
        """
        Analyse a sequence of frames for technique quality.
        Samples every Nth frame if >20 frames to keep cost low.
        """
        # Avoid circular import — media_processor passed as argument
        if not frames:
            return TechniqueReport(
                video_id="", sport=sport, profile=role,
                n_frames=0, key_findings=[], improvements=[],
                strengths=[], overall_score=0.0,
            )

        # Sample at most 20 frames
        MAX_FRAMES = 20
        step = max(1, len(frames) // MAX_FRAMES)
        sampled = frames[::step][:MAX_FRAMES]

        video_id = getattr(sampled[0], "video_id", "") if sampled else ""
        analyses: list[FrameAnalysis] = []

        for frame in sampled:
            # Populate base64 if not already done
            if not frame.base64:
                if media_processor is not None:
                    try:
                        b64 = media_processor.frame_to_base64(frame)
                    except Exception:
                        logger.warning(
                            "Could not load base64 for frame %s", frame.frame_id
                        )
                        continue
                else:
                    try:
                        import base64
                        with open(frame.path, "rb") as fh:
                            b64 = base64.b64encode(fh.read()).decode()
                    except Exception:
                        continue
            else:
                b64 = frame.base64

            fa = self.analyze_frame(b64, sport, role)
            analyses.append(fa)

        if not analyses:
            return TechniqueReport(
                video_id=video_id, sport=sport, profile=role,
                n_frames=len(sampled), key_findings=[], improvements=[],
                strengths=[], overall_score=0.0, frame_analyses=[],
            )

        scores       = [a.quality_score for a in analyses if a.quality_score > 0]
        overall      = sum(scores) / len(scores) if scores else 0.0
        all_tags     = [t for a in analyses for t in a.tags]
        improvements = [
            a.technique_notes for a in analyses if a.technique_notes
        ]
        key_findings = list(dict.fromkeys(all_tags))[:10]

        return TechniqueReport(
            video_id        = video_id,
            sport           = sport,
            profile         = role,
            n_frames        = len(sampled),
            key_findings    = key_findings,
            improvements    = improvements[:5],
            strengths       = [],
            overall_score   = round(overall, 3),
            frame_analyses  = analyses,
        )

    # ── Tactical situation ───────────────────────────────────────────────────

    def detect_tactical_situation(
        self,
        base64_image: str,
        sport:        str,
    ) -> TacticalContext:
        """Extract tactical context from a single frame."""
        prompt = (
            f"Identify the tactical situation in this {sport} frame. "
            f"Respond ONLY with JSON: "
            f'{{\"phase\":\"build_up|transition|attack|defend\", '
            f'\"zone\":\"own_third|middle|final_third\", '
            f'\"pressure_0_to_1\":0.0-1.0, '
            f'\"space_0_to_1\":0.0-1.0, '
            f'\"formation\":\"...\", '
            f'\"tags\":[...]}}'
        )

        try:
            data = self._call_ollama(prompt, images=[base64_image], expect_json=True)
        except ConnectionError:
            logger.warning("VisionAnalyzer: Ollama down; returning default TacticalContext")
            return TacticalContext(
                formation_guess="", ball_zone="", pressure_level=0.0,
                space_available=0.0, phase="", tags=[],
            )
        except Exception:
            logger.exception("detect_tactical_situation error")
            return TacticalContext(
                formation_guess="", ball_zone="", pressure_level=0.0,
                space_available=0.0, phase="", tags=[],
            )

        if not isinstance(data, dict):
            data = {}

        try:
            pressure = float(data.get("pressure_0_to_1", 0.0))
        except (ValueError, TypeError):
            pressure = 0.0
        try:
            space = float(data.get("space_0_to_1", 0.0))
        except (ValueError, TypeError):
            space = 0.0
        tags = data.get("tags", [])
        if not isinstance(tags, list):
            tags = []

        return TacticalContext(
            formation_guess = str(data.get("formation", "")),
            ball_zone       = str(data.get("zone", "")),
            pressure_level  = round(min(1.0, max(0.0, pressure)), 3),
            space_available = round(min(1.0, max(0.0, space)), 3),
            phase           = str(data.get("phase", "")),
            tags            = tags,
        )

    # ── Session summary ──────────────────────────────────────────────────────

    def summarize_session(
        self,
        frame_analyses: list[FrameAnalysis],
        sport:          str,
        role:           str,
        session_notes:  str = "",
    ) -> SessionSummary:
        """
        Synthesise all frame analyses into a session summary.
        Uses a text-only Ollama call (no image) to aggregate.
        """
        session_id = uuid.uuid4().hex

        if not frame_analyses:
            return SessionSummary(
                session_id=session_id, n_clips=0,
                highlights=[], load_estimate=0.0,
                technique_score=0.0, tactical_insights=[],
                recommendations=[],
            )

        # Build condensed summary text for the prompt
        tags_all   = [t for a in frame_analyses for t in a.tags]
        scores     = [a.quality_score for a in frame_analyses if a.quality_score > 0]
        avg_score  = sum(scores) / len(scores) if scores else 0.0
        tag_sample = ", ".join(list(dict.fromkeys(tags_all))[:15])

        analyses_summary = (
            f"{len(frame_analyses)} frames analysed. "
            f"Average quality: {avg_score:.2f}/1.0. "
            f"Common tags: {tag_sample}. "
            f"Notes: {session_notes}"
        )

        prompt = (
            f"Given these frame analyses from a {sport} session for a {role}: "
            f"{analyses_summary}. "
            f"Provide a session summary in JSON: "
            f'{{\"highlights\":[...], \"load_estimate\":0.0-1.0, '
            f'\"technique_score\":0.0-1.0, \"tactical_insights\":[...], '
            f'\"recommendations\":[...]}}'
        )

        try:
            data = self._call_ollama(prompt, images=None, expect_json=True)
        except ConnectionError:
            logger.warning("VisionAnalyzer: Ollama down; returning default SessionSummary")
            return SessionSummary(
                session_id=session_id, n_clips=len(frame_analyses),
                highlights=[], load_estimate=avg_score,
                technique_score=avg_score, tactical_insights=[],
                recommendations=[],
            )
        except Exception:
            logger.exception("summarize_session error")
            return SessionSummary(
                session_id=session_id, n_clips=len(frame_analyses),
                highlights=[], load_estimate=0.0, technique_score=0.0,
                tactical_insights=[], recommendations=[],
            )

        if not isinstance(data, dict):
            data = {}

        def _safe_list(key: str) -> list:
            v = data.get(key, [])
            return v if isinstance(v, list) else []

        def _safe_float(key: str, default: float = 0.0) -> float:
            try:
                return float(data.get(key, default))
            except (ValueError, TypeError):
                return default

        return SessionSummary(
            session_id        = session_id,
            n_clips           = len(frame_analyses),
            highlights        = _safe_list("highlights"),
            load_estimate     = round(min(1.0, max(0.0, _safe_float("load_estimate"))), 3),
            technique_score   = round(min(1.0, max(0.0, _safe_float("technique_score"))), 3),
            tactical_insights = _safe_list("tactical_insights"),
            recommendations   = _safe_list("recommendations"),
        )

    # ── Session comparison ───────────────────────────────────────────────────

    def compare_sessions(
        self,
        session_a: SessionSummary,
        session_b: SessionSummary,
    ) -> dict:
        """
        Text-only call: compare two sessions and identify trends.
        Returns: {improvements, regressions, consistency_score, notes}
        """
        prompt = (
            f"Compare two sports sessions.\n"
            f"Session A: technique={session_a.technique_score:.2f}, "
            f"load={session_a.load_estimate:.2f}, "
            f"highlights={session_a.highlights[:3]}\n"
            f"Session B: technique={session_b.technique_score:.2f}, "
            f"load={session_b.load_estimate:.2f}, "
            f"highlights={session_b.highlights[:3]}\n"
            f"Respond ONLY with JSON: "
            f'{{\"improvements\":[...], \"regressions\":[...], '
            f'\"consistency_score\":0.0-1.0, \"notes\":\"...\"}}'
        )

        try:
            data = self._call_ollama(prompt, images=None, expect_json=True)
        except ConnectionError:
            logger.warning("VisionAnalyzer: Ollama down; compare_sessions returning empty dict")
            return {
                "improvements": [], "regressions": [],
                "consistency_score": 0.0, "notes": "",
            }
        except Exception:
            logger.exception("compare_sessions error")
            return {
                "improvements": [], "regressions": [],
                "consistency_score": 0.0, "notes": "",
            }

        if not isinstance(data, dict):
            return {
                "improvements": [], "regressions": [],
                "consistency_score": 0.0, "notes": "",
            }
        return data
