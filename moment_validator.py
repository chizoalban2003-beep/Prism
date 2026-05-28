"""
moment_validator.py
===================
KDE Moment Platform — Season-Scale Validation

Runs an entire StatsBomb season through MomentAnalyzer and reports accuracy.

Two accuracy metrics:
  1. Action accuracy:  did the model recommend the action the player actually took?
     Target: > 55% (random baseline: 1/n_options ≈ 12% for 8-option beam)
  2. xG Brier score:   mean squared error between raw xG and actual goal (0 or 1)
     Target: < 0.18 (well-calibrated xG models score ~0.16)
"""

from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from moment_analyzer import MomentAnalyzer, MomentResult
from moment_pipeline import StatsBombMomentPipeline
from sport_data import StatsBombConnector

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result data classes
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    """Accuracy report for one player/zone/moment_type slice."""
    slice_key:       str    # e.g. "Salah:box:1v1_keeper"
    n_moments:       int
    correct:         int    # model recommendation matched actual action
    accuracy:        float
    avg_xg_pred:     float  # model predicted xG (from moment.xg_raw)
    avg_xg_actual:   float  # actual xG realized (1.0=goal, 0.0=miss)
    xg_brier_score:  float  # mean((pred_xg - actual_xg)^2) — lower is better
    calibration_gap: float  # avg_xg_pred - avg_xg_actual (bias)


@dataclass
class SeasonValidationReport:
    """Full validation report for one competition/season."""
    season:            str
    competition:       str
    n_matches:         int
    n_moments:         int
    overall_accuracy:  float
    overall_brier:     float
    by_player:         list[ValidationResult]
    by_zone:           list[ValidationResult]
    by_moment_type:    list[ValidationResult]
    best_calibrated:   list[str]   # player names with lowest brier score
    worst_calibrated:  list[str]   # player names with highest brier score


# ---------------------------------------------------------------------------
# Internal tracking analyser
# ---------------------------------------------------------------------------

class _TrackedMomentAnalyzer(MomentAnalyzer):
    """
    Thin wrapper around MomentAnalyzer that records every
    (MomentResult, ActionOutcome) pair for post-hoc validation.
    """

    def __init__(self) -> None:
        super().__init__()
        # list of (result, outcome_dict)
        # outcome_dict: {"action_taken": str, "success": bool, "xg_delta": float}
        self.tracked: list[tuple[MomentResult, dict]] = []
        self._last_result: Optional[MomentResult] = None

    def analyze(self, moment):
        result = super().analyze(moment)
        self._last_result = result
        return result

    def calibrate(self, moment, outcome):
        super().calibrate(moment, outcome)
        if self._last_result is not None:
            self.tracked.append((
                self._last_result,
                {
                    "action_taken": outcome.action_taken,
                    "success":      outcome.success,
                    "xg_delta":     outcome.xg_delta,
                },
            ))
            self._last_result = None


# ---------------------------------------------------------------------------
# MomentValidator
# ---------------------------------------------------------------------------

class MomentValidator:
    """
    Runs a full StatsBomb season through MomentAnalyzer and measures accuracy.

    Usage::

        validator = MomentValidator(competition_id=11, season_id=90)
        report = validator.run()
        print(f"Overall accuracy: {report.overall_accuracy:.1%}")
        print(f"Brier score: {report.overall_brier:.3f}")
    """

    def __init__(
        self,
        competition_id: int,
        season_id:      int,
        profile_map:    Optional[dict[str, str]] = None,
        cache_dir:      str = "~/.kde/statsbomb_cache",
        n_matches:      Optional[int] = None,
        sport:          str = "Football",
    ) -> None:
        self.competition_id = competition_id
        self.season_id      = season_id
        self.profile_map    = profile_map or {}
        self.cache_dir      = cache_dir
        self.n_matches      = n_matches
        self.sport          = sport

    # ------------------------------------------------------------------

    def run(self) -> SeasonValidationReport:
        """
        Steps:
          1. Load match list from StatsBomb API (or cache)
          2. For each match: process through StatsBombMomentPipeline
          3. Collect all (MomentResult, outcome_dict) pairs
          4. Compute overall and sliced accuracy metrics
          5. Return SeasonValidationReport
        """
        connector = StatsBombConnector(cache_dir=self.cache_dir)
        analyzer  = _TrackedMomentAnalyzer()
        pipeline  = StatsBombMomentPipeline(
            analyzer    = analyzer,
            profile_map = self.profile_map,
            sport       = self.sport,
        )

        matches = connector.get_matches(self.competition_id, self.season_id)
        if self.n_matches is not None:
            matches = matches[: self.n_matches]

        processed_matches = 0
        for match in matches:
            match_id = str(match.get("match_id", "unknown"))
            try:
                events       = connector.get_match_events(int(match_id))
                freeze_frames = connector.get_match_freeze_frames(int(match_id))
                pipeline.process_match(events, freeze_frames, match_id=match_id)
                processed_matches += 1
            except Exception as exc:
                logger.warning("Skipping match %s: %s", match_id, exc)

        tracked = analyzer.tracked
        n_moments = len(tracked)

        # Gather competition/season label
        competition_label = str(self.competition_id)
        season_label      = str(self.season_id)

        if n_moments == 0:
            return SeasonValidationReport(
                season           = season_label,
                competition      = competition_label,
                n_matches        = processed_matches,
                n_moments        = 0,
                overall_accuracy = 0.0,
                overall_brier    = 1.0,
                by_player        = [],
                by_zone          = [],
                by_moment_type   = [],
                best_calibrated  = [],
                worst_calibrated = [],
            )

        # Overall accuracy: recommended == actual action (case-insensitive partial match)
        correct_total = sum(
            1 for result, outcome in tracked
            if self._actions_match(result.recommended, outcome["action_taken"])
        )
        overall_accuracy = correct_total / n_moments

        # Overall Brier score
        preds   = [result.moment.xg_raw for result, _ in tracked]
        actuals = [1.0 if o["success"] else 0.0 for _, o in tracked]
        overall_brier = self._brier_score(preds, actuals)

        # Sliced reports
        by_player      = self.slice_by(tracked, "player")
        by_zone        = self.slice_by(tracked, "zone")
        by_moment_type = self.slice_by(tracked, "moment_type")

        # Best/worst calibrated players (by brier score)
        player_results_sorted = sorted(
            [r for r in by_player if r.n_moments >= 3],
            key=lambda r: r.xg_brier_score,
        )
        best_calibrated  = [r.slice_key for r in player_results_sorted[:5]]
        worst_calibrated = [r.slice_key for r in reversed(player_results_sorted[-5:])]

        return SeasonValidationReport(
            season           = season_label,
            competition      = competition_label,
            n_matches        = processed_matches,
            n_moments        = n_moments,
            overall_accuracy = overall_accuracy,
            overall_brier    = overall_brier,
            by_player        = by_player,
            by_zone          = by_zone,
            by_moment_type   = by_moment_type,
            best_calibrated  = best_calibrated,
            worst_calibrated = worst_calibrated,
        )

    # ------------------------------------------------------------------

    def slice_by(
        self,
        results: list[tuple[MomentResult, dict]],
        key:     str,   # "player" | "zone" | "moment_type"
    ) -> list[ValidationResult]:
        """
        Slice validation results by player, zone, or moment_type and
        compute accuracy metrics per slice.
        """
        buckets: dict[str, list[tuple[MomentResult, dict]]] = defaultdict(list)

        for result, outcome in results:
            moment = result.moment
            if key == "player":
                bucket_key = moment.focal_player
            elif key == "zone":
                bucket_key = self._zone_label(moment.pitch_x)
            else:  # moment_type
                bucket_key = moment.moment_type
            buckets[bucket_key].append((result, outcome))

        validation_results: list[ValidationResult] = []
        for slice_key, bucket in sorted(buckets.items()):
            n       = len(bucket)
            correct = sum(
                1 for r, o in bucket
                if self._actions_match(r.recommended, o["action_taken"])
            )
            preds   = [r.moment.xg_raw for r, _ in bucket]
            actuals = [1.0 if o["success"] else 0.0 for _, o in bucket]
            brier   = self._brier_score(preds, actuals)
            avg_pred   = sum(preds) / n   if n else 0.0
            avg_actual = sum(actuals) / n if n else 0.0
            validation_results.append(ValidationResult(
                slice_key       = slice_key,
                n_moments       = n,
                correct         = correct,
                accuracy        = correct / n if n else 0.0,
                avg_xg_pred     = avg_pred,
                avg_xg_actual   = avg_actual,
                xg_brier_score  = brier,
                calibration_gap = avg_pred - avg_actual,
            ))

        return validation_results

    # ------------------------------------------------------------------

    def _brier_score(
        self,
        predicted: list[float],
        actual:    list[float],
    ) -> float:
        """
        Mean squared error between predicted xG and actual outcome (0 or 1).
        Lower = better calibrated.  Perfect = 0.0.  Random = ~0.25.
        """
        if not predicted:
            return 1.0
        return sum((p - a) ** 2 for p, a in zip(predicted, actual)) / len(predicted)

    # ------------------------------------------------------------------

    def compare_profiles(
        self,
        players: list[str],
        report:  SeasonValidationReport,
    ) -> list[dict]:
        """
        Compare model accuracy across named players.
        Returns sorted list: [{player, accuracy, brier, n_moments}, ...]
        """
        player_map = {r.slice_key: r for r in report.by_player}
        out = []
        for player in players:
            if player in player_map:
                r = player_map[player]
                out.append({
                    "player":    player,
                    "accuracy":  r.accuracy,
                    "brier":     r.xg_brier_score,
                    "n_moments": r.n_moments,
                })
            else:
                out.append({
                    "player":    player,
                    "accuracy":  0.0,
                    "brier":     1.0,
                    "n_moments": 0,
                })
        out.sort(key=lambda x: x["accuracy"], reverse=True)
        return out

    # ------------------------------------------------------------------

    def export_report(
        self,
        report: SeasonValidationReport,
        path:   str,
        fmt:    str = "markdown",   # "markdown" | "json" | "html"
    ) -> str:
        """Write validation report to file.  Return path."""
        if fmt == "json":
            import dataclasses as _dc
            content = json.dumps(_dc.asdict(report), indent=2, default=str)
        elif fmt == "html":
            content = self._to_html(report)
        else:  # markdown (default)
            content = self._to_markdown(report)

        with open(path, "w", encoding="utf-8") as fh:
            fh.write(content)
        return path

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _zone_label(pitch_x: float) -> str:
        """Classify normalised pitch_x into a zone label."""
        if pitch_x > 0.83:
            return "box"
        if pitch_x > 0.67:
            return "attacking_third"
        if pitch_x > 0.33:
            return "midfield"
        return "defensive_third"

    @staticmethod
    def _actions_match(recommended: str, actual: str) -> bool:
        """
        Return True when the recommended action matches the actual.
        Uses a lenient word-overlap check (any word with len>3 in recommended
        appears in actual, case-insensitive) so that minor label differences
        still count as a match.
        """
        if not actual:
            return False
        rec_lower    = recommended.lower()
        actual_lower = actual.lower()
        # Exact match
        if rec_lower == actual_lower:
            return True
        # Any significant word overlap
        for word in rec_lower.split():
            if len(word) > 3 and word in actual_lower:
                return True
        return False

    def _to_markdown(self, report: SeasonValidationReport) -> str:
        lines = [
            f"## Season Validation Report",
            f"",
            f"- **Competition:** {report.competition}",
            f"- **Season:** {report.season}",
            f"- **Matches:** {report.n_matches}",
            f"- **Moments:** {report.n_moments}",
            f"- **Overall accuracy:** {report.overall_accuracy:.1%}",
            f"- **Overall Brier score:** {report.overall_brier:.4f}",
            f"",
            f"## By Moment Type",
            f"",
            f"| Moment Type | N | Accuracy | Brier | Cal Gap |",
            f"|---|---|---|---|---|",
        ]
        for r in report.by_moment_type:
            lines.append(
                f"| {r.slice_key} | {r.n_moments} "
                f"| {r.accuracy:.1%} | {r.xg_brier_score:.4f} "
                f"| {r.calibration_gap:+.3f} |"
            )
        lines += [
            f"",
            f"## By Zone",
            f"",
            f"| Zone | N | Accuracy | Brier | Cal Gap |",
            f"|---|---|---|---|---|",
        ]
        for r in report.by_zone:
            lines.append(
                f"| {r.slice_key} | {r.n_moments} "
                f"| {r.accuracy:.1%} | {r.xg_brier_score:.4f} "
                f"| {r.calibration_gap:+.3f} |"
            )
        lines += [
            f"",
            f"## By Player (top 20)",
            f"",
            f"| Player | N | Accuracy | Brier | Cal Gap |",
            f"|---|---|---|---|---|",
        ]
        for r in sorted(report.by_player, key=lambda x: -x.n_moments)[:20]:
            lines.append(
                f"| {r.slice_key} | {r.n_moments} "
                f"| {r.accuracy:.1%} | {r.xg_brier_score:.4f} "
                f"| {r.calibration_gap:+.3f} |"
            )
        lines += [
            f"",
            f"## Calibration",
            f"",
            f"**Best calibrated:** {', '.join(report.best_calibrated) or 'N/A'}",
            f"",
            f"**Worst calibrated:** {', '.join(report.worst_calibrated) or 'N/A'}",
        ]
        return "\n".join(lines) + "\n"

    def _to_html(self, report: SeasonValidationReport) -> str:
        rows = "".join(
            f"<tr><td>{r.slice_key}</td><td>{r.n_moments}</td>"
            f"<td>{r.accuracy:.1%}</td><td>{r.xg_brier_score:.4f}</td>"
            f"<td>{r.calibration_gap:+.3f}</td></tr>"
            for r in report.by_moment_type
        )
        return (
            f"<html><body>"
            f"<h2>Season Validation Report</h2>"
            f"<p>Competition: {report.competition} | Season: {report.season} | "
            f"Matches: {report.n_matches} | Moments: {report.n_moments}</p>"
            f"<p>Overall accuracy: {report.overall_accuracy:.1%} &nbsp; "
            f"Brier: {report.overall_brier:.4f}</p>"
            f"<h3>By Moment Type</h3>"
            f"<table border='1'><tr><th>Type</th><th>N</th>"
            f"<th>Accuracy</th><th>Brier</th><th>Cal Gap</th></tr>"
            f"{rows}</table>"
            f"</body></html>\n"
        )
