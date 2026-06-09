"""
PRISM Watchdog
Heartbeat loop that monitors Drift Magnitude (Dm = pending WAL entries)
and resurrects the Shadow Pipeline if it dies while mutations are pending.
"""
from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from prism_shadow_pipeline import PrismShadowPipeline

_log = logging.getLogger(__name__)

_DM_THRESHOLD      = 50     # warn when this many entries are pending
_CHECK_INTERVAL    = 30.0   # seconds between heartbeats
_BACKOFF_BASE      = 30.0   # initial resurrection wait (seconds)
_BACKOFF_MAX       = 300.0  # cap at 5 minutes
_ALERT_THRESHOLD   = 5      # alert after this many consecutive failures


class PrismWatchdog:
    """
    Usage::

        watchdog = PrismWatchdog(pipeline)
        watchdog.start()
        ...
        watchdog.stop()
    """

    def __init__(
        self,
        pipeline:       "PrismShadowPipeline",
        dm_threshold:   int   = _DM_THRESHOLD,
        check_interval: float = _CHECK_INTERVAL,
    ) -> None:
        self._pipeline             = pipeline
        self._dm_threshold         = dm_threshold
        self._interval             = check_interval
        self._stop                 = threading.Event()
        self._thread: threading.Thread | None = None
        self._resurrections        = 0
        self._consecutive_failures = 0
        self._backoff_s            = _BACKOFF_BASE

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="prism-watchdog"
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.is_set():
            self._stop.wait(self._interval)
            if self._stop.is_set():
                break
            self._check()

    def _check(self) -> None:
        try:
            from prism_metrics import metrics as _metrics
        except Exception:
            _metrics = None

        status = self._pipeline.status()
        dm     = status["pending"]

        if _metrics:
            _metrics.record_dm(dm)

        if dm > self._dm_threshold:
            _log.warning("Watchdog: Dm=%d exceeds threshold %d — pipeline may be slow",
                         dm, self._dm_threshold)

        if not status["alive"] and dm > 0:
            _log.warning(
                "Watchdog: pipeline dead with %d pending entries — "
                "resurrecting (attempt %d, backoff=%.0fs)",
                dm, self._consecutive_failures + 1, self._backoff_s,
            )
            self._stop.wait(self._backoff_s)
            if self._stop.is_set():
                return

            self._pipeline.start()
            self._resurrections += 1
            if _metrics:
                _metrics.inc("pipeline_restarts")

            # Check if resurrection succeeded
            if not self._pipeline.status()["alive"]:
                self._consecutive_failures += 1
                self._backoff_s = min(self._backoff_s * 2, _BACKOFF_MAX)
                if self._consecutive_failures >= _ALERT_THRESHOLD:
                    _log.error(
                        "Watchdog: pipeline failed to resurrect %d times in a row — "
                        "possible DB corruption or config error",
                        self._consecutive_failures,
                    )
            else:
                self._consecutive_failures = 0
                self._backoff_s = _BACKOFF_BASE

    def status(self) -> dict:
        return {
            "resurrections":        self._resurrections,
            "consecutive_failures": self._consecutive_failures,
            "pipeline":             self._pipeline.status(),
        }
