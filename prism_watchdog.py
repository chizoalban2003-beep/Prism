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

_DM_THRESHOLD   = 50    # warn when this many entries are pending
_CHECK_INTERVAL = 30.0  # seconds between heartbeats


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
        self._pipeline      = pipeline
        self._dm_threshold  = dm_threshold
        self._interval      = check_interval
        self._stop          = threading.Event()
        self._thread: threading.Thread | None = None
        self._resurrections = 0

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
        status = self._pipeline.status()
        dm     = status["pending"]

        if dm > self._dm_threshold:
            _log.warning("Watchdog: Dm=%d exceeds threshold %d — pipeline may be slow",
                         dm, self._dm_threshold)

        if not status["alive"] and dm > 0:
            _log.warning(
                "Watchdog: pipeline dead with %d pending entries — resurrecting", dm
            )
            self._pipeline.start()
            self._resurrections += 1

    def status(self) -> dict:
        return {
            "resurrections": self._resurrections,
            "pipeline":      self._pipeline.status(),
        }
