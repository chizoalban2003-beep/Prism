"""
PRISM Shadow Pipeline
Background daemon that drains the hot buffer into the cold persistent graph
on a fixed interval. Isolated from the user interaction loop.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from prism_memory_graph import PrismMemoryGraph

_log = logging.getLogger(__name__)


class PrismShadowPipeline:
    """
    Drains uncommitted WAL entries into the cold layer on a fixed interval.

    Usage::

        pipeline = PrismShadowPipeline(graph)
        pipeline.start()          # non-blocking; starts daemon thread
        ...
        pipeline.stop()           # graceful shutdown; waits for in-flight commit
    """

    def __init__(
        self,
        graph:        "PrismMemoryGraph",
        interval_s:   float = 5.0,
        max_restarts: int   = 10,
    ) -> None:
        self._graph        = graph
        self._interval     = interval_s
        self._max_restarts = max_restarts
        self._restarts     = 0
        self._stop         = threading.Event()
        self._thread: threading.Thread | None = None
        self._committed_total = 0
        self._last_commit_ts: float = 0.0

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="prism-shadow"
        )
        self._thread.start()
        _log.info("Shadow pipeline started (interval=%.1fs)", self._interval)

    def stop(self, timeout: float = 10.0) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=timeout)
        _log.info("Shadow pipeline stopped (committed_total=%d)", self._committed_total)

    # ── Internal loop ─────────────────────────────────────────────────────────

    def _run(self) -> None:
        try:
            from prism_metrics import metrics as _metrics
        except Exception:
            _metrics = None

        while not self._stop.is_set():
            t0 = time.monotonic()
            try:
                n = self._graph.commit_pending()
                if n:
                    self._committed_total += n
                    self._last_commit_ts = time.monotonic()
                    elapsed = time.monotonic() - t0
                    _log.debug("Pipeline committed %d entries (total=%d)",
                               n, self._committed_total)
                    if _metrics:
                        _metrics.inc("commits_total", n)
                        _metrics.record_latency(elapsed)
                        _metrics.record_dm(self._graph.consistency_psi())
            except Exception as exc:
                self._restarts += 1
                _log.warning("Pipeline error (restart %d/%d): %s",
                             self._restarts, self._max_restarts, exc)
                if _metrics:
                    _metrics.inc("pipeline_restarts")
                if self._restarts >= self._max_restarts:
                    _log.error("Pipeline exceeded max restarts — halting")
                    break
            self._stop.wait(self._interval)

    # ── Introspection ─────────────────────────────────────────────────────────

    @property
    def is_alive(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def status(self) -> dict:
        return {
            "alive":            self.is_alive,
            "committed_total":  self._committed_total,
            "restarts":         self._restarts,
            "last_commit_ts":   self._last_commit_ts,
            "pending":          self._graph.consistency_psi(),
        }
