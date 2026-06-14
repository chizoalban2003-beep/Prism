"""
PRISM Shadow Pipeline
Background daemon that drains the hot buffer into the cold persistent graph
on a fixed interval. Isolated from the user interaction loop.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from prism_memory_graph import PrismMemoryGraph
    from prism_soul import PrismSoul

try:
    import prism_phase as _prism_phase_mod
except ImportError:
    _prism_phase_mod = None  # type: ignore[assignment]

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

    # Entailment check runs every N commit cycles to amortize the cost
    _ENTAILMENT_INTERVAL = 12

    def __init__(
        self,
        graph:         PrismMemoryGraph,
        interval_s:    float = 5.0,
        max_restarts:  int   = 10,
        soul: Optional[PrismSoul | None] = None,
        phase_engine:  Any | None = None,
        bridge:        Any | None = None,
        kinetic:       Any | None = None,
    ) -> None:
        self._graph        = graph
        self._interval     = interval_s
        self._max_restarts = max_restarts
        self._soul         = soul
        self._phase_engine = phase_engine
        self._bridge       = bridge
        self._kinetic      = kinetic
        self._restarts     = 0
        self._stop         = threading.Event()
        self._thread: threading.Thread | None = None
        self._committed_total = 0
        self._last_commit_ts: float = 0.0
        self._commit_cycles: int = 0

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
        _metrics: Any
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
                    self._commit_cycles += 1
                    _log.debug("Pipeline committed %d entries (total=%d)",
                               n, self._committed_total)
                    if _metrics:
                        _metrics.inc("commits_total", n)
                        _metrics.record_latency(elapsed)
                        _metrics.record_dm(self._graph.consistency_psi())

                # Periodic soul entailment check (every _ENTAILMENT_INTERVAL cycles)
                if (self._soul is not None
                        and self._commit_cycles > 0
                        and self._commit_cycles % self._ENTAILMENT_INTERVAL == 0):
                    try:
                        new_contradictions = self._soul.run_entailment_check()
                        if new_contradictions:
                            _log.info("Entailment check: %d new contradictions found",
                                      len(new_contradictions))
                    except Exception as ec:
                        _log.debug("Entailment check error: %s", ec)

                # Phase engine feedback loop — after each commit cycle
                if self._phase_engine is not None:
                    try:
                        reading = self._phase_engine.compute(
                            soul=self._soul,
                            bridge=getattr(self, "_bridge", None),
                            kinetic=getattr(self, "_kinetic", None),
                        )
                        if self._phase_engine.should_melt():
                            _log.info(
                                "[shadow] Φ_melt=%.3f → %s — applying VEAX delta",
                                reading.phi, reading.phase.value,
                            )
                            deltas = self._phase_engine.veax_delta(reading.phase)
                            if deltas:
                                from prism_veax import (
                                    SpectrumGates,
                                    get_current_gates,
                                    save_spectrum_state,
                                )
                                current = get_current_gates()
                                if current is not None:
                                    new_vals = {
                                        "V": current.V,
                                        "E": current.E,
                                        "A": current.A,
                                        "X": current.X,
                                    }
                                    for axis, val in deltas.items():
                                        if axis in new_vals:
                                            # LIQUID uses absolute values; others are deltas
                                            if reading.phase.value == "LIQUID":
                                                new_vals[axis] = max(0.0, min(1.0, val))
                                            else:
                                                new_vals[axis] = max(0.0, min(1.0,
                                                    new_vals[axis] + val))
                                    save_spectrum_state(SpectrumGates(**new_vals))
                    except Exception as _pe:
                        _log.debug("[shadow] phase engine error: %s", _pe)

                # Silicon policy update — log current budget when throttle active
                if self._phase_engine is not None and _prism_phase_mod is not None:
                    try:
                        import prism_silicon_policy as _sp

                        bridge_db = getattr(self, "_bridge", None)
                        delta_b = bridge_db.biological_pressure() if bridge_db is not None else 0.0
                        reading_phase = reading.phase.value if "reading" in dir() else "STABLE"
                        budget = _sp.get_policy().current_budget(delta_b=delta_b, phase_name=reading_phase)
                        if budget.throttle_reason:
                            _log.info(
                                "[shadow] silicon budget active: %s (tokens≤%d cap≤%d)",
                                budget.throttle_reason,
                                budget.max_tokens,
                                budget.capability_ceil,
                            )
                    except Exception:
                        pass
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
