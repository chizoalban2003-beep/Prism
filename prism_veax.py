"""
prism_veax.py — Canonical VEAX single source of truth.

VEAX axes (each a float in [0.0, 1.0]):
    V — Verification  : 0 = accept all, 1 = require strict proof
    E — Evolution     : 0 = anchor bias, 1 = high plasticity
    A — Autonomy      : 0 = human oversight, 1 = fully autonomous
    X — Explanation   : 0 = silent execution, 1 = full audit traces

Previously, VEAX state was split across:
  - prism_spectrum_middleware  (SpectrumGates, VEAX_PRESETS, load/save state)
  - organs/veax_control.py    (NL control organ)
  - prism_perception.py       (BiometricVEAXBridge)

Import from here instead of reaching into those modules directly.
"""
from prism_spectrum_middleware import _DEFAULTS as VEAX_DEFAULTS
from prism_spectrum_middleware import (
    VEAX_PRESETS,
    SpectrumGates,
    get_current_gates,
    get_current_network,
    load_spectrum,
    nl_to_veax,
    render_gates,
    save_spectrum_state,
    set_current_gates,
)

__all__ = [
    # Core type
    "SpectrumGates",
    # State accessors
    "get_current_gates",
    "set_current_gates",
    "get_current_network",
    "load_spectrum",
    "save_spectrum_state",
    # NL control
    "nl_to_veax",
    # Display
    "render_gates",
    # Constants
    "VEAX_PRESETS",
    "VEAX_DEFAULTS",
]
