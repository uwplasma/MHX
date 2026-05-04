"""Version and public API identifiers."""

from __future__ import annotations

import os
import warnings
from typing import Dict

__version__ = "0.2.0"

SIM_API_VERSION = "1"
INVERSE_API_VERSION = "1"
PHYSICS_API_VERSION = "1"
NPZ_SCHEMA_VERSION = "1"


def expected_api_versions() -> Dict[str, str]:
    """Return expected API versions, honoring MHX_API_VERSION overrides.

    Supported keys: sim, inverse, physics, npz.
    Formats:
      - "1" (apply to all)
      - "sim=1,inverse=1,physics=1,npz=1"
    """
    expected = {
        "sim": SIM_API_VERSION,
        "inverse": INVERSE_API_VERSION,
        "physics": PHYSICS_API_VERSION,
        "npz": NPZ_SCHEMA_VERSION,
    }
    override = os.getenv("MHX_API_VERSION", "").strip()
    if not override:
        return expected
    if override.isdigit():
        return {k: override for k in expected}
    for chunk in override.split(","):
        if not chunk:
            continue
        if "=" not in chunk:
            continue
        key, value = [s.strip() for s in chunk.split("=", 1)]
        if key in expected and value:
            expected[key] = value
    return expected


def check_api_version(kind: str, actual: str | None) -> None:
    """Validate API versions; warn on mismatch unless overridden."""
    if actual is None:
        return
    expected = expected_api_versions().get(kind)
    if expected is None:
        return
    if str(actual) != str(expected):
        msg = f"API version mismatch for {kind}: expected {expected}, got {actual}."
        if os.getenv("MHX_API_VERSION"):
            raise ValueError(msg)
        warnings.warn(msg)
