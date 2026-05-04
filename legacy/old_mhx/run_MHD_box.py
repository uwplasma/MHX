#!/usr/bin/env python3
from __future__ import annotations

import runpy
import warnings
from pathlib import Path


def main() -> None:
    warnings.warn(
        "Deprecated: this legacy script moved to scripts/legacy/run_MHD_box.py. "
        "Prefer `mhx simulate`.",
        DeprecationWarning,
    )
    target = Path(__file__).parent / "scripts" / "legacy" / "run_MHD_box.py"
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
