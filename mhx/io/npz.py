from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np


def savez(path: Path | str, payload: Dict[str, Any]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(p, **payload)
    return p


def load_npz(path: Path | str) -> Dict[str, Any]:
    p = Path(path)
    data = np.load(p, allow_pickle=True)
    return {k: data[k] for k in data.files}

