from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np

from mhx.version import NPZ_SCHEMA_VERSION, __version__


def _augment_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    data = dict(payload)
    data.setdefault("schema_version", NPZ_SCHEMA_VERSION)
    data.setdefault("mhx_version", __version__)
    return data


def savez(path: Path | str, payload: Dict[str, Any]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(p, **_augment_payload(payload))
    return p


def load_npz(path: Path | str) -> Dict[str, Any]:
    p = Path(path)
    data = np.load(p, allow_pickle=True)
    return {k: data[k] for k in data.files}
