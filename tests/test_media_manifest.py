"""Regression test for the committed documentation-media contract."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECKER_PATH = ROOT / "tools" / "render_all_media.py"


def _load_checker():
    spec = importlib.util.spec_from_file_location("mhx_render_all_media", CHECKER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_committed_media_contract() -> None:
    checker = _load_checker()
    assert checker.collect_media_errors() == []
