from __future__ import annotations

from tools.check_legacy_imports import find_legacy_imports


def test_legacy_import_guard_flags_archived_modules(tmp_path) -> None:
    source = tmp_path / "bad.py"
    source.write_text("import mhd_tearing_solve\nfrom legacy.old_mhx import run_MHD\n")

    violations = find_legacy_imports((source,))

    assert len(violations) == 2
    assert "mhd_tearing_solve" in violations[0]
    assert "legacy" in violations[1]


def test_legacy_import_guard_accepts_active_package_imports(tmp_path) -> None:
    source = tmp_path / "good.py"
    source.write_text("from mhx.config import RunConfig\n")

    assert find_legacy_imports((source,)) == []
