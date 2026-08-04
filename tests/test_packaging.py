from __future__ import annotations

import re
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib

ROOT = Path(__file__).parents[1]


def test_runtime_dependencies_are_unpinned_and_solvax_is_required() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    dependencies = project["dependencies"]
    names = {re.split(r"[ ;<>=!~]", dependency, maxsplit=1)[0] for dependency in dependencies}

    assert "solvax" in names
    assert "matplotlib" in names
    for dependency in dependencies:
        package = dependency.split(";", maxsplit=1)[0].strip()
        assert not re.search(r"[<>=!~]", package), dependency


def test_optional_dependencies_have_two_clear_groups() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]

    assert set(project["optional-dependencies"]) == {"dev", "research"}
