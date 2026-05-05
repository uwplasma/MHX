#!/usr/bin/env python3
"""Fail if active source files import archived MHX modules or legacy scripts."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

LEGACY_TOP_LEVEL_MODULES = {
    "mhd_linear_benchmarks",
    "mhd_reconnection_rate",
    "mhd_tearing_energy_plasmoid_opt",
    "mhd_tearing_ideal_tearing_opt",
    "mhd_tearing_inverse_design",
    "mhd_tearing_inverse_design_figures",
    "mhd_tearing_island_evolution",
    "mhd_tearing_ml",
    "mhd_tearing_ml_v2",
    "mhd_tearing_postprocess",
    "mhd_tearing_postprocess_ml",
    "mhd_tearing_postprocess_ml_v2",
    "mhd_tearing_scan",
    "mhd_tearing_solve",
    "run_MHD",
    "run_MHD_box",
}

DEFAULT_ROOTS = ("src", "tests", "examples", "tools")


def _import_roots(node: ast.AST) -> tuple[str, ...]:
    if isinstance(node, ast.Import):
        return tuple(alias.name.split(".", maxsplit=1)[0] for alias in node.names)
    if isinstance(node, ast.ImportFrom) and node.module:
        root = node.module.split(".", maxsplit=1)[0]
        return (root,)
    return ()


def find_legacy_imports(paths: tuple[Path, ...]) -> list[str]:
    """Return active-file legacy imports as human-readable violations."""
    violations: list[str] = []
    for root in paths:
        if not root.exists():
            continue
        files = [root] if root.is_file() else sorted(root.rglob("*.py"))
        for path in files:
            if "legacy" in path.parts or "__pycache__" in path.parts:
                continue
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            except SyntaxError as exc:
                violations.append(f"{path}:{exc.lineno}: cannot parse Python source")
                continue
            for node in ast.walk(tree):
                for root_name in _import_roots(node):
                    if root_name == "legacy" or root_name in LEGACY_TOP_LEVEL_MODULES:
                        line = getattr(node, "lineno", 1)
                        violations.append(f"{path}:{line}: imports archived module {root_name!r}")
    return violations


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    raw_paths = tuple(argv if argv is not None else sys.argv[1:])
    paths = tuple(Path(item) for item in (raw_paths or DEFAULT_ROOTS))
    violations = find_legacy_imports(paths)
    if violations:
        print("Legacy import guard failed:", file=sys.stderr)
        for violation in violations:
            print(f"- {violation}", file=sys.stderr)
        return 1
    print("Legacy import guard passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
