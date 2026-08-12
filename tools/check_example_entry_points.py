"""Import every example entry point, so they are exercised on every Python we claim.

The CI matrix runs the test suite across the full supported range, but the jobs
that actually execute example scripts pin a single interpreter. That left the
examples covered on one version only, and a stdlib symbol newer than the
declared floor could reach main unnoticed.

Only scripts guarded by ``if __name__ == "__main__":`` are imported here: they
define functions and constants at module scope, so importing them is fast and
side-effect free. Scripts without that guard run their simulation on import and
are deliberately skipped, since importing them is not cheap.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
EXAMPLES = ROOT / "examples"
# The plugin template is an independently packaged example with its own tests.
SKIP_PARTS = {"plugin_template"}


def has_main_guard(path: Path) -> bool:
    """Return True when the module body ends in an ``if __name__ == "__main__"`` block."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return False
    for node in tree.body:
        if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
            continue
        left = node.test.left
        comparators = node.test.comparators
        if not isinstance(left, ast.Name) or left.id != "__name__":
            continue
        if any(isinstance(c, ast.Constant) and c.value == "__main__" for c in comparators):
            return True
    return False


def entry_points() -> list[Path]:
    return sorted(
        path
        for path in EXAMPLES.rglob("*.py")
        if not SKIP_PARTS.intersection(path.parts) and has_main_guard(path)
    )


def main() -> int:
    targets = entry_points()
    if not targets:
        print("no example entry points found", file=sys.stderr)
        return 1

    failures = 0
    for path in targets:
        relative = path.relative_to(ROOT)
        spec = importlib.util.spec_from_file_location(f"_entry_point_{path.stem}", path)
        if spec is None or spec.loader is None:
            print(f"FAIL {relative}: could not build an import spec", file=sys.stderr)
            failures += 1
            continue
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        try:
            spec.loader.exec_module(module)
        except Exception as exc:  # noqa: BLE001 - report every failure, do not stop at the first
            print(f"FAIL {relative}: {type(exc).__name__}: {exc}", file=sys.stderr)
            failures += 1
        else:
            print(f"ok   {relative}")

    if failures:
        print(f"{failures} example entry point(s) failed to import", file=sys.stderr)
        return 1
    print(f"{len(targets)} example entry points import cleanly on {sys.version.split()[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
