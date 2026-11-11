"""Run an example script with its top-level input parameters overridden.

Examples follow one template: a docstring, input parameters as module-level
constants, the run, then the printed, plotted and saved results. Tests (and the
Modal launcher) shrink a run by replacing those constants before executing the
script, so no example needs command-line arguments.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")


def patched_source(path: Path, overrides: dict[str, Any]) -> str:
    """Return the script source with simple top-level assignments replaced.

    Args:
        path: The example script.
        overrides: Maps a module-level constant name to its new literal value.
            A ``Path`` value becomes ``Path('...')``, so the script must import ``Path``.

    Raises:
        KeyError: If a name in ``overrides`` is not assigned at module level.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in overrides
        ):
            name = node.targets[0].id
            value = overrides[name]
            expression = f"Path({str(value)!r})" if isinstance(value, Path) else repr(value)
            node.value = ast.parse(expression, mode="eval").body
            found.add(name)
    missing = sorted(set(overrides) - found)
    if missing:
        raise KeyError(f"{path.name} has no module-level assignment for: {', '.join(missing)}")
    return ast.unparse(ast.fix_missing_locations(tree))


def run_example(
    path: Path,
    workdir: Path,
    overrides: dict[str, Any] | None = None,
    *,
    monkeypatch: Any,
) -> dict[str, Any]:
    """Execute an example in ``workdir`` and return its module namespace.

    The script runs with its original ``__file__``, so paths computed from it
    still resolve, and with ``workdir`` as the current directory so relative
    output folders stay inside the test's temporary directory.
    """
    source = patched_source(path, overrides or {})
    monkeypatch.chdir(workdir)
    namespace: dict[str, Any] = {"__file__": str(path), "__name__": "__example__"}
    try:
        exec(compile(source, str(path), "exec"), namespace)  # noqa: S102 - trusted examples
    finally:
        import matplotlib.pyplot as plt

        # Scripts leave their figures open; close them so later tests see none.
        plt.close("all")
    return namespace
