"""Re-execute every tutorial notebook and fail on any execution error.

Documentation builds never run notebooks. This script is the drift guard: a
scheduled CI job re-executes each committed tutorial so stale outputs cannot
survive a solver change silently. It rewrites the notebooks in place, so a
local run followed by ``git diff`` shows any output drift.
"""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbclient import NotebookClient

TUTORIALS = Path(__file__).parents[1] / "docs" / "tutorials"


def main() -> int:
    notebooks = sorted(TUTORIALS.glob("*.ipynb"))
    if not notebooks:
        print("no tutorial notebooks found")
        return 1
    for path in notebooks:
        print(f"executing {path.name}")
        notebook = nbformat.read(path, as_version=4)
        client = NotebookClient(notebook, timeout=600, kernel_name="python3")
        client.execute()
        nbformat.write(notebook, path)
    print(f"executed {len(notebooks)} notebooks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
