from __future__ import annotations

import json
from pathlib import Path

import pytest

NOTEBOOKS = (
    "examples/run_kelvin_helmholtz_incompressible.ipynb",
    "examples/run_kelvin_helmholtz_backpropagation.ipynb",
)


@pytest.mark.slow
@pytest.mark.parametrize("notebook_path", NOTEBOOKS)
def test_clean_kelvin_helmholtz_notebook_executes(
    notebook_path: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MHX_EXAMPLE_OUTDIR_ROOT", str(tmp_path))
    notebook = json.loads(Path(notebook_path).read_text())
    namespace: dict[str, object] = {"__name__": "__main__"}

    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        exec(compile(source, f"{notebook_path}:cell{index}", "exec"), namespace)

    if "incompressible" in notebook_path:
        output_dir = tmp_path / "kelvin_helmholtz_incompressible"
        assert (output_dir / "kh_incompressible_snapshots.png").exists()
        assert (output_dir / "kh_incompressible_entropy.png").exists()
    else:
        output_dir = tmp_path / "kelvin_helmholtz_backpropagation"
        assert (output_dir / "kh_backpropagation_history.png").exists()


def test_clean_kelvin_helmholtz_notebooks_are_output_free() -> None:
    for notebook_path in NOTEBOOKS:
        notebook = json.loads(Path(notebook_path).read_text())
        for cell in notebook["cells"]:
            if cell.get("cell_type") == "code":
                assert cell.get("outputs", []) == []
                assert cell.get("execution_count") is None
