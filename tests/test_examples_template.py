"""Every example script follows the same template and needs no arguments.

The template is a docstring, input parameters as module-level constants, the
run, then the printed, plotted and saved results. Scripts therefore define no
``main()``, parse no command-line arguments and have no ``__main__`` guard.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from _example_runner import patched_source

EXAMPLES = Path(__file__).parents[1] / "examples"
# Importable modules and package templates, not run scripts.
NOT_SCRIPTS = {
    "__init__.py",
    "local_extension_plugin.py",
    "kelvin_helmholtz.py",
}
SCRIPTS = sorted(
    path
    for path in EXAMPLES.rglob("*.py")
    if path.name not in NOT_SCRIPTS and "plugin_template" not in path.parts
)


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda path: path.relative_to(EXAMPLES).as_posix())
def test_example_follows_the_template(path: Path) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    functions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    guards = [
        node
        for node in tree.body
        if isinstance(node, ast.If) and "__name__" in ast.unparse(node.test)
    ]

    assert ast.get_docstring(tree), "missing module docstring"
    assert "argparse" not in imported
    assert "typer" not in imported
    assert "main" not in functions
    assert not guards, "remove the __main__ guard"


def test_patched_source_replaces_top_level_constants(tmp_path: Path) -> None:
    script = tmp_path / "demo.py"
    script.write_text('"""Demo."""\n\nN = 4\nNAME = "a"\nVALUES = [1, 2]\nresult = N * 2\n')

    patched = patched_source(script, {"N": 8, "NAME": "b", "VALUES": [3]})
    namespace: dict[str, object] = {}
    exec(patched, namespace)

    assert namespace["N"] == 8
    assert namespace["NAME"] == "b"
    assert namespace["VALUES"] == [3]
    assert namespace["result"] == 16
    with pytest.raises(KeyError, match="MISSING"):
        patched_source(script, {"MISSING": 1})
