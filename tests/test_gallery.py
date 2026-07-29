from __future__ import annotations

import ast
from pathlib import Path

GALLERY = Path(__file__).parents[1] / "examples" / "gallery"


def test_gallery_has_one_level_and_uniform_scripts() -> None:
    scripts = sorted(GALLERY.glob("*.py"))

    assert len(scripts) >= 6
    assert not [path for path in GALLERY.iterdir() if path.is_dir()]
    for script in scripts:
        source = script.read_text(encoding="utf-8")
        tree = ast.parse(source)
        function_names = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
        imports = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }

        assert "main" not in function_names
        assert "argparse" not in imports
        assert "Simulation(" in source
        assert ".run()" in source
        assert ".print_summary()" in source
        assert ".plot(" in source or script.name == "06_strong_scaling.py"
        assert ".save(" in source or script.name == "06_strong_scaling.py"
