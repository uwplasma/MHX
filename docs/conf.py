"""Sphinx configuration for MHX."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

project = "MHX"
author = "UW Plasma Group"
copyright = "2026, UW Plasma Group"
extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
]
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
html_theme = "alabaster"
html_static_path: list[str] = []
autodoc_typehints = "description"
myst_enable_extensions = ["dollarmath", "amsmath"]

