"""Sphinx configuration for MHX."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

project = "MHX"
author = "UW Plasma Group"
copyright = "2026, UW Plasma Group"

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinxcontrib.bibtex",
    "sphinxcontrib.video",
    "sphinxext.opengraph",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst-nb",
    ".ipynb": "myst-nb",
}
exclude_patterns = ["_build"]

# MyST settings. Heading anchors give every section a stable link target.
myst_enable_extensions = ["amsmath", "colon_fence", "deflist", "dollarmath"]
myst_heading_anchors = 3

# Tutorials commit their executed outputs. Documentation builds never run
# notebook code, so ReadTheDocs stays inside its build-time budget.
nb_execution_mode = "off"

html_theme = "sphinx_book_theme"
html_title = "MHX"
html_static_path = ["_static"]
html_theme_options = {
    "repository_url": "https://github.com/uwplasma/MHX",
    "use_repository_button": True,
    "use_issues_button": True,
    "path_to_docs": "docs",
    "show_navbar_depth": 1,
}

autodoc_typehints = "description"

bibtex_bibfiles = ["references.bib"]
bibtex_default_style = "alpha"

copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

intersphinx_mapping = {
    "jax": ("https://docs.jax.dev/en/latest/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "python": ("https://docs.python.org/3/", None),
}

ogp_site_url = "https://mhx.readthedocs.io/en/latest/"
ogp_site_name = "MHX documentation"
