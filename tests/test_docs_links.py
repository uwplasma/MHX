"""Documentation integrity checks: navigation, links, images, claim levels."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
IMAGE_LINK_RE = re.compile(r"!\[[^\]]*\]\((?P<target>[^)\s]+)(?:\s+[^)]*)?\)")
TOCTREE_RE = re.compile(r"```{toctree}\n(?P<body>.*?)```", re.DOTALL)

REQUIRED_TOCTREE_ENTRIES = {
    "getting_started/install",
    "getting_started/first_run",
    "how_to/run_from_toml",
    "how_to/extend",
    "physics/reduced_mhd",
    "physics/numerics",
    "gallery",
    "validation/index",
    "validation/nonlinear",
    "reference/api",
    "reference/cli",
    "reference/output_schema",
    "reference/literature",
    "examples/autodiff_gradients",
}

# Each documentation area must keep linking the source files it describes.
# The listed paths must exist and appear in the combined text of the area.
REQUIRED_SOURCE_LINKS = {
    ("docs/validation",): {
        "src/mhx/benchmarks/eigenvalue.py",
        "src/mhx/benchmarks/orszag_tang.py",
    },
}


def _area_text(area: tuple[str, ...]) -> str:
    """Concatenate the Markdown text of a file or directory area."""
    chunks = []
    for entry in area:
        path = ROOT / entry
        if path.is_dir():
            for page in sorted(path.rglob("*.md")):
                chunks.append(page.read_text(encoding="utf-8"))
        else:
            chunks.append(path.read_text(encoding="utf-8"))
    return "\n".join(chunks)


def test_required_docs_pages_are_in_toctrees() -> None:
    index = (DOCS / "index.md").read_text(encoding="utf-8")
    entries: set[str] = set()
    for match in TOCTREE_RE.finditer(index):
        entries.update(
            line.strip()
            for line in match.group("body").splitlines()
            if line.strip() and not line.strip().startswith(":")
        )
    assert entries >= REQUIRED_TOCTREE_ENTRIES


def test_every_docs_page_is_reachable_from_a_toctree() -> None:
    """Sphinx -W fails on orphans; this keeps the failure local and clear."""
    entries: set[str] = set()
    toctree_holders = {
        DOCS / "index.md": "",
    }
    for holder, prefix in toctree_holders.items():
        for match in TOCTREE_RE.finditer(holder.read_text(encoding="utf-8")):
            entries.update(
                prefix + line.strip()
                for line in match.group("body").splitlines()
                if line.strip() and not line.strip().startswith(":")
            )
    missing = []
    for page in DOCS.rglob("*.md"):
        if "_build" in page.parts:
            continue
        name = str(page.relative_to(DOCS).with_suffix(""))
        if name == "index":
            continue
        if name not in entries:
            missing.append(name)
    assert missing == []


def test_required_source_links_point_to_existing_paths() -> None:
    for area, expected_paths in REQUIRED_SOURCE_LINKS.items():
        text = _area_text(area)
        for source_path in expected_paths:
            assert (ROOT / source_path).exists(), source_path
            assert source_path in text, f"{area} should link to {source_path}"


def test_docs_image_links_point_to_existing_files() -> None:
    missing = []
    for doc_path in DOCS.rglob("*.md"):
        if "_build" in doc_path.parts:
            continue
        for match in IMAGE_LINK_RE.finditer(doc_path.read_text(encoding="utf-8")):
            target = match.group("target")
            split_target = urlsplit(target)
            if split_target.scheme:
                continue
            linked_path = (doc_path.parent / Path(unquote(split_target.path))).resolve()
            if not linked_path.is_file():
                missing.append(f"{doc_path.relative_to(ROOT)} -> {target}")

    assert missing == []


def test_validation_pages_keep_their_gates() -> None:
    """Every gate section survived the merge into two validation pages."""
    expectations = {
        "index.md": [
            "Exact resistive decay",
            "Diffusion eigenvalue scaffold",
            "FKR growth-rate gate",
            "Direct Harris-sheet tearing eigenvalue gate",
            "Reconnection scaling gates",
        ],
        "nonlinear.md": [
            "Nonlinear Orszag--Tang reduced-MHD gate",
            "Nonlinear reduced-MHD energy budget",
            "Seeded double-Harris long-run validation",
        ],
    }
    for page, headings in expectations.items():
        text = (DOCS / "validation" / page).read_text(encoding="utf-8")
        for heading in headings:
            assert heading in text, f"validation/{page} lost {heading!r}"
