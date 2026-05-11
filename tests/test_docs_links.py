"""Cheap documentation integrity checks for reviewer-facing pages."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"


REQUIRED_TOCTREE_ENTRIES = {
    "validation",
    "seed_robust_qi",
    "performance",
    "campaigns",
    "paper_plan",
}

REQUIRED_SOURCE_LINKS = {
    "docs/validation.md": {
        "src/mhx/benchmarks/seed_robust_qi.py",
        "src/mhx/benchmarks/eigenvalue.py",
        "src/mhx/benchmarks/suite.py",
    },
    "docs/performance.md": {
        ".github/workflows/ci.yml",
        "src/mhx/benchmarks/timing.py",
        "tests/test_timing_benchmark.py",
    },
    "docs/paper_plan.md": {
        "src/mhx/benchmarks/campaigns.py",
        "src/mhx/benchmarks/campaign_runner.py",
        "src/mhx/benchmarks/seed_robust_qi.py",
        "src/mhx/cli/main.py",
        "src/mhx/physics/equilibria.py",
    },
    "docs/seed_robust_qi.md": {
        "src/mhx/benchmarks/seed_robust_qi.py",
        "src/mhx/benchmarks/suite.py",
        "src/mhx/cli/main.py",
        "tests/test_seed_robust_qi.py",
    },
}


def test_required_docs_pages_are_in_toctree() -> None:
    index = (DOCS / "index.md").read_text(encoding="utf-8")
    toctree_match = re.search(r"```{toctree}\n(?P<body>.*?)```", index, re.DOTALL)
    assert toctree_match is not None

    entries = {
        line.strip()
        for line in toctree_match.group("body").splitlines()
        if line.strip() and not line.strip().startswith(":")
    }
    assert entries >= REQUIRED_TOCTREE_ENTRIES


def test_required_source_links_point_to_existing_paths() -> None:
    for doc_path, expected_paths in REQUIRED_SOURCE_LINKS.items():
        text = (ROOT / doc_path).read_text(encoding="utf-8")
        for source_path in expected_paths:
            assert (ROOT / source_path).exists(), source_path
            assert source_path in text, f"{doc_path} should link to {source_path}"


def test_remaining_large_push_gap_is_explicitly_tracked() -> None:
    combined_text = "\n".join(
        (DOCS / name).read_text(encoding="utf-8")
        for name in ("index.md", "paper_plan.md", "validation.md")
    )
    assert "campaign_runner.md" in combined_text
    assert "long runner exists" in combined_text or "planned" in combined_text
