"""Cheap documentation integrity checks for reviewer-facing pages."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
IMAGE_LINK_RE = re.compile(r"!\[[^\]]*\]\((?P<target>[^)\s]+)(?:\s+[^)]*)?\)")


REQUIRED_TOCTREE_ENTRIES = {
    "validation",
    "benchmarks",
    "reviewer_evidence",
    "long_run_evidence",
    "seed_robust_qi",
    "neural_ode_reproducibility",
    "performance",
    "time_windows",
    "campaigns",
    "campaign_runner",
    "publication_checklist",
    "paper_plan",
    "media",
    "audit",
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
        "src/mhx/neural_ode/reproducibility.py",
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
    "docs/neural_ode_reproducibility.md": {
        "src/mhx/neural_ode/reproducibility.py",
        "src/mhx/neural_ode/__init__.py",
        "src/mhx/cli/main.py",
        "examples/make_neural_ode_reproducibility.py",
        "tests/test_neural_ode_reproducibility.py",
    },
    "docs/reviewer_evidence.md": {
        "src/mhx/benchmarks/seed_robust_qi.py",
        "src/mhx/benchmarks/campaign_runner.py",
        "src/mhx/benchmarks/duration_policy.py",
        "src/mhx/benchmarks/readiness.py",
        "tests/test_campaign_runner.py",
    },
    "docs/campaign_runner.md": {
        "src/mhx/benchmarks/campaigns.py",
        "src/mhx/benchmarks/campaign_runner.py",
        "src/mhx/benchmarks/duration_policy.py",
        "src/mhx/campaigns/production.py",
        "src/mhx/campaigns/__init__.py",
        "tests/test_campaign_runner.py",
        "tests/test_production_campaign.py",
    },
    "docs/publication_checklist.md": {
        "examples/make_validation_media.py",
        "examples/make_readme_media.py",
        "src/mhx/plotting/reduced_mhd.py",
        "tests/test_readme_media.py",
    },
    "docs/media.md": {
        "examples/make_readme_media.py",
        "src/mhx/benchmarks/current_sheet.py",
        "src/mhx/campaigns/production.py",
        "tests/test_current_sheet_eigenvalue_validation.py",
        "tests/test_production_campaign.py",
    },
    "docs/long_run_evidence.md": {
        "src/mhx/benchmarks/nonlinear.py",
        "src/mhx/benchmarks/current_sheet.py",
        "src/mhx/physics/equilibria.py",
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


def test_docs_image_links_point_to_existing_files() -> None:
    missing = []
    for doc_path in DOCS.glob("*.md"):
        for match in IMAGE_LINK_RE.finditer(doc_path.read_text(encoding="utf-8")):
            target = match.group("target")
            split_target = urlsplit(target)
            if split_target.scheme:
                continue
            linked_path = (doc_path.parent / Path(unquote(split_target.path))).resolve()
            if not linked_path.is_file():
                missing.append(f"{doc_path.relative_to(ROOT)} -> {target}")

    assert missing == []


def test_remaining_long_run_gap_is_explicitly_tracked() -> None:
    combined_text = "\n".join(
        (DOCS / name).read_text(encoding="utf-8")
        for name in (
            "index.md",
            "paper_plan.md",
            "validation.md",
            "campaign_runner.md",
            "publication_checklist.md",
        )
    )
    assert "campaign_runner.md" in combined_text
    assert "completed long-run artifact bundle" in combined_text
    assert "production nonlinear result" in combined_text


def test_reviewer_claim_boundaries_are_explicit() -> None:
    reviewer_text = (DOCS / "reviewer_evidence.md").read_text(encoding="utf-8")
    checklist_text = (DOCS / "publication_checklist.md").read_text(encoding="utf-8")
    runner_text = (DOCS / "campaign_runner.md").read_text(encoding="utf-8")

    for claim_level in (
        "`smoke`",
        "`validation`",
        "`production_template`",
        "`production`",
    ):
        assert claim_level in reviewer_text

    assert "not production UQ" in reviewer_text
    assert "not a production nonlinear result" in runner_text
    assert "Rutherford production" in checklist_text
    assert "Neural-ODE" in checklist_text


def test_relocated_validation_content_has_doc_entrypoints() -> None:
    readme_text = (ROOT / "README.md").read_text(encoding="utf-8")
    index_text = (DOCS / "index.md").read_text(encoding="utf-8")
    media_text = (DOCS / "media.md").read_text(encoding="utf-8")
    long_run_text = (DOCS / "long_run_evidence.md").read_text(encoding="utf-8")
    time_window_text = (DOCS / "time_windows.md").read_text(encoding="utf-8")

    for doc_link in (
        "docs/media.md",
        "docs/validation.md",
        "docs/benchmarks.md",
        "docs/long_run_evidence.md",
        "docs/campaign_runner.md",
    ):
        assert doc_link in readme_text

    for doc_entry in ("media", "validation", "long_run_evidence", "time_windows"):
        assert doc_entry in index_text

    assert "_static/validation/periodic_double_harris_seeded_long_run" in media_text
    assert "--t-end 100" in media_text
    assert "readme_media_visual_qa.json" in media_text
    assert "Rutherford-duration executor run" in long_run_text
    assert "Current claim boundary" in long_run_text
    assert "Practical duration labels" in time_window_text
    assert "short_validation" in time_window_text
