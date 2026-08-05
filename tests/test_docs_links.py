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
    "how_to/extend_physics",
    "how_to/add_diagnostics",
    "physics/reduced_mhd",
    "validation/index",
    "validation/exact_limits",
    "validation/linear_tearing",
    "validation/nonlinear",
    "validation/reconnection_campaigns",
    "validation/scaling_theory",
    "reference/api/index",
    "reference/cli",
    "reference/output_schema",
    "reference/performance",
    "develop/style",
    "develop/release",
    "project/media_inventory",
    "project/reviewer_evidence",
    "project/publication_checklist",
    "project/campaign_runner",
    "project/long_run_evidence",
}

# Each documentation area must keep linking the source files it describes.
# The listed paths must exist and appear in the combined text of the area.
REQUIRED_SOURCE_LINKS = {
    ("docs/validation",): {
        "src/mhx/benchmarks/seed_robust_qi.py",
        "src/mhx/benchmarks/eigenvalue.py",
        "src/mhx/benchmarks/suite.py",
        "src/mhx/benchmarks/orszag_tang.py",
    },
    ("docs/reference/performance.md",): {
        ".github/workflows/ci.yml",
        "src/mhx/benchmarks/timing.py",
        "tests/test_timing_benchmark.py",
    },
    ("docs/project/paper_plan.md",): {
        "src/mhx/benchmarks/campaigns.py",
        "src/mhx/benchmarks/campaign_runner.py",
        "src/mhx/neural_ode/reproducibility.py",
        "src/mhx/benchmarks/seed_robust_qi.py",
        "src/mhx/cli/main.py",
        "src/mhx/physics/equilibria.py",
    },
    ("docs/project/seed_robust_qi.md",): {
        "src/mhx/benchmarks/seed_robust_qi.py",
        "src/mhx/benchmarks/suite.py",
        "src/mhx/cli/main.py",
        "tests/test_seed_robust_qi.py",
    },
    ("docs/project/neural_ode_reproducibility.md",): {
        "src/mhx/neural_ode/reproducibility.py",
        "src/mhx/neural_ode/__init__.py",
        "src/mhx/cli/main.py",
        "examples/make_neural_ode_reproducibility.py",
        "tests/test_neural_ode_reproducibility.py",
    },
    ("docs/project/reviewer_evidence.md",): {
        "src/mhx/benchmarks/seed_robust_qi.py",
        "src/mhx/benchmarks/campaign_runner.py",
        "src/mhx/benchmarks/duration_policy.py",
        "src/mhx/benchmarks/readiness.py",
        "tests/test_campaign_runner.py",
    },
    ("docs/project/campaign_runner.md",): {
        "src/mhx/benchmarks/campaigns.py",
        "src/mhx/benchmarks/campaign_runner.py",
        "src/mhx/benchmarks/duration_policy.py",
        "src/mhx/campaigns/production.py",
        "src/mhx/campaigns/__init__.py",
        "tests/test_campaign_runner.py",
        "tests/test_production_campaign.py",
    },
    ("docs/project/publication_checklist.md",): {
        "examples/tools/verify_paper_artifacts.py",
        "examples/make_validation_media.py",
        "examples/make_readme_media.py",
        "src/mhx/plotting/reduced_mhd.py",
        "tests/test_readme_media.py",
    },
    ("docs/project/paper_pipeline.md",): {
        "src/mhx/benchmarks/paper_pipeline.py",
        "src/mhx/cli/main.py",
    },
    ("docs/project/media_inventory.md",): {
        "examples/make_readme_media.py",
        "src/mhx/benchmarks/current_sheet.py",
        "src/mhx/benchmarks/orszag_tang.py",
        "src/mhx/campaigns/production.py",
        "tests/test_current_sheet_eigenvalue_validation.py",
        "tests/test_orszag_tang_validation.py",
        "tests/test_production_campaign.py",
    },
    ("docs/project/long_run_evidence.md",): {
        "src/mhx/benchmarks/nonlinear.py",
        "src/mhx/benchmarks/current_sheet.py",
        "src/mhx/physics/equilibria.py",
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
        DOCS / "reference" / "api" / "index.md": "reference/api/",
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
        if name in {"index", "reference/api/index"}:
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


def test_reviewer_claim_levels_stay_explicit() -> None:
    reviewer_text = (DOCS / "project" / "reviewer_evidence.md").read_text(encoding="utf-8")

    for claim_level in (
        "`smoke`",
        "`validation`",
        "`production_template`",
        "`production`",
    ):
        assert claim_level in reviewer_text

    assert "not production UQ" in reviewer_text


def test_validation_split_pages_keep_their_gates() -> None:
    """Every gate section survived the split into topic pages."""
    expectations = {
        "exact_limits.md": ["## Exact resistive decay", "## Diffusion eigenvalue scaffold"],
        "linear_tearing.md": [
            "## FKR growth-rate gate",
            "## Direct Harris-sheet tearing eigenvalue gate",
        ],
        "nonlinear.md": [
            "## Nonlinear Orszag--Tang reduced-MHD gate",
            "## Nonlinear reduced-MHD energy budget",
        ],
        "reconnection_campaigns.md": [
            "## Seeded double-Harris long-run validation",
            "## Seed-robust QI validation",
        ],
        "scaling_theory.md": ["## Reconnection scaling gates"],
    }
    for page, headings in expectations.items():
        text = (DOCS / "validation" / page).read_text(encoding="utf-8")
        for heading in headings:
            assert heading in text, f"validation/{page} lost {heading!r}"
