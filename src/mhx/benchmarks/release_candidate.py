"""Static release-candidate gate for public repository readiness."""

from __future__ import annotations

import ast
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib

from mhx._version import __version__
from mhx.io import write_manifest

RELEASE_CANDIDATE_SCHEMA = "mhx.release_candidate.v1"
RELEASE_CANDIDATE_GATES_SCHEMA = "mhx.release_candidate.gates.v1"
DOCS_FIGURE_SCHEMA = "mhx.docs.figures_manifest.v1"

REQUIRED_ROOT_FILES = (
    "README.md",
    "pyproject.toml",
    "LICENSE",
    "CITATION.cff",
    "CHANGELOG.md",
    "RELEASE.md",
    ".github/workflows/ci.yml",
    ".github/workflows/docs.yml",
    ".github/workflows/benchmark-smoke.yml",
    ".github/workflows/publish.yml",
)

REQUIRED_DOCS = (
    "docs/conf.py",
    "docs/index.md",
    "docs/getting_started/install.md",
    "docs/getting_started/first_run.md",
    "docs/validation/index.md",
    "docs/project/media_inventory.md",
    "docs/project/reviewer_evidence.md",
    "docs/project/publication_checklist.md",
    "docs/project/paper_pipeline.md",
    "docs/develop/release.md",
    "docs/develop/release.md",
)

REQUIRED_EXAMPLES = (
    "examples/publication_linear_harris_tearing.py",
    "examples/publication_double_harris_reconnection.py",
    "examples/publication_orszag_tang_turbulence.py",
    "examples/publication_neural_ode.py",
    "examples/publication_rutherford_production.py",
    "examples/plugin_template/README.md",
)

README_REQUIRED_MARKERS = (
    "[![CI]",
    "[![Coverage]",
    "[![Documentation]",
    "First run",
    "Example gallery",
    "Physics and numerics",
    "Parallel runs",
    "Development",
    "does not solve the full three-dimensional MHD",
    "bounded validation runs",
)

RELEASE_CANDIDATE_PRODUCTION_GAPS = (
    "publication Rutherford-scaling and plasmoid-chain claims remain blocked "
    "until algebraic-growth, Lundquist/aspect-ratio, and secondary-island "
    "convergence evidence is attached",
    "README and docs media are validation-level communication artifacts, not "
    "production reconnection-rate or plasmoid-chain measurements",
)


@dataclass(frozen=True)
class ReleaseCandidateAssessment:
    """Machine-readable static release-candidate assessment."""

    release_candidate_ready: bool
    publication_claim_ready: bool
    diagnostics: dict[str, Any]
    validation: dict[str, Any]


def run_release_candidate_assessment(
    repo_root: str | Path = ".",
    *,
    readiness: str | Path | dict[str, Any] | None = None,
    require_readiness: bool = False,
) -> ReleaseCandidateAssessment:
    """Assess static repository gates for public release handoff.

    This gate is intentionally static and fast. It checks that the repository has
    packaging, CI, ReadTheDocs, citation, release, examples, media provenance,
    and current figure-hash metadata. Dynamic physics gates remain owned by
    ``mhx validate all``, ``mhx validate readiness``, and promoted production
    campaign reports.
    """
    root = Path(repo_root).resolve()
    readiness_payload = _load_readiness(readiness)
    file_checks = _required_paths_present(root, REQUIRED_ROOT_FILES)
    docs_checks = _required_paths_present(root, REQUIRED_DOCS)
    example_checks = _required_paths_present(root, REQUIRED_EXAMPLES)
    readme_checks = _readme_checks(root)
    metadata_checks = _metadata_checks(root)
    figure_checks = _figure_manifest_checks(root)
    legacy_checks = _legacy_import_checks(root)
    workflow_checks = _workflow_checks(root)
    readiness_checks = _readiness_checks(readiness_payload, require_readiness=require_readiness)

    check_groups = {
        "required_root_files": file_checks,
        "required_docs": docs_checks,
        "required_examples": example_checks,
        "readme": readme_checks,
        "metadata": metadata_checks,
        "figure_manifest": figure_checks,
        "legacy_imports": legacy_checks,
        "workflows": workflow_checks,
        "readiness": readiness_checks,
    }
    checks = {
        group_name: all(group.values()) for group_name, group in check_groups.items()
    }
    release_candidate_ready = all(checks.values())
    publication_claim_ready = False
    diagnostics = {
        "schema": RELEASE_CANDIDATE_SCHEMA,
        "package_version": __version__,
        "repo_root": str(root),
        "release_candidate_ready": release_candidate_ready,
        "publication_claim_ready": publication_claim_ready,
        "check_groups": check_groups,
        "checks": checks,
        "readiness_source": _readiness_source(readiness),
        "production_publication_gaps": list(RELEASE_CANDIDATE_PRODUCTION_GAPS),
        "interpretation": (
            "Static public-release gates are separate from nonlinear publication "
            "claims. A passing release-candidate gate means packaging, docs, CI, "
            "metadata, examples, and media provenance are coherent; production "
            "physics claims still require promoted long-campaign artifacts."
        ),
    }
    validation = {
        "schema": RELEASE_CANDIDATE_GATES_SCHEMA,
        "passed": release_candidate_ready,
        "checks": checks,
        "diagnostics": diagnostics,
    }
    return ReleaseCandidateAssessment(
        release_candidate_ready=release_candidate_ready,
        publication_claim_ready=publication_claim_ready,
        diagnostics=diagnostics,
        validation=validation,
    )


def write_release_candidate_report(
    outdir: str | Path,
    *,
    repo_root: str | Path = ".",
    readiness: str | Path | dict[str, Any] | None = None,
    require_readiness: bool = False,
) -> tuple[Path, dict[str, Any]]:
    """Write release-candidate diagnostics, validation, Markdown, and manifest."""
    output_dir = Path(outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    assessment = run_release_candidate_assessment(
        repo_root,
        readiness=readiness,
        require_readiness=require_readiness,
    )
    diagnostics_path = output_dir / "release_candidate.json"
    validation_path = output_dir / "validation.json"
    markdown_path = output_dir / "release_candidate.md"
    manifest_path = output_dir / "manifest.json"
    diagnostics_path.write_text(
        json.dumps(assessment.diagnostics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    validation_path.write_text(
        json.dumps(assessment.validation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(_release_candidate_markdown(assessment.diagnostics), encoding="utf-8")
    write_manifest(
        manifest_path,
        config=assessment.diagnostics,
        outputs={
            "release_candidate": diagnostics_path.name,
            "validation": validation_path.name,
            "release_candidate_markdown": markdown_path.name,
        },
        claim_level="validation",
        claim_scope=(
            "Static release-candidate readiness report. Nonlinear publication "
            "claims remain blocked by production-campaign gates."
        ),
    )
    return diagnostics_path, assessment.validation


def _required_paths_present(root: Path, paths: tuple[str, ...]) -> dict[str, bool]:
    return {path: (root / path).exists() for path in paths}


def _readme_checks(root: Path) -> dict[str, bool]:
    readme_path = root / "README.md"
    text = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""
    return {f"contains:{marker}": marker in text for marker in README_REQUIRED_MARKERS}


def _metadata_checks(root: Path) -> dict[str, bool]:
    pyproject = _load_toml(root / "pyproject.toml")
    citation_text = _read_text(root / "CITATION.cff")
    citation_version_match = re.search(r'^version:\s*"?([^"\n]+)"?', citation_text, re.M)
    pyproject_name = pyproject.get("project", {}).get("name")
    scripts = pyproject.get("project", {}).get("scripts", {})
    return {
        "pyproject_name_is_mhx": pyproject_name == "mhx",
        "pyproject_exposes_mhx_cli": scripts.get("mhx") == "mhx.cli.main:app",
        "citation_version_matches_package": bool(citation_version_match)
        and citation_version_match.group(1) == __version__,
        "readthedocs_config_present": (root / ".readthedocs.yaml").exists()
        or (root / "readthedocs.yaml").exists(),
        "root_readthedocs_yaml_present": (root / "readthedocs.yaml").exists(),
    }


def _figure_manifest_checks(root: Path) -> dict[str, bool]:
    manifest_path = root / "docs" / "figures" / "manifest.toml"
    manifest = _load_toml(manifest_path)
    figures = manifest.get("figures", [])
    checks = {
        "schema": manifest.get("schema") == DOCS_FIGURE_SCHEMA,
        "has_figures": isinstance(figures, list) and bool(figures),
        "hashes_current": True,
        "sources_present": True,
        "tests_present": True,
    }
    if not isinstance(figures, list):
        return checks | {"hashes_current": False, "sources_present": False, "tests_present": False}
    for figure in figures:
        if not isinstance(figure, dict):
            checks["hashes_current"] = False
            continue
        figure_path = root / str(figure.get("path", ""))
        if not figure_path.is_file() or figure.get("sha256") != _sha256_file(figure_path):
            checks["hashes_current"] = False
        for key, check_name in (("sources", "sources_present"), ("tests", "tests_present")):
            entries = figure.get(key, [])
            if not isinstance(entries, list) or not entries:
                checks[check_name] = False
                continue
            for entry in entries:
                if not (root / str(entry)).exists():
                    checks[check_name] = False
    return checks


def _legacy_import_checks(root: Path) -> dict[str, bool]:
    offenders: list[str] = []
    for directory in ("src", "tests", "examples", "tools"):
        for path in (root / directory).rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            imports_legacy = any(_node_imports_legacy(node) for node in ast.walk(tree))
            if imports_legacy:
                offenders.append(path.relative_to(root).as_posix())
    return {"no_active_legacy_imports": not offenders}


def _node_imports_legacy(node: ast.AST) -> bool:
    if isinstance(node, ast.Import):
        return any(
            alias.name == "legacy" or alias.name.startswith("legacy.")
            for alias in node.names
        )
    if isinstance(node, ast.ImportFrom):
        module = node.module or ""
        return module == "legacy" or module.startswith("legacy.")
    return False


def _workflow_checks(root: Path) -> dict[str, bool]:
    workflow_text = "\n".join(
        _read_text(root / workflow)
        for workflow in (
            ".github/workflows/ci.yml",
            ".github/workflows/docs.yml",
            ".github/workflows/benchmark-smoke.yml",
            ".github/workflows/publish.yml",
        )
    )
    return {
        "ci_uses_node24_compatible_checkout": "actions/checkout@v6.0.2" in workflow_text,
        "ci_uses_node24_compatible_setup_python": "actions/setup-python@v6.2.0"
        in workflow_text,
        "artifact_upload_uses_v7": "actions/upload-artifact@v7.0.1" in workflow_text,
    }


def _readiness_checks(
    readiness_payload: dict[str, Any] | None,
    *,
    require_readiness: bool,
) -> dict[str, bool]:
    if readiness_payload is None:
        return {
            "readiness_required_or_not_requested": not require_readiness,
            "public_release_ready_when_supplied": not require_readiness,
            "publication_claim_blocked": True,
        }
    return {
        "readiness_required_or_not_requested": True,
        "public_release_ready_when_supplied": bool(readiness_payload.get("public_release_ready")),
        "publication_claim_blocked": readiness_payload.get("publication_claim_ready") is False,
    }


def _load_readiness(readiness: str | Path | dict[str, Any] | None) -> dict[str, Any] | None:
    if readiness is None:
        return None
    if isinstance(readiness, dict):
        return readiness
    path = Path(readiness)
    if path.is_dir():
        path = path / "readiness.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _readiness_source(readiness: str | Path | dict[str, Any] | None) -> str | None:
    if readiness is None:
        return None
    if isinstance(readiness, dict):
        return "<in-memory>"
    return str(readiness)


def _release_candidate_markdown(diagnostics: dict[str, Any]) -> str:
    status = "yes" if diagnostics["release_candidate_ready"] else "no"
    paper_status = "yes" if diagnostics["publication_claim_ready"] else "no"
    group_rows = "\n".join(
        f"| `{name}` | `{value}` |" for name, value in diagnostics["checks"].items()
    )
    gaps = "\n".join(f"- {gap}" for gap in diagnostics["production_publication_gaps"])
    return (
        "# MHX release-candidate report\n\n"
        f"- Static release-candidate ready: **{status}**\n"
        f"- Publication nonlinear-claim ready: **{paper_status}**\n"
        f"- Package version: `{diagnostics['package_version']}`\n\n"
        "## Gate groups\n\n"
        "| Group | Passed |\n"
        "| --- | --- |\n"
        f"{group_rows}\n\n"
        "## Production publication gaps\n\n"
        f"{gaps}\n"
    )


def _load_toml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
