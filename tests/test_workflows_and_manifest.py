from __future__ import annotations

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib

import mhx
from mhx.benchmarks import REQUIRED_PUBLIC_RELEASE_CASES
from mhx.benchmarks.suite import validation_suite_cases

ROOT = Path(__file__).resolve().parents[1]


def test_top_level_python_api_runs_smoke_config(tmp_path) -> None:
    config_path = ROOT / "examples" / "linear_tearing.toml"

    cfg = mhx.load_config(config_path)
    manifest_path = mhx.run(config_path, outdir=tmp_path / "api_run")

    assert cfg.physics.model == "reduced_mhd_linear_tearing"
    assert manifest_path == tmp_path / "api_run" / "manifest.json"
    assert manifest_path.exists()


def test_release_readiness_requires_every_validation_suite_case() -> None:
    suite_names = {case.name for case in validation_suite_cases()}

    assert set(REQUIRED_PUBLIC_RELEASE_CASES) == suite_names


def test_required_workflows_exist() -> None:
    workflow_dir = ROOT / ".github" / "workflows"
    required = {
        "ci.yml",
        "docs.yml",
        "benchmark-smoke.yml",
        "publish.yml",
    }

    assert {path.name for path in workflow_dir.glob("*.yml")} >= required


def test_docs_figure_manifest_is_parseable_and_complete() -> None:
    manifest_path = ROOT / "docs" / "figures" / "manifest.toml"
    manifest = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
    figure_ids = {entry["id"] for entry in manifest["figures"]}

    assert manifest["schema"] == "mhx.docs.figures_manifest.v1"
    assert {
        "readme_double_harris_reconnection",
        "readme_forced_turbulent_reconnection",
        "readme_decaying_mhd_turbulence",
        "validation_linear_tearing_eigenvalue",
        "validation_latent_ode_fit",
    } <= figure_ids

    for entry in manifest["figures"]:
        path = ROOT / entry["path"]
        assert path.exists(), entry["path"]
        assert entry["claim_level"] in {"smoke", "validation", "production_template"}
        assert entry["command"]
