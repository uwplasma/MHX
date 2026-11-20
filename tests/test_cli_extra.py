"""Edge-case and error-path tests for the ``mhx`` Typer command line interface.

The benchmark commands are expensive, so their failed-validation exit codes are
exercised by substituting the validation writer with a recording stub that
reports ``passed=False``; the CLI plumbing (option forwarding, exit status and
failure reporting) is what is under test there.
"""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

from mhx.cli import main as cli_main
from mhx.cli.main import app
from mhx.versioning import api_version_info

runner = CliRunner()


# ---------------------------------------------------------------------------
# Failed-validation exit codes for every benchmark command
# ---------------------------------------------------------------------------

FAILING_COMMANDS = [
    (["decay", "--nx", "8", "--ny", "8"], "write_resistive_decay_validation"),
    (["scaling", "--ka", "0.4"], "write_reconnection_scaling_validation"),
    (["fkr-window", "--lundquist", "1e5"], "write_fkr_window_validation"),
    (["fkr-growth", "--fixed-ka", "0.3"], "write_fkr_growth_rate_validation"),
    (["harris-delta-prime", "--steps", "10"], "write_harris_delta_prime_validation"),
    (
        ["linear-tearing-eigenvalue", "--grid-points", "16,32"],
        "write_linear_tearing_eigenvalue_validation",
    ),
    (
        ["linear-tearing-dispersion", "--wavenumber", "0.3,0.5"],
        "write_linear_tearing_dispersion_validation",
    ),
    (["linear-tearing-layer", "--lundquist", "250,500"], "write_linear_tearing_layer_validation"),
    (
        ["linear-tearing-timedomain", "--t-end", "1.0"],
        "write_linear_tearing_timedomain_validation",
    ),
    (["linearized-rhs", "--epsilon", "1e-4"], "write_linearized_rhs_validation"),
    (["reduced-mhd-eigenmode"], "write_reduced_mhd_linear_eigenmode_validation"),
    (
        ["cosine-equilibrium-linearization"],
        "write_cosine_equilibrium_linearization_validation",
    ),
    (["current-sheet-eigenvalue"], "write_periodic_current_sheet_eigenvalue_validation"),
    (["current-sheet-timedomain"], "write_periodic_current_sheet_timedomain_validation"),
    (
        ["current-sheet-nonlinear-bridge"],
        "write_periodic_current_sheet_nonlinear_bridge_validation",
    ),
    (["double-harris-growth"], "write_periodic_double_harris_nonlinear_growth_validation"),
    (["orszag-tang", "--movies"], "write_orszag_tang_vortex_validation"),
    (["decaying-turbulence", "--seed", "3"], "write_decaying_mhd_turbulence_validation"),
    (
        ["forced-turbulent-reconnection", "--seed", "5"],
        "write_forced_turbulent_reconnection_validation",
    ),
    (["double-harris-long-run"], "write_periodic_double_harris_seeded_long_run_validation"),
    (["double-harris-convergence"], "write_periodic_double_harris_convergence_validation"),
    (["double-harris-parameter-sweep"], "write_periodic_double_harris_parameter_sweep_validation"),
    (["nonlinear-energy-budget"], "write_nonlinear_energy_budget_validation"),
    (["diffusion-eigenvalue"], "write_diffusion_eigenvalue_validation"),
    (["power-iteration", "--iterations", "3"], "write_power_iteration_validation"),
    (["arnoldi"], "write_arnoldi_validation"),
]


@pytest.mark.parametrize("passed", [False, True], ids=["fails", "passes"])
@pytest.mark.parametrize(("args", "writer"), FAILING_COMMANDS, ids=[c[1] for c in FAILING_COMMANDS])
def test_benchmark_command_exit_status_follows_validation(
    args, writer, passed, tmp_path, monkeypatch
) -> None:
    calls: list[dict[str, object]] = []

    def fake_writer(outdir, **kwargs):
        calls.append({"outdir": Path(outdir), **kwargs})
        return Path(outdir) / "manifest.json", {"passed": passed}

    monkeypatch.setattr(cli_main, writer, fake_writer)
    result = runner.invoke(app, ["benchmark", *args, "--outdir", str(tmp_path / "out")])
    assert result.exit_code == (0 if passed else 1), result.output
    assert f"wrote {tmp_path / 'out' / 'manifest.json'}" in result.output
    assert len(calls) == 1
    assert calls[0]["outdir"] == tmp_path / "out"


def test_double_harris_long_run_ci_fast_preset_overrides_duration(tmp_path, monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_writer(outdir, **kwargs):
        calls.append(kwargs)
        return Path(outdir) / "manifest.json", {"passed": True}

    monkeypatch.setattr(
        cli_main, "write_periodic_double_harris_seeded_long_run_validation", fake_writer
    )
    preset = cli_main.double_harris_seeded_long_run_presets()["ci_fast"]

    fast = runner.invoke(
        app, ["benchmark", "double-harris-long-run", "--outdir", str(tmp_path), "--ci-fast"]
    )
    full = runner.invoke(
        app,
        ["benchmark", "double-harris-long-run", "--outdir", str(tmp_path), "--t-end", "7.5"],
    )
    assert fast.exit_code == 0 and full.exit_code == 0
    assert calls[0]["t_end"] == float(preset["t_end"])
    assert calls[0]["duration_label"] == str(preset["duration_label"])
    assert calls[0]["fit_window"] == tuple(preset["fit_window"])
    assert calls[1]["t_end"] == 7.5
    assert calls[1]["duration_label"] is None


# ---------------------------------------------------------------------------
# Readiness / promotion checks: failure reporting
# ---------------------------------------------------------------------------


def _stub_check_writer(validation):
    def fake_writer(run_dir, **kwargs):
        return Path(run_dir) / "report.json", validation

    return fake_writer


@pytest.mark.parametrize(
    ("command", "writer"),
    [
        (
            "forced-turbulent-reconnection-readiness-check",
            "write_forced_turbulent_reconnection_readiness_report",
        ),
        ("double-harris-promotion-check", "write_periodic_double_harris_promotion_report"),
    ],
)
def test_check_commands_report_failed_checks_and_diagnostics(
    command, writer, tmp_path, monkeypatch
) -> None:
    validation = {
        "passed": False,
        "checks": {"b_check": False, "a_check": False, "ok_check": True},
        "diagnostics": {"steps_run": 12, "dt": 0.5, "unrelated": 1},
    }
    monkeypatch.setattr(cli_main, writer, _stub_check_writer(validation))
    result = runner.invoke(app, ["benchmark", command, str(tmp_path)])
    assert result.exit_code == 1
    assert f"{command} failed validation checks: a_check, b_check" in result.output
    assert "steps_run: 12" in result.output
    assert "dt: 0.5" in result.output
    assert "unrelated" not in result.output


def test_exit_helper_is_silent_on_success() -> None:
    cli_main._exit_if_validation_failed({"passed": True}, context="x")


def test_exit_helper_tolerates_non_mapping_checks_and_diagnostics() -> None:
    with pytest.raises(typer.Exit) as info:
        cli_main._exit_if_validation_failed(
            {"passed": False, "checks": ["a"], "diagnostics": "nope"}, context="x"
        )
    assert info.value.exit_code == 1


def test_exit_helper_with_all_checks_true_prints_no_failure_line(capsys) -> None:
    with pytest.raises(typer.Exit):
        cli_main._exit_if_validation_failed(
            {"passed": False, "checks": {"a": True}, "diagnostics": {"shape": [8, 8]}},
            context="ctx",
        )
    captured = capsys.readouterr()
    assert "failed validation checks" not in captured.err
    assert "shape: [8, 8]" in captured.err


def test_promotion_check_forwards_convergence_dirs(tmp_path, monkeypatch) -> None:
    seen: dict[str, object] = {}

    def fake_writer(run_dir, **kwargs):
        seen.update(kwargs)
        return Path(run_dir) / "report.json", {"passed": True}

    monkeypatch.setattr(cli_main, "write_periodic_double_harris_promotion_report", fake_writer)
    result = runner.invoke(
        app,
        [
            "benchmark",
            "double-harris-promotion-check",
            str(tmp_path),
            "--convergence-dir",
            str(tmp_path / "a"),
            "--convergence-dir",
            str(tmp_path / "b"),
        ],
    )
    assert result.exit_code == 0
    assert seen["convergence_dirs"] == (tmp_path / "a", tmp_path / "b")


# ---------------------------------------------------------------------------
# Option parsing helpers
# ---------------------------------------------------------------------------


def test_parse_helpers_accept_valid_input() -> None:
    assert cli_main._parse_int_tuple("1, 2,,3") == (1, 2, 3)
    assert cli_main._parse_float_tuple("0.5, 1e-3") == (0.5, 1.0e-3)
    assert cli_main._parse_string_tuple(" a, b ,") == ("a", "b")
    assert cli_main._parse_mode_tuple("1:1, 2:3,") == ((1, 1), (2, 3))


@pytest.mark.parametrize(
    ("parser", "value", "message"),
    [
        (cli_main._parse_int_tuple, "1,x", "comma-separated integers"),
        (cli_main._parse_float_tuple, "0.5,y", "comma-separated floats"),
        (cli_main._parse_string_tuple, " , ", "at least one comma-separated value"),
        (cli_main._parse_mode_tuple, "1-1", "comma-separated modes"),
        (cli_main._parse_mode_tuple, "a:b", "comma-separated modes"),
        (cli_main._parse_mode_tuple, ",", "at least one mode"),
    ],
)
def test_parse_helpers_reject_bad_input(parser, value, message) -> None:
    with pytest.raises(typer.BadParameter, match=message):
        parser(value)


def test_cli_rejects_malformed_comma_separated_option(tmp_path) -> None:
    result = runner.invoke(
        app,
        [
            "benchmark",
            "double-harris-parameter-sweep",
            "--outdir",
            str(tmp_path),
            "--modes",
            "not-a-mode",
        ],
    )
    assert result.exit_code != 0
    assert "comma-separated modes" in result.output


def test_parameter_sweep_forwards_optional_viscosities(tmp_path, monkeypatch) -> None:
    seen: dict[str, object] = {}

    def fake_writer(outdir, **kwargs):
        seen.update(kwargs)
        return Path(outdir) / "manifest.json", {"passed": True}

    monkeypatch.setattr(
        cli_main, "write_periodic_double_harris_parameter_sweep_validation", fake_writer
    )
    result = runner.invoke(
        app,
        [
            "benchmark",
            "double-harris-parameter-sweep",
            "--outdir",
            str(tmp_path),
            "--viscosities",
            "0.001,0.002",
        ],
    )
    assert result.exit_code == 0, result.output
    assert seen["viscosity_values"] == (0.001, 0.002)
    assert seen["modes"] == ((2, 1), (2, 2), (4, 1))


# ---------------------------------------------------------------------------
# api status, run, figures, artifact-manifest, validate
# ---------------------------------------------------------------------------


def test_api_status_text_and_json_outputs_agree() -> None:
    info = api_version_info().to_dict()
    text = runner.invoke(app, ["api", "status"])
    as_json = runner.invoke(app, ["api", "status", "--json"])
    assert text.exit_code == 0 and as_json.exit_code == 0
    assert f"Package version: {info['package_version']}" in text.output
    assert f"Public API: {info['public_api_version']}" in text.output
    assert f"Physics API: {info['physics_api_version']}" in text.output
    assert f"Trajectory schema: {info['trajectory_schema']}" in text.output
    assert f"Claim levels: {', '.join(info['claim_levels'])}" in text.output
    assert json.loads(as_json.output) == info


def test_run_rejects_unsupported_physics_model(tmp_path) -> None:
    cfg = tmp_path / "cfg.toml"
    cfg.write_text(
        'name = "bad"\n\n[mesh]\nshape = [8, 8]\n\n[physics]\nmodel = "not_a_model"\n',
        encoding="utf-8",
    )
    result = runner.invoke(app, ["run", str(cfg), "--outdir", str(tmp_path / "out")])
    assert result.exit_code != 0
    assert isinstance(result.exception, ValueError)
    assert "physics.model must be one of" in str(result.exception)
    assert "reduced_mhd_linear_tearing" in str(result.exception)
    assert not (tmp_path / "out" / "manifest.json").exists()


def _tiny_run(tmp_path: Path) -> Path:
    cfg = tmp_path / "tiny.toml"
    cfg.write_text(
        textwrap.dedent(
            """
            name = "tiny"

            [mesh]
            shape = [8, 8]
            lower = [0.0, 0.0]
            upper = [6.283185307179586, 6.283185307179586]
            periodic = [true, true]

            [time]
            t1 = 0.02
            dt = 0.01
            save_every = 1
            """
        ),
        encoding="utf-8",
    )
    outdir = tmp_path / "run"
    result = runner.invoke(app, ["run", str(cfg), "--outdir", str(outdir)])
    assert result.exit_code == 0, result.output
    return outdir


def test_figures_prints_warning_when_diagnostic_inputs_missing(tmp_path) -> None:
    outdir = _tiny_run(tmp_path)
    (outdir / "config_effective.json").unlink()
    result = runner.invoke(app, ["figures", str(outdir)])
    assert result.exit_code == 0, result.output
    assert "warning: config_effective.json missing" in result.output
    assert (outdir / "figures" / "energy_history.png").exists()


def test_artifact_manifest_lists_hashed_files(tmp_path) -> None:
    root = tmp_path / "bundle"
    root.mkdir()
    (root / "a.txt").write_text("alpha", encoding="utf-8")
    (root / "b.txt").write_text("beta", encoding="utf-8")
    default = runner.invoke(app, ["artifact-manifest", str(root)])
    assert default.exit_code == 0, default.output
    assert "artifact_manifest.json (2 files)" in default.output
    stored = json.loads((root / "artifact_manifest.json").read_text())
    assert {item["path"] for item in stored["files"]} == {"a.txt", "b.txt"}

    custom = tmp_path / "elsewhere.json"
    explicit = runner.invoke(app, ["artifact-manifest", str(root), "--output", str(custom)])
    assert explicit.exit_code == 0, explicit.output
    assert custom.exists()
    assert f"wrote {custom}" in explicit.output


# ---------------------------------------------------------------------------
# Plugin lint commands
# ---------------------------------------------------------------------------

BAD_PLUGIN = """
from dataclasses import dataclass
from typing import ClassVar
from mhx.diagnostics import DiagnosticSpec
from mhx.state import ReducedMHDState

@dataclass(frozen=True)
class OldApiTerm:
    name: ClassVar[str] = "old_api_term"
    api_version: ClassVar[str] = "mhx.physics.v0"
    description: ClassVar[str] = "Term declaring a stale API version."

    def rhs_addition(self, state, params, *, lengths):
        return ReducedMHDState(psi=0.0 * state.psi, omega=0.0 * state.omega)

def register_physics(registry):
    registry.register("old_api_term", lambda parameters: OldApiTerm())

def register_diagnostics(registry):
    registry.register(
        DiagnosticSpec(
            name="no_keys",
            description="Declares no outputs.",
            output_keys=(),
            compute=lambda context: {},
        )
    )
"""


@pytest.fixture
def bad_plugin(tmp_path, monkeypatch):
    (tmp_path / "bad_cli_plugin.py").write_text(BAD_PLUGIN, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    yield "bad_cli_plugin"
    sys.modules.pop("bad_cli_plugin", None)


def test_physics_lint_rejects_stale_api_version(bad_plugin) -> None:
    result = runner.invoke(app, ["physics", "lint", "old_api_term", "--plugin-module", bad_plugin])
    assert result.exit_code != 0
    assert "mhx.physics.v0" in result.output


def test_diagnostics_lint_rejects_missing_output_keys(bad_plugin) -> None:
    result = runner.invoke(app, ["diagnostics", "lint", "no_keys", "--plugin-module", bad_plugin])
    assert result.exit_code != 0
    assert "must declare at least one output key" in result.output


def test_lint_accepts_builtin_diagnostic() -> None:
    result = runner.invoke(app, ["diagnostics", "lint", "energy"])
    assert result.exit_code == 0
    assert "energy: ok" in result.output


def test_version_and_registry_listing_commands() -> None:
    version = runner.invoke(app, ["version"])
    assert version.exit_code == 0
    assert version.output.strip() == cli_main.__version__
    physics = runner.invoke(app, ["physics", "list"])
    assert physics.exit_code == 0
    assert f"Physics API: {cli_main.PHYSICS_API_VERSION}" in physics.output
    equilibria = runner.invoke(app, ["physics", "equilibria"])
    assert equilibria.exit_code == 0
    assert equilibria.output.startswith("- ")
