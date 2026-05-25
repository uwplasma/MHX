from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "tools" / "run_nonlinear_production_campaign.py"
)


def _load_script_module():
    spec = importlib.util.spec_from_file_location("run_nonlinear_production_campaign", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _options(module, tmp_path):
    return module.CampaignOptions(
        outdir=tmp_path / "lane",
        python_executable="python",
        production_efolds=1.0,
        safety_factor=2.0,
        nx=64,
        ny=48,
        dt=5.0e-2,
        target_saved_frames=40,
        width=0.36,
        widths=(0.30, 0.36, 0.42),
        eta=4.5e-3,
        etas=(6.0e-3, 4.5e-3, 3.0e-3),
        seeds=(2, 3, 4),
        convergence_resolutions=(32, 48, 64),
        crosscheck_resolutions=(40, 56, 64),
        dt_values=(4.0e-2, 1.0e-2),
        crosscheck_dt_values=(3.0e-2, 1.5e-2),
        min_production_resolution=32,
        timeout_seconds=123,
        gate_timeout_seconds=45,
    )


def test_production_campaign_manifest_schema_and_gates(tmp_path) -> None:
    module = _load_script_module()
    options = _options(module, tmp_path)

    manifest = module.build_manifest(options)

    assert manifest["schema"] == module.SCHEMA
    assert manifest["mode"] == "dry_run"
    assert manifest["claim_level"] == "validation"
    assert manifest["validation"]["schema"] == module.GATES_SCHEMA
    assert manifest["validation"]["passed"] is True
    assert manifest["campaign"]["duration"]["t_end"] == pytest.approx(
        options.production_efolds * options.safety_factor / options.harris_growth_rate
    )

    gates = {gate["gate"]: gate for gate in manifest["required_gates"]}
    assert set(gates) == {
        "duration",
        "convergence",
        "seed_qi",
        "width_aspect",
        "eta_lundquist",
        "movie",
        "promotion",
    }
    assert gates["promotion"]["command_ids"][-1] == "rutherford_finalize_production_claim"


def test_production_campaign_default_fit_stop_uses_growth_timescale(tmp_path) -> None:
    module = _load_script_module()
    options = module.CampaignOptions(
        outdir=tmp_path / "lane",
        python_executable="python",
        harris_growth_rate=0.125,
        nx=128,
        ny=128,
        target_saved_frames=121,
        min_production_resolution=128,
    )

    assert options.t_end == pytest.approx(240.0)
    assert options.resolved_save_every == 100
    assert options.resolved_save_interval == pytest.approx(2.0)
    assert options.resolved_fit_stop == pytest.approx(16.0)

    manifest = module.build_manifest(options)
    assert manifest["campaign"]["time"]["fit_stop"] == pytest.approx(16.0)


def test_production_campaign_explicit_fit_stop_still_overrides(tmp_path) -> None:
    module = _load_script_module()
    options = module.CampaignOptions(
        outdir=tmp_path / "lane",
        python_executable="python",
        harris_growth_rate=0.125,
        nx=128,
        ny=128,
        fit_stop=24.0,
        min_production_resolution=128,
    )

    assert options.resolved_fit_stop == pytest.approx(24.0)


def test_production_campaign_command_generation_is_exact(tmp_path) -> None:
    module = _load_script_module()
    manifest = module.build_manifest(_options(module, tmp_path))
    commands = {command["id"]: command for command in manifest["commands"]}

    long_run = commands["double_harris_long_run"]
    assert long_run["timeout_seconds"] == 123
    assert long_run["expensive"] is True
    assert "--movies" in long_run["command"]
    assert float(long_run["command"][long_run["command"].index("--t-end") + 1]) == pytest.approx(
        manifest["campaign"]["duration"]["t_end"]
    )

    rutherford_execute = commands["rutherford_execute"]["command"]
    assert "--movies" in rutherford_execute
    assert "--allow-production-claim" not in rutherford_execute

    rutherford_promotion = commands["rutherford_promotion_check"]["command"]
    assert rutherford_promotion.count("--convergence-dir") == 2
    assert "--seed-qi-dir" in rutherford_promotion
    assert "--min-convergence-dirs" in rutherford_promotion

    final_command = commands["rutherford_finalize_production_claim"]["command"]
    assert "--allow-production-claim" in final_command
    assert final_command[final_command.index("--max-steps") + 1] == "0"


def test_production_campaign_dry_run_writes_manifest_and_shell(tmp_path, capsys) -> None:
    module = _load_script_module()
    exit_code = module.main(
        [
            "--dry-run",
            "--outdir",
            str(tmp_path / "lane"),
            "--python-executable",
            "python",
            "--production-efolds",
            "1",
            "--safety-factor",
            "1",
            "--nx",
            "32",
            "--ny",
            "32",
            "--target-saved-frames",
            "20",
            "--convergence-resolutions",
            "16,24,32",
            "--crosscheck-resolutions",
            "20,28,32",
            "--min-production-resolution",
            "16",
            "--timeout-seconds",
            "99",
            "--gate-timeout-seconds",
            "11",
        ]
    )

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "dry run only" in captured.out
    manifest_path = tmp_path / "lane" / "production_campaign_manifest.json"
    commands_path = tmp_path / "lane" / "run_commands.sh"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["execution"]["status"] == "skipped_dry_run"
    assert manifest["commands"][0]["timeout_seconds"] == 11
    assert any(command["timeout_seconds"] == 99 for command in manifest["commands"])
    assert "double-harris-long-run" in commands_path.read_text()
    assert commands_path.stat().st_mode & 0o111


def test_production_campaign_execute_uses_manifest_timeouts(tmp_path, monkeypatch) -> None:
    module = _load_script_module()
    manifest = module.build_manifest(_options(module, tmp_path), mode="execute")
    calls = []

    def fake_run(command, *, check, timeout):
        calls.append({"command": command, "check": check, "timeout": timeout})
        return module.subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    exit_code = module.execute_manifest(manifest)

    assert exit_code == 0
    assert manifest["execution"]["status"] == "passed"
    assert calls[0]["timeout"] == 45
    assert calls[1]["timeout"] == 123
    assert all(call["check"] is False for call in calls)
