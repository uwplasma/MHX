from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import mhx
from mhx.io import read_reduced_mhd_trajectory_npz

ROOT = Path(__file__).parents[1]


def test_simulation_runs_reports_plots_and_saves(tmp_path, capsys) -> None:
    simulation = mhx.Simulation(
        shape=(12, 12),
        dt=0.002,
        t_end=0.006,
        save_every=2,
        verbose=True,
    )

    result = simulation.run()
    result.print_summary()
    figure_path = result.plot(tmp_path / "summary.png")
    data_path = result.save(tmp_path / "run")
    direct_data_path = result.save(tmp_path / "direct.npz")
    loaded, diagnostics = read_reduced_mhd_trajectory_npz(data_path)

    assert result.final_time == pytest.approx(0.006)
    assert result.final_state.psi.shape == (12, 12)
    assert result.device_count == 1
    assert result.config["integrator"] == "rk4"
    assert figure_path.is_file()
    assert data_path.is_file()
    assert direct_data_path.is_file()
    assert loaded.states.psi.shape == result.trajectory.states.psi.shape
    assert diagnostics["final_total_energy"] == pytest.approx(
        result.diagnostics["final_total_energy"]
    )
    assert "MHX result" in capsys.readouterr().out


def test_simulation_uses_solvax_for_backward_euler() -> None:
    result = mhx.Simulation(
        shape=(8, 8),
        equilibrium=mhx.ZeroEquilibrium(),
        dt=0.01,
        t_end=0.02,
        save_every=1,
        integrator="backward_euler",
        verbose=False,
    ).run()

    assert result.diagnostics["implicit_converged"] is True
    assert result.diagnostics["implicit_linear_converged"] is True


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"shape": (3, 8)}, "shape"),
        ({"dt": 0.0}, "dt"),
        ({"t_end": 0.0}, "t_end"),
        ({"save_every": 0}, "save_every"),
        ({"t_end": 0.03, "dt": 0.02}, "integer multiple"),
        ({"integrator": "unknown"}, "integrator"),
        ({"dealiasing": "unknown"}, "dealiasing"),
    ),
)
def test_simulation_rejects_invalid_settings(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        mhx.Simulation(**kwargs)


def test_spatial_sharding_validates_shape_and_device_count() -> None:
    plan = mhx.make_spatial_sharding((8, 8), 1)

    assert plan.device_count == 1
    assert len(mhx.available_devices()) >= 1
    with pytest.raises(ValueError, match="at least 1"):
        mhx.make_spatial_sharding((8, 8), 0)
    with pytest.raises(ValueError, match="divisible"):
        mhx.make_spatial_sharding((9, 8), 2)
    unavailable = len(mhx.available_devices()) + 1
    with pytest.raises(ValueError, match="JAX found"):
        mhx.make_spatial_sharding((8 * unavailable, 8), unavailable)


def test_two_device_cpu_simulation_in_fresh_process() -> None:
    program = """
import jax
import mhx

result = mhx.Simulation(
    shape=(8, 8),
    dt=0.001,
    t_end=0.002,
    save_every=1,
    device_count=2,
    verbose=False,
).run()
assert result.device_count == 2
assert len(result.trajectory.states.psi.addressable_shards) == 2
assert result.trajectory.states.psi.shape == (2, 8, 8)
print(jax.device_count())
"""
    environment = os.environ.copy()
    environment["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
    completed = subprocess.run(
        [sys.executable, "-c", program],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "2"


def test_simulation_result_arrays_are_finite() -> None:
    result = mhx.Simulation(
        shape=(8, 8),
        dt=0.001,
        t_end=0.002,
        save_every=1,
        device_count=1,
        verbose=False,
    ).run()

    assert np.isfinite(np.asarray(result.trajectory.states.psi)).all()
    assert np.isfinite(np.asarray(result.trajectory.states.omega)).all()


def test_checked_sharding_measurements_are_finite_and_consistent() -> None:
    data_dir = ROOT / "docs" / "_static" / "performance"

    for name in ("cpu_spatial_sharding.json", "gpu_spatial_sharding.json"):
        data = json.loads((data_dir / name).read_text(encoding="utf-8"))
        times = np.asarray(data["run_seconds"])
        speedup = np.asarray(data["speedup"])

        assert data["finite"] is True
        assert np.isfinite(times).all()
        assert (times > 0.0).all()
        assert np.allclose(speedup, times[0] / times)

    assert (ROOT / "docs" / "_static" / "readme" / "strong_scaling.png").is_file()
