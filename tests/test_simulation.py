from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import mhx
import mhx.parallel as parallel_module
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


def test_simulation_ensemble_runs_plots_and_saves(tmp_path, capsys, monkeypatch) -> None:
    equilibria = tuple(
        mhx.PeriodicDoubleHarrisEquilibrium(
            perturbation_amplitude=1.0e-3 + case * 1.0e-4
        )
        for case in range(5)
    )
    result = mhx.Simulation(
        shape=(8, 8),
        dt=0.001,
        t_end=0.002,
        save_every=1,
        verbose=True,
    ).run_ensemble(equilibria)

    result.print_summary()
    output = result.save(tmp_path / "ensemble")
    figure = result.plot(tmp_path / "ensemble.png")

    assert result.case_count == 5
    assert result.final_states.psi.shape == (5, 8, 8)
    assert result.trajectory.states.psi.shape == (5, 2, 8, 8)
    assert result.config["parallel_axis"] == "case"
    assert len(tuple(output.glob("case_*.npz"))) == 5
    assert figure.is_file()
    assert "MHX ensemble" in capsys.readouterr().out

    monkeypatch.setattr(jax, "process_index", lambda: 1)
    assert result.save(tmp_path / "not-written") == tmp_path / "not-written"
    assert result.plot(tmp_path / "not-written.png") == tmp_path / "not-written.png"
    assert not (tmp_path / "not-written").exists()
    assert not (tmp_path / "not-written.png").exists()


def test_simulation_ensemble_validates_configuration(monkeypatch) -> None:
    simulation = mhx.Simulation(
        shape=(8, 8),
        dt=0.001,
        t_end=0.002,
        device_count=2,
        verbose=False,
    )
    equilibrium = mhx.ZeroEquilibrium()

    with pytest.raises(ValueError, match="at least one"):
        simulation.run_ensemble(())
    with pytest.raises(ValueError, match="divisible"):
        simulation.run_ensemble((equilibrium,))
    with pytest.raises(ValueError, match="RK4"):
        mhx.Simulation(
            shape=(8, 8),
            dt=0.001,
            t_end=0.002,
            integrator="backward_euler",
            verbose=False,
        ).run_ensemble((equilibrium,))

    monkeypatch.setattr(jax, "process_count", lambda: 2)
    monkeypatch.setattr(jax, "device_count", lambda: 2)
    with pytest.raises(ValueError, match="every global"):
        mhx.Simulation(
            shape=(8, 8),
            dt=0.001,
            t_end=0.002,
            device_count=1,
            verbose=False,
        ).run_ensemble((equilibrium, equilibrium))

    monkeypatch.setattr(jax, "process_count", lambda: 3)
    with pytest.raises(ValueError, match="processes"):
        simulation.run_ensemble((equilibrium,) * 4)


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
    mesh = mhx.make_device_mesh(1)
    plan = mhx.make_spatial_sharding((8, 8), 1)

    assert mesh.shape == {"device": 1}
    assert plan.device_count == 1
    assert len(mhx.available_devices()) >= 1
    with pytest.raises(ValueError, match="at least 1"):
        mhx.make_spatial_sharding((8, 8), 0)
    with pytest.raises(ValueError, match="divisible"):
        mhx.make_spatial_sharding((9, 8), 2)
    unavailable = len(mhx.available_devices()) + 1
    with pytest.raises(ValueError, match="JAX found"):
        mhx.make_spatial_sharding((8 * unavailable, 8), unavailable)


def test_parallel_wrappers_delegate_and_fall_back(monkeypatch) -> None:
    initialization = {}

    def record_initialize(**kwargs) -> None:
        initialization.update(kwargs)

    monkeypatch.setattr(parallel_module.jax.distributed, "initialize", record_initialize)
    mhx.initialize_distributed(
        coordinator_address="localhost:1234",
        num_processes=2,
        process_id=1,
        local_device_ids=(0,),
    )

    mesh = mhx.make_device_mesh(1)
    sentinel = object()
    monkeypatch.setattr(
        parallel_module.solvax,
        "shard_batch",
        lambda *args, **kwargs: sentinel,
        raising=False,
    )
    delegated = parallel_module.shard_batch(
        lambda values: values,
        mesh=mesh,
        input_rank=1,
        output_rank=1,
    )
    monkeypatch.delattr(parallel_module.solvax, "shard_batch")
    fallback = parallel_module.shard_batch(
        lambda values: values + 1,
        mesh=mesh,
        input_rank=1,
        output_rank=1,
    )

    assert initialization == {
        "coordinator_address": "localhost:1234",
        "num_processes": 2,
        "process_id": 1,
        "local_device_ids": (0,),
    }
    assert delegated is sentinel
    assert np.array_equal(jax.jit(fallback)(jnp.arange(3)), np.arange(1, 4))


def test_two_device_cpu_simulation_in_fresh_process() -> None:
    program = """
import jax
import mhx
import numpy as np

result = mhx.Simulation(
    shape=(8, 8),
    dt=0.001,
    t_end=0.002,
    save_every=1,
    device_count=2,
    verbose=False,
).run()
reference = mhx.Simulation(
    shape=(8, 8),
    dt=0.001,
    t_end=0.002,
    save_every=1,
    device_count=1,
    verbose=False,
).run()
assert result.device_count == 2
assert len(result.trajectory.states.psi.addressable_shards) == 2
assert result.trajectory.states.psi.shape == (2, 8, 8)
assert np.allclose(result.trajectory.states.psi, reference.trajectory.states.psi)
assert np.allclose(result.trajectory.states.omega, reference.trajectory.states.omega)
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


def test_two_device_cpu_ensemble_in_fresh_process() -> None:
    program = """
import mhx

equilibria = tuple(mhx.ZeroEquilibrium() for _ in range(4))
result = mhx.Simulation(
    shape=(9, 8),
    dt=0.001,
    t_end=0.002,
    save_every=1,
    device_count=2,
    verbose=False,
).run_ensemble(equilibria)
assert result.device_count == 2
assert len(result.trajectory.states.psi.addressable_shards) == 2
assert result.trajectory.states.psi.shape == (4, 2, 9, 8)
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


def test_checked_strong_scaling_measurements_are_finite_and_faster() -> None:
    data_dir = ROOT / "docs" / "_static" / "performance"

    for name in (
        "cpu_ensemble_strong_scaling.json",
        "gpu_ensemble_strong_scaling.json",
    ):
        data = json.loads((data_dir / name).read_text(encoding="utf-8"))
        times = np.asarray(data["run_seconds"])
        samples = np.asarray(data["run_samples_seconds"])
        speedup = np.asarray(data["speedup"])

        assert data["finite"] is True
        assert data["parallel_axis"] == "independent_case"
        assert np.isfinite(times).all()
        assert samples.shape == (len(times), data["samples_per_count"])
        assert np.allclose(times, np.median(samples, axis=1))
        assert (times > 0.0).all()
        assert np.allclose(speedup, times[0] / times)
        assert speedup[-1] > 1.5

    assert (ROOT / "docs" / "_static" / "readme" / "strong_scaling.png").is_file()
