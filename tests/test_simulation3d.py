"""The same-API contract: a 2D Simulation call becomes 3D by two fields."""

from __future__ import annotations

import numpy as np
import pytest

import mhx


def make_simulation(**overrides):
    settings = dict(
        shape=(8, 8, 8),
        equations="mhd3d",
        equilibrium=mhx.OrszagTang3DEquilibrium(),
        viscosity=5.0e-3,
        resistivity=5.0e-3,
        dt=1.0e-2,
        t_end=0.1,
        save_every=5,
        verbose=False,
    )
    settings.update(overrides)
    return mhx.Simulation(**settings)


def test_mhd3d_run_reports_plots_and_saves(tmp_path) -> None:
    result = make_simulation().run()

    assert result.trajectory.times.shape[0] == 2
    assert result.diagnostics["final_total_energy"] < result.diagnostics[
        "initial_total_energy"
    ]
    scale = float(np.max(np.abs(np.asarray(result.final_state.b_hat))))
    assert result.diagnostics["final_magnetic_divergence_linf"] < 1.0e-10 * scale

    figure = result.plot(tmp_path / "summary.png")
    assert figure.exists()
    data = result.save(tmp_path)
    payload = np.load(data)
    assert payload["dimension"] == 3
    assert payload["velocity"].shape == (2, 3, 8, 8, 8)


def test_mhd3d_default_integrator_upgrades_and_validates() -> None:
    assert make_simulation().integrator == "if_rk3"
    assert make_simulation(integrator="etdrk4").integrator == "etdrk4"
    with pytest.raises(ValueError, match="3D integrator"):
        make_simulation(integrator="backward_euler")


def test_mhd3d_shape_and_equilibrium_validation() -> None:
    with pytest.raises(ValueError, match="three grid sizes"):
        make_simulation(shape=(8, 8))
    with pytest.raises(TypeError, match="initial_fields"):
        mhx.Simulation(
            shape=(8, 8, 8),
            equations="mhd3d",
            dt=1.0e-2,
            t_end=0.1,
            verbose=False,
        ).run()


def test_reduced_mhd_rejects_three_dimensional_shapes() -> None:
    with pytest.raises(ValueError, match="two grid sizes"):
        mhx.Simulation(shape=(8, 8, 8))


def test_mhd3d_substeps_above_cfl_and_stops(capsys) -> None:
    # OT at 8^3 has a CFL limit near 0.02 at cfl=0.5, so dt=0.5 needs substeps.
    result = make_simulation(dt=0.5, t_end=1.0, save_every=1, verbose=True).run()
    assert result.diagnostics["max_substeps"] > 1
    assert np.isfinite(np.asarray(result.final_state.b_hat)).all()
    assert "step 2/2" in capsys.readouterr().out

    with pytest.raises(FloatingPointError, match="more than 100 substeps"):
        make_simulation(dt=0.5, t_end=0.5, cfl=1.0e-4).run()
    with pytest.raises(FloatingPointError, match="non-finite fields at step"):
        make_simulation(
            equilibrium=mhx.OrszagTang3DEquilibrium(beta=float("nan"))
        ).run()
