"""The same-API contract for ``equations="compressible"``."""

from __future__ import annotations

import numpy as np
import pytest

import mhx


def make_simulation(**overrides):
    settings = dict(
        shape=(8, 8, 4),
        equations="compressible",
        equilibrium=mhx.OrszagTang3DEquilibrium(beta=0.8),
        sound_speed=4.0,
        viscosity=5.0e-3,
        bulk_viscosity=5.0e-3,
        resistivity=5.0e-3,
        dt=1.0e-3,
        t_end=0.05,
        save_every=10,
        verbose=False,
    )
    settings.update(overrides)
    return mhx.Simulation(**settings)


def test_compressible_run_reports_plots_and_saves(tmp_path, capsys) -> None:
    result = make_simulation().run()

    assert result.trajectory.times.shape[0] == 5
    assert result.diagnostics["final_density_relative_spread"] > 0.0
    scale = float(np.max(np.abs(np.asarray(result.final_state.b_hat))))
    assert result.diagnostics["final_magnetic_divergence_linf"] < 1.0e-10 * scale
    assert result.final_time == pytest.approx(0.05)

    result.print_summary()
    captured = capsys.readouterr().out
    assert "MHX compressible result" in captured
    assert "Density spread" in captured

    figure = result.plot(tmp_path / "summary.png")
    assert figure.exists()
    data = result.save(tmp_path)
    payload = np.load(data)
    assert payload["dimension"] == 3
    assert payload["density"].shape == (5, 8, 8, 4)
    assert payload["velocity"].shape == (5, 3, 8, 8, 4)
    # Log-density evolution keeps density positive by construction.
    assert float(payload["density"].min()) > 0.0


def test_compressible_validation_and_equilibrium_errors() -> None:
    with pytest.raises(ValueError, match="three grid sizes"):
        make_simulation(shape=(8, 8))
    with pytest.raises(TypeError, match="initial_fields"):
        mhx.Simulation(
            shape=(8, 8, 4),
            equations="compressible",
            dt=1.0e-3,
            t_end=0.01,
            verbose=False,
        ).run()
    with pytest.raises(ValueError, match="equations must be"):
        mhx.Simulation(shape=(8, 8), equations="unknown")
