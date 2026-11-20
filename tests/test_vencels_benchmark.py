"""Tests for the Vencels Orszag--Tang benchmark run and reference-data loaders."""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from mhx.benchmarks.vencels import (
    VENCELS_CSV,
    VENCELS_DOMAIN_SIZE,
    load_vencels_reference,
    load_vencels_spectrum,
    run_orszag_tang,
    vencels_orszag_tang_initial_state,
)
from mhx.config import MeshConfig
from mhx.grids import CartesianGrid


def _grid(n: int) -> CartesianGrid:
    return CartesianGrid.from_mesh_config(
        MeshConfig(
            shape=(n, n),
            lower=(0.0, 0.0),
            upper=(VENCELS_DOMAIN_SIZE, VENCELS_DOMAIN_SIZE),
        )
    )


def test_constants() -> None:
    assert VENCELS_DOMAIN_SIZE == 10.0
    assert VENCELS_CSV.name == "vencelsdata.csv"
    assert VENCELS_CSV.exists()


def test_initial_state_matches_analytic_formula() -> None:
    grid = _grid(8)
    state = vencels_orszag_tang_initial_state(grid)
    x, y = grid.mesh()
    k = 2.0 * np.pi / 10.0

    psi = 0.2 * (np.cos(k * y) / k + 0.5 * np.cos(2.0 * k * x) / k)
    omega = 0.2 * k * (np.cos(k * x) + np.cos(k * y))

    assert state.psi.shape == (8, 8)
    np.testing.assert_allclose(state.psi, psi, atol=1e-13)
    np.testing.assert_allclose(state.omega, omega, atol=1e-13)
    assert abs(float(jnp.mean(state.omega))) < 1e-13


def test_run_orszag_tang_outputs_and_energy_budget() -> None:
    result = run_orszag_tang(
        shape=(16, 16), resistivity=0.02, viscosity=0.02, dt=0.01, t_end=0.06, save_every=2
    )

    assert set(result) == {
        "time",
        "psi",
        "omega",
        "current_density",
        "magnetic_energy",
        "kinetic_energy",
        "total_energy",
        "final_divergence",
        "grid",
    }
    np.testing.assert_allclose(result["time"], [0.0, 0.02, 0.04, 0.06], atol=1e-12)
    for key in ("psi", "omega", "current_density"):
        assert result[key].shape == (4, 16, 16)
        assert np.all(np.isfinite(result[key]))
    for key in ("magnetic_energy", "kinetic_energy", "total_energy"):
        assert result[key].shape == (4,)
        assert np.all(np.isfinite(result[key])) and np.all(result[key] > 0.0)
    np.testing.assert_allclose(
        result["total_energy"], result["magnetic_energy"] + result["kinetic_energy"]
    )
    # Dissipative run: total energy must not increase.
    assert np.all(np.diff(result["total_energy"]) <= 1e-12)
    assert result["total_energy"][-1] < result["total_energy"][0]
    assert result["final_divergence"] < 1e-10
    assert result["grid"].shape == (16, 16)
    # The first stored frame is the analytic initial condition.
    initial = vencels_orszag_tang_initial_state(result["grid"])
    np.testing.assert_allclose(result["psi"][0], initial.psi, atol=1e-13)


def test_run_orszag_tang_rejects_indivisible_save_every() -> None:
    with pytest.raises(ValueError, match="divisible by save_every"):
        run_orszag_tang(
            shape=(16, 16), resistivity=0.01, viscosity=0.01, dt=0.01, t_end=0.05, save_every=2
        )


def test_loaders_read_packaged_csv() -> None:
    blue, red, green = load_vencels_reference()
    for curve in (blue, red, green):
        assert callable(curve)
        values = curve(np.linspace(0.0, 1.0, 5))
        assert values.shape == (5,)
        assert np.all(np.isfinite(values))

    k, power = load_vencels_spectrum()
    assert k.ndim == 1 and k.size > 5 and k.shape == power.shape
    assert np.all(np.isfinite(k)) and np.all(np.isfinite(power))
    assert np.all(power > 0.0)
    assert k[0] == pytest.approx(0.6358052354380777)
    assert power[0] == pytest.approx(137.59855127211685)


def test_loaders_return_none_for_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "nope.csv"
    assert load_vencels_reference(missing) == (None, None, None)
    assert load_vencels_spectrum(missing) == (None, None)


def test_loaders_parse_handwritten_csv(tmp_path: Path) -> None:
    rows = [
        "BLUE\tred:\tgreen:\tspectrum_k\tspectrum_power",
        '"2.0, 4.0"\t"2.0, 40.0"\t"2.0, -4.0"\t1.0\t10.0',
        "",
        '"0.0, 0.0"\t"0.0, 0.0"\t"0.0, 0.0"\t2.0\t5.0',
        '"1.0, 2.0"\t\t"1.0, -2.0"',
        '"3.0, 6.0"\t"1.0, 10.0"\t"junk"\t3.0',
    ]
    path = tmp_path / "data.csv"
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    blue, red, green = load_vencels_reference(str(path))
    # Unsorted input is ordered by x; the curves are y = 2x (blue) and y = -2x (green).
    assert float(blue(0.5)) == pytest.approx(1.0)
    assert float(blue(3.0)) == pytest.approx(6.0)
    assert float(blue(4.0)) == pytest.approx(8.0)  # linear extrapolation
    assert float(green(1.5)) == pytest.approx(-3.0)
    assert float(red(1.0)) == pytest.approx(10.0)
    assert float(red(1.5)) == pytest.approx(25.0)

    k, power = load_vencels_spectrum(path)
    # Only the two complete rows with all five columns contribute.
    np.testing.assert_array_equal(k, [1.0, 2.0])
    np.testing.assert_array_equal(power, [10.0, 5.0])
