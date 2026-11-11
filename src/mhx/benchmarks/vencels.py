"""Orszag--Tang run and reference data for the Vencels kinetic comparison.

The reduced-MHD Orszag--Tang vortex is evolved on the ``L = 10 d_i`` box of the
Vencels kinetic simulation, scaled to ``delta_B / B_0 = 0.2``. The reference
curves (magnetic, electron, and ion energy changes, plus the magnetic power
spectrum) ship with the package in ``data/vencelsdata.csv``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
from scipy.interpolate import interp1d

from mhx.config import MeshConfig
from mhx.diagnostics import kinetic_energy, magnetic_divergence_linf, magnetic_energy
from mhx.equations.reduced_mhd import current_density, reduced_mhd_rhs
from mhx.grids import CartesianGrid
from mhx.state import ReducedMHDParams, ReducedMHDState
from mhx.time_integrators import evolve_rk4

# The Vencels box is ``L_x = 10 d_i``, so the Alfven time is ``t_A = 10``.
VENCELS_DOMAIN_SIZE = 10.0
VENCELS_CSV = Path(__file__).resolve().parent / "data" / "vencelsdata.csv"


def vencels_orszag_tang_initial_state(grid: CartesianGrid) -> ReducedMHDState:
    """Return the Orszag--Tang initial condition scaled to ``delta_B / B_0 = 0.2``.

    The flux and stream-function amplitudes are scaled to match the Vencels
    bulk-flow and perturbation magnitudes, so this is not the unit-amplitude
    :class:`mhx.physics.OrszagTangEquilibrium`.
    """
    x, y = grid.mesh()
    length_x, length_y = grid.lengths
    kx = 2.0 * jnp.pi / length_x
    ky = 2.0 * jnp.pi / length_y

    psi = 0.2 * ((1.0 / ky) * jnp.cos(ky * y) + (0.5 / kx) * jnp.cos(2.0 * kx * x))
    omega = 0.2 * (kx * jnp.cos(kx * x) + ky * jnp.cos(ky * y))

    return ReducedMHDState(psi=psi, omega=omega)


def run_orszag_tang(
    *,
    shape: tuple[int, int],
    resistivity: float,
    viscosity: float,
    dt: float,
    t_end: float,
    save_every: int,
) -> dict[str, Any]:
    """Evolve the Orszag--Tang vortex and return time histories plus states.

    Returns a dictionary with ``time``, ``psi``, ``omega``, ``current_density``,
    the magnetic, kinetic and total energy histories, ``final_divergence`` and
    the ``grid``. The histories include the initial state as their first entry.
    """
    grid = CartesianGrid.from_mesh_config(
        MeshConfig(
            shape=shape,
            lower=(0.0, 0.0),
            upper=(VENCELS_DOMAIN_SIZE, VENCELS_DOMAIN_SIZE),
        )
    )
    initial_state = vencels_orszag_tang_initial_state(grid)
    params = ReducedMHDParams(resistivity=resistivity, viscosity=viscosity)

    def rhs(state: ReducedMHDState) -> ReducedMHDState:
        return reduced_mhd_rhs(state, params, lengths=grid.lengths)

    steps = int(round(t_end / dt))
    if steps % save_every != 0:
        raise ValueError("t_end / dt must be divisible by save_every")

    trajectory = evolve_rk4(initial_state, rhs, dt=dt, steps=steps, save_every=save_every)

    time = np.concatenate(([0.0], np.asarray(trajectory.times, dtype=np.float64)))
    psi = np.concatenate(
        (
            np.asarray(initial_state.psi, dtype=np.float64)[None, ...],
            np.asarray(trajectory.states.psi, dtype=np.float64),
        ),
        axis=0,
    )
    omega = np.concatenate(
        (
            np.asarray(initial_state.omega, dtype=np.float64)[None, ...],
            np.asarray(trajectory.states.omega, dtype=np.float64),
        ),
        axis=0,
    )
    current = np.asarray(
        [np.asarray(current_density(frame, lengths=grid.lengths)) for frame in psi],
        dtype=np.float64,
    )
    states = tuple(
        ReducedMHDState(psi=jnp.asarray(psi[i]), omega=jnp.asarray(omega[i]))
        for i in range(time.size)
    )
    magnetic = np.asarray(
        [float(magnetic_energy(state, lengths=grid.lengths)) for state in states],
        dtype=np.float64,
    )
    kinetic = np.asarray(
        [float(kinetic_energy(state, lengths=grid.lengths)) for state in states],
        dtype=np.float64,
    )

    return {
        "time": time,
        "psi": psi,
        "omega": omega,
        "current_density": current,
        "magnetic_energy": magnetic,
        "kinetic_energy": kinetic,
        "total_energy": magnetic + kinetic,
        "final_divergence": float(magnetic_divergence_linf(states[-1], lengths=grid.lengths)),
        "grid": grid,
    }


def _curve(column: str) -> tuple[list[float], list[float]]:
    """Split one ``"x, y"`` CSV column into x and y lists."""
    x_values: list[float] = []
    y_values: list[float] = []
    cleaned = column.replace('"', "").split(",")
    if len(cleaned) == 2:
        x_values.append(float(cleaned[0]))
        y_values.append(float(cleaned[1]))
    return x_values, y_values


def load_vencels_reference(csv_path: Path = VENCELS_CSV):
    """Return interpolators of the magnetic, electron, and ion energy-change curves.

    The three interpolators take a time in units of the Alfven time and
    extrapolate outside the digitized range. If ``csv_path`` does not exist the
    result is ``(None, None, None)``.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return None, None, None

    curves: list[tuple[list[float], list[float]]] = [([], []), ([], []), ([], [])]
    with open(csv_path, encoding="utf-8") as handle:
        for line in handle.readlines()[1:]:
            columns = line.strip().split("\t")
            for index in range(3):
                if len(columns) > index and columns[index]:
                    x_values, y_values = _curve(columns[index])
                    curves[index][0].extend(x_values)
                    curves[index][1].extend(y_values)

    interpolators = []
    for x_values, y_values in curves:
        ordered_x, ordered_y = zip(*sorted(zip(x_values, y_values, strict=True)), strict=True)
        interpolators.append(
            interp1d(ordered_x, ordered_y, bounds_error=False, fill_value="extrapolate")
        )
    return tuple(interpolators)


def load_vencels_spectrum(csv_path: Path = VENCELS_CSV):
    """Return the Vencels magnetic power spectrum as ``(k, power)`` arrays.

    If ``csv_path`` does not exist the result is ``(None, None)``.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return None, None
    k_ref: list[float] = []
    power_ref: list[float] = []
    with open(csv_path, encoding="utf-8") as handle:
        for line in handle.readlines()[1:]:
            if not line.strip():
                continue
            columns = line.strip().split("\t")
            if len(columns) >= 5 and columns[3] and columns[4]:
                k_ref.append(float(columns[3]))
                power_ref.append(float(columns[4]))
    return np.array(k_ref), np.array(power_ref)
