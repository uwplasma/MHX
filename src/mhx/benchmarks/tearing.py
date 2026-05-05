"""Small reduced-MHD tearing-mode smoke benchmark."""

from __future__ import annotations

import jax.numpy as jnp

from mhx.config import RunConfig
from mhx.diagnostics import (
    fit_exponential_growth,
    select_fit_window,
    total_energy,
    trajectory_energies,
    trajectory_mode_amplitude,
)
from mhx.equations.reduced_mhd import reduced_mhd_rhs
from mhx.grids import CartesianGrid
from mhx.physics import build_physics_terms
from mhx.state import ReducedMHDParams, ReducedMHDState, ReducedMHDTrajectory
from mhx.time_integrators import evolve_rk4


def linear_tearing_initial_state(
    grid: CartesianGrid,
    *,
    perturbation_amplitude: float = 1.0e-3,
) -> ReducedMHDState:
    r"""Return a periodic current-sheet-like reduced-MHD initial condition.

    This is a deterministic FAST smoke benchmark, not yet an FKR-calibrated
    tearing eigenmode. The equilibrium is ``ψ₀ = cos(y)`` with a small
    ``cos(x) cos(y)`` perturbation.
    """
    x, y = grid.mesh()
    length_x, length_y = grid.lengths
    psi_equilibrium = jnp.cos(2.0 * jnp.pi * y / length_y)
    perturbation = perturbation_amplitude * jnp.cos(2.0 * jnp.pi * x / length_x) * jnp.cos(
        2.0 * jnp.pi * y / length_y
    )
    omega = jnp.zeros_like(psi_equilibrium)
    return ReducedMHDState(psi=psi_equilibrium + perturbation, omega=omega)


def run_linear_tearing_smoke(
    config: RunConfig,
    *,
    perturbation_amplitude: float = 1.0e-3,
) -> tuple[ReducedMHDTrajectory, dict[str, float]]:
    """Run the deterministic FAST reduced-MHD smoke benchmark."""
    grid = CartesianGrid.from_mesh_config(config.mesh)
    state0 = linear_tearing_initial_state(grid, perturbation_amplitude=perturbation_amplitude)
    params = ReducedMHDParams(
        resistivity=config.physics.resistivity,
        viscosity=config.physics.viscosity,
    )
    terms = build_physics_terms(config.physics.rhs_terms, config.physics.term_parameters)
    steps = max(1, round((config.time.t1 - config.time.t0) / config.time.dt))

    def rhs(state: ReducedMHDState) -> ReducedMHDState:
        return reduced_mhd_rhs(state, params, lengths=grid.lengths, terms=terms)

    trajectory = evolve_rk4(
        state0,
        rhs,
        dt=config.time.dt,
        steps=steps,
        save_every=config.time.save_every,
    )
    energies = trajectory_energies(trajectory, lengths=grid.lengths)
    mode = config.diagnostics.mode
    amplitudes = trajectory_mode_amplitude(trajectory, mode=mode)
    fit_times, fit_amplitudes = select_fit_window(
        trajectory.times,
        amplitudes,
        window=config.diagnostics.fit_time_window,
    )
    diagnostics = {
        "n_steps": float(steps),
        "physics_terms": list(config.physics.rhs_terms),
        "initial_total_energy": float(total_energy(state0, lengths=grid.lengths)),
        "final_total_energy": float(energies["total"][-1]),
        "final_magnetic_energy": float(energies["magnetic"][-1]),
        "final_kinetic_energy": float(energies["kinetic"][-1]),
        "final_time": float(trajectory.times[-1]),
        "diagnostic_mode": list(mode),
        "fit_time_window": (
            None
            if config.diagnostics.fit_time_window is None
            else list(config.diagnostics.fit_time_window)
        ),
        "fit_sample_count": float(fit_times.shape[0]),
        "initial_mode_amplitude": float(amplitudes[0]),
        "final_mode_amplitude": float(amplitudes[-1]),
        "gamma_fit": float(fit_exponential_growth(fit_times, fit_amplitudes)),
    }
    return trajectory, diagnostics
