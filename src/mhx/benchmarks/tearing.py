"""Small reduced-MHD tearing-mode smoke benchmark."""

from __future__ import annotations

from mhx.config import RunConfig
from mhx.diagnostics import compute_reduced_mhd_diagnostics
from mhx.equations.reduced_mhd import reduced_mhd_rhs
from mhx.grids import CartesianGrid
from mhx.physics import CosineTearingEquilibrium, build_equilibrium, build_physics_terms
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
    return CosineTearingEquilibrium(perturbation_amplitude=perturbation_amplitude).initial_state(
        grid
    )


def run_linear_tearing_smoke(
    config: RunConfig,
    *,
    perturbation_amplitude: float = 1.0e-3,
) -> tuple[ReducedMHDTrajectory, dict[str, float]]:
    """Run the deterministic FAST reduced-MHD smoke benchmark."""
    grid = CartesianGrid.from_mesh_config(config.mesh)
    equilibrium_parameters = dict(config.physics.equilibrium_parameters)
    if not equilibrium_parameters and config.physics.equilibrium == "cosine_tearing":
        equilibrium_parameters = {"perturbation_amplitude": perturbation_amplitude}
    equilibrium = build_equilibrium(config.physics.equilibrium, equilibrium_parameters)
    state0 = equilibrium.initial_state(grid)
    params = ReducedMHDParams(
        resistivity=config.physics.resistivity,
        viscosity=config.physics.viscosity,
    )
    terms = build_physics_terms(
        config.physics.rhs_terms,
        config.physics.term_parameters,
        plugin_modules=config.physics.plugin_modules,
    )
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
    diagnostics = {
        "n_steps": float(steps),
        "equilibrium": config.physics.equilibrium,
        "equilibrium_parameters": dict(equilibrium_parameters),
        "physics_plugin_modules": list(config.physics.plugin_modules),
        "physics_terms": list(config.physics.rhs_terms),
        "diagnostic_plugin_modules": list(config.diagnostics.plugin_modules),
        "final_time": float(trajectory.times[-1]),
    }
    diagnostics.update(
        compute_reduced_mhd_diagnostics(
            trajectory,
            initial_state=state0,
            lengths=grid.lengths,
            quantities=config.diagnostics.quantities,
            mode=config.diagnostics.mode,
            fit_time_window=config.diagnostics.fit_time_window,
            plugin_modules=config.diagnostics.plugin_modules,
        )
    )
    return trajectory, diagnostics
