"""Small reduced-MHD tearing-mode smoke benchmark."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from mhx.config import RunConfig
from mhx.diagnostics import compute_reduced_mhd_diagnostics
from mhx.equations.reduced_mhd import reduced_mhd_rhs
from mhx.grids import CartesianGrid
from mhx.numerics import spectral_diffusion_preconditioner
from mhx.physics import CosineTearingEquilibrium, build_equilibrium, build_physics_terms
from mhx.state import ReducedMHDParams, ReducedMHDState, ReducedMHDTrajectory
from mhx.time_integrators import evolve_backward_euler, evolve_rk4


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
) -> tuple[ReducedMHDTrajectory, dict[str, Any]]:
    """Run the deterministic FAST reduced-MHD smoke benchmark."""
    numerics = config.numerics.validated()
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
        plugin_entry_point_groups=config.physics.plugin_entry_point_groups,
    )
    steps = config.time.steps

    def rhs(state: ReducedMHDState) -> ReducedMHDState:
        return reduced_mhd_rhs(
            state,
            params,
            lengths=grid.lengths,
            terms=terms,
            dealiasing=numerics.dealiasing,
        )

    active_rhs = jax.jit(rhs) if numerics.enable_jit else rhs
    solver_diagnostics: dict[str, Any] = {}
    if numerics.time_integrator == "rk4":
        trajectory = evolve_rk4(
            state0,
            active_rhs,
            dt=config.time.dt,
            steps=steps,
            save_every=config.time.save_every,
            t0=config.time.t0,
        )
    else:
        preconditioner = (
            None
            if numerics.preconditioner == "none"
            else spectral_diffusion_preconditioner(
                params,
                lengths=grid.lengths,
                dt=config.time.dt,
            )
        )
        implicit = evolve_backward_euler(
            state0,
            active_rhs,
            dt=config.time.dt,
            steps=steps,
            save_every=config.time.save_every,
            t0=config.time.t0,
            preconditioner=preconditioner,
            rtol=numerics.rtol,
            atol=numerics.atol,
            max_steps=numerics.nonlinear_max_steps,
            linear_restart=numerics.linear_restart,
            linear_max_restarts=numerics.linear_max_restarts,
        )
        trajectory = implicit.trajectory
        solver_diagnostics = {
            "implicit_converged": bool(jnp.all(implicit.converged)),
            "implicit_linear_converged": bool(jnp.all(implicit.linear_converged)),
            "implicit_max_residual_norm": float(jnp.max(implicit.residual_norms)),
            "implicit_nonlinear_iterations": int(
                jnp.sum(implicit.nonlinear_iterations)
            ),
            "implicit_linear_iterations": int(jnp.sum(implicit.linear_iterations)),
        }
        if not solver_diagnostics["implicit_converged"]:
            raise RuntimeError("backward-Euler nonlinear solve did not converge")
        if not solver_diagnostics["implicit_linear_converged"]:
            raise RuntimeError("backward-Euler linear solve did not converge")
    diagnostics = {
        "n_steps": float(steps),
        "spatial_method": numerics.spatial_method,
        "dealiasing": numerics.dealiasing,
        "time_integrator": numerics.time_integrator,
        "linear_solver": numerics.linear_solver,
        "nonlinear_solver": numerics.nonlinear_solver,
        "preconditioner": numerics.preconditioner,
        "equilibrium": config.physics.equilibrium,
        "equilibrium_parameters": dict(equilibrium_parameters),
        "physics_plugin_modules": list(config.physics.plugin_modules),
        "physics_plugin_entry_point_groups": list(config.physics.plugin_entry_point_groups),
        "physics_terms": list(config.physics.rhs_terms),
        "diagnostic_plugin_modules": list(config.diagnostics.plugin_modules),
        "diagnostic_plugin_entry_point_groups": list(
            config.diagnostics.plugin_entry_point_groups
        ),
        "final_time": float(trajectory.times[-1]),
    }
    diagnostics.update(solver_diagnostics)
    diagnostics.update(
        compute_reduced_mhd_diagnostics(
            trajectory,
            initial_state=state0,
            lengths=grid.lengths,
            quantities=config.diagnostics.quantities,
            mode=config.diagnostics.mode,
            fit_time_window=config.diagnostics.fit_time_window,
            plugin_modules=config.diagnostics.plugin_modules,
            plugin_entry_point_groups=config.diagnostics.plugin_entry_point_groups,
        )
    )
    return trajectory, diagnostics
