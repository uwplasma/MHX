"""Differentiable Kelvin--Helmholtz benchmark helpers.

This module factors the clean, reusable pieces behind the notebook examples:
smooth double-shear initial conditions, passive-dye advection, scalar
diagnostics, and small differentiable objectives.  The equations are the
periodic incompressible/reduced-MHD hydrodynamic limit used by the existing
Kelvin--Helmholtz notebook: ``psi=0`` and vorticity evolves through the
reduced-MHD vorticity equation while a passive dye is advected by the
streamfunction velocity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array

from mhx.config import MeshConfig
from mhx.equations.reduced_mhd import poisson_bracket, reduced_mhd_rhs, stream_function
from mhx.grids import CartesianGrid
from mhx.numerics.spectral import laplacian
from mhx.state import ReducedMHDParams, ReducedMHDState
from mhx.time_integrators import evolve_rk4


class KelvinHelmholtzDyeState(NamedTuple):
    """PyTree state for reduced-MHD Kelvin--Helmholtz plus passive dye."""

    mhd: ReducedMHDState
    dye: Array


class KelvinHelmholtzDyeTrajectory(NamedTuple):
    """Saved trajectory samples for the passive-dye Kelvin--Helmholtz example."""

    times: Array
    states: KelvinHelmholtzDyeState


@dataclass(frozen=True)
class KelvinHelmholtzConfig:
    """Configuration for a smooth periodic Kelvin--Helmholtz run."""

    shape: tuple[int, int] = (32, 64)
    lower: tuple[float, float] = (0.0, 0.0)
    upper: tuple[float, float] = (1.0, 2.0)
    viscosity: float = 1.0e-3
    resistivity: float = 0.0
    dt: float = 2.0e-3
    t_end: float = 0.2
    save_every: int = 20
    shear_width: float = 0.05
    perturbation_width: float = 0.2
    perturbation_amplitude: float = 1.0e-2
    flow_speed: float = 1.0
    y1: float = 0.5
    y2: float = 1.5

    @property
    def steps(self) -> int:
        """Number of RK4 steps implied by ``dt`` and ``t_end``."""
        return int(round(self.t_end / self.dt))

    def validated(self) -> KelvinHelmholtzConfig:
        """Return ``self`` after validating lightweight numerical controls."""
        MeshConfig(shape=self.shape, lower=self.lower, upper=self.upper).validated()
        if self.viscosity < 0.0:
            raise ValueError("viscosity must be non-negative")
        if self.resistivity < 0.0:
            raise ValueError("resistivity must be non-negative")
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")
        if self.t_end <= 0.0:
            raise ValueError("t_end must be positive")
        if self.save_every < 1:
            raise ValueError("save_every must be >= 1")
        if self.steps < 1:
            raise ValueError("configuration must advance at least one step")
        if self.shear_width <= 0.0:
            raise ValueError("shear_width must be positive")
        if self.perturbation_width <= 0.0:
            raise ValueError("perturbation_width must be positive")
        if self.flow_speed <= 0.0:
            raise ValueError("flow_speed must be positive")
        return self


@dataclass(frozen=True)
class KelvinHelmholtzRunResult:
    """Result bundle for a FAST Kelvin--Helmholtz run."""

    config: KelvinHelmholtzConfig
    grid: CartesianGrid
    params: ReducedMHDParams
    initial_state: KelvinHelmholtzDyeState
    trajectory: KelvinHelmholtzDyeTrajectory
    entropy: Array

    @property
    def final_state(self) -> KelvinHelmholtzDyeState:
        """Final saved state."""
        return jax.tree_util.tree_map(lambda leaf: leaf[-1], self.trajectory.states)


def kelvin_helmholtz_grid(config: KelvinHelmholtzConfig) -> CartesianGrid:
    """Build the periodic grid for a Kelvin--Helmholtz configuration."""
    cfg = config.validated()
    return CartesianGrid.from_mesh_config(
        MeshConfig(shape=cfg.shape, lower=cfg.lower, upper=cfg.upper)
    )


def kelvin_helmholtz_initial_state(
    grid: CartesianGrid,
    *,
    shear_width: float = 0.05,
    perturbation_width: float = 0.2,
    perturbation_amplitude: float = 1.0e-2,
    flow_speed: float = 1.0,
    y1: float = 0.5,
    y2: float = 1.5,
) -> KelvinHelmholtzDyeState:
    """Return the smooth double-shear Kelvin--Helmholtz initial condition."""
    x, y = grid.mesh()
    length_x, _ = grid.lengths
    wavenumber_x = 2.0 * jnp.pi / length_x
    sech1 = 1.0 / jnp.cosh((y - y1) / shear_width)
    sech2 = 1.0 / jnp.cosh((y - y2) / shear_width)
    dy_ux = (flow_speed / shear_width) * (sech1**2 - sech2**2)
    envelope1 = jnp.exp(-((y - y1) ** 2) / (perturbation_width**2))
    envelope2 = jnp.exp(-((y - y2) ** 2) / (perturbation_width**2))
    dx_uy = (
        perturbation_amplitude
        * wavenumber_x
        * jnp.cos(wavenumber_x * x)
        * (envelope1 + envelope2)
    )
    omega = dx_uy - dy_ux
    psi = jnp.zeros_like(x)
    dye = 0.5 * (
        jnp.tanh((y - y2) / shear_width)
        - jnp.tanh((y - y1) / shear_width)
        + 2.0
    )
    return KelvinHelmholtzDyeState(
        mhd=ReducedMHDState(psi=psi, omega=omega),
        dye=dye,
    )


def kelvin_helmholtz_initial_state_from_config(
    grid: CartesianGrid,
    config: KelvinHelmholtzConfig,
) -> KelvinHelmholtzDyeState:
    """Return the initial state represented by ``config`` on ``grid``."""
    cfg = config.validated()
    return kelvin_helmholtz_initial_state(
        grid,
        shear_width=cfg.shear_width,
        perturbation_width=cfg.perturbation_width,
        perturbation_amplitude=cfg.perturbation_amplitude,
        flow_speed=cfg.flow_speed,
        y1=cfg.y1,
        y2=cfg.y2,
    )


def kelvin_helmholtz_dye_rhs(
    state: KelvinHelmholtzDyeState,
    params: ReducedMHDParams,
    *,
    lengths: tuple[float, float],
) -> KelvinHelmholtzDyeState:
    """Return RHS for reduced-MHD Kelvin--Helmholtz plus passive dye."""
    mhd_rhs = reduced_mhd_rhs(state.mhd, params, lengths=lengths)
    phi = stream_function(state.mhd.omega, lengths=lengths)
    dye_rhs = -poisson_bracket(phi, state.dye, lengths=lengths) + params.viscosity * laplacian(
        state.dye,
        lengths=lengths,
    )
    return KelvinHelmholtzDyeState(mhd=mhd_rhs, dye=dye_rhs)


def dye_entropy(dye: Array, grid: CartesianGrid) -> Array:
    """Return volume-integrated passive-dye entropy."""
    dye_safe = jnp.clip(dye, 1.0e-12, 1.0)
    return jnp.mean(-dye_safe * jnp.log(dye_safe)) * grid.lengths[0] * grid.lengths[1]


def dye_entropy_history(
    trajectory: KelvinHelmholtzDyeTrajectory,
    grid: CartesianGrid,
) -> Array:
    """Return passive-dye entropy for each saved state."""
    return jax.vmap(lambda dye: dye_entropy(dye, grid))(trajectory.states.dye)


def run_kelvin_helmholtz_dye(
    config: KelvinHelmholtzConfig | None = None,
) -> KelvinHelmholtzRunResult:
    """Run a FAST smooth Kelvin--Helmholtz passive-dye simulation."""
    cfg = (config or KelvinHelmholtzConfig()).validated()
    grid = kelvin_helmholtz_grid(cfg)
    params = ReducedMHDParams(resistivity=cfg.resistivity, viscosity=cfg.viscosity)
    initial_state = kelvin_helmholtz_initial_state_from_config(grid, cfg)

    def rhs(state: KelvinHelmholtzDyeState) -> KelvinHelmholtzDyeState:
        return kelvin_helmholtz_dye_rhs(state, params, lengths=grid.lengths)

    trajectory_raw = evolve_rk4(
        initial_state,
        rhs,
        dt=cfg.dt,
        steps=cfg.steps,
        save_every=cfg.save_every,
    )
    trajectory = KelvinHelmholtzDyeTrajectory(
        times=trajectory_raw.times,
        states=trajectory_raw.states,
    )
    return KelvinHelmholtzRunResult(
        config=cfg,
        grid=grid,
        params=params,
        initial_state=initial_state,
        trajectory=trajectory,
        entropy=dye_entropy_history(trajectory, grid),
    )


def kelvin_helmholtz_entropy_objective(
    perturbation_amplitude: Array,
    config: KelvinHelmholtzConfig | None = None,
) -> Array:
    """Return final dye entropy as a differentiable scalar objective."""
    cfg = config or KelvinHelmholtzConfig(
        shape=(16, 32),
        dt=1.0e-3,
        t_end=8.0e-3,
        save_every=8,
    )
    cfg = KelvinHelmholtzConfig(
        shape=cfg.shape,
        lower=cfg.lower,
        upper=cfg.upper,
        viscosity=cfg.viscosity,
        resistivity=cfg.resistivity,
        dt=cfg.dt,
        t_end=cfg.t_end,
        save_every=cfg.save_every,
        shear_width=cfg.shear_width,
        perturbation_width=cfg.perturbation_width,
        perturbation_amplitude=perturbation_amplitude,
        flow_speed=cfg.flow_speed,
        y1=cfg.y1,
        y2=cfg.y2,
    )
    return run_kelvin_helmholtz_dye(cfg).entropy[-1]


def kelvin_helmholtz_entropy_value_and_grad(
    perturbation_amplitude: float,
    config: KelvinHelmholtzConfig | None = None,
) -> tuple[Array, Array]:
    """Return final-entropy objective and reverse-mode gradient."""

    def objective(amplitude: Array) -> Array:
        return kelvin_helmholtz_entropy_objective(amplitude, config)

    return jax.value_and_grad(objective)(jnp.asarray(perturbation_amplitude))


def kelvin_helmholtz_entropy_jvp(
    perturbation_amplitude: float,
    tangent: float,
    config: KelvinHelmholtzConfig | None = None,
) -> tuple[Array, Array]:
    """Return final-entropy objective and forward-mode tangent."""

    def objective(amplitude: Array) -> Array:
        return kelvin_helmholtz_entropy_objective(amplitude, config)

    return jax.jvp(
        objective,
        (jnp.asarray(perturbation_amplitude),),
        (jnp.asarray(tangent),),
    )
