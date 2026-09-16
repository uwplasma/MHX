"""Differentiable Kelvin--Helmholtz benchmark helpers and default runner.

This module factors the clean, reusable pieces behind the Kelvin--Helmholtz
examples: smooth double-shear initial conditions, passive-dye advection,
scalar diagnostics, and small differentiable objectives.  The equations are
the periodic incompressible/reduced-MHD hydrodynamic limit: ``psi=0`` and
vorticity evolves through the reduced-MHD vorticity equation while a
passive dye is advected by the streamfunction velocity.

Running this file as a script (``python examples/kelvin_helmholtz.py``)
executes the FAST incompressible passive-dye benchmark and writes snapshot
and entropy figures to ``$MHX_EXAMPLE_OUTDIR_ROOT/kelvin_helmholtz_incompressible``
(default ``outputs/examples``).  It reuses the same setup style as Lecoanet
et al. and is deliberately small enough to run locally; the production-scale
hints live in the ``__main__`` block at the bottom.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from mhx.config import MeshConfig
from mhx.equations.reduced_mhd import poisson_bracket, reduced_mhd_rhs, stream_function
from mhx.grids import CartesianGrid
from mhx.io import write_manifest
from mhx.numerics.spectral import fft_derivative, laplacian
from mhx.state import ReducedMHDParams, ReducedMHDState
from mhx.time_integrators import evolve_rk4


class CompressibleMHDState(NamedTuple):
    """2D conservative compressible-MHD state with a passive dye scalar."""

    density: Array
    momentum_x: Array
    momentum_y: Array
    total_energy: Array
    magnetic_x: Array
    magnetic_y: Array
    dye_density: Array


class CompressibleMHDParams(NamedTuple):
    """Parameters for the smooth pedagogical compressible-MHD model."""

    gamma: float = 5.0 / 3.0
    density_floor: float = 1.0e-8
    pressure_floor: float = 1.0e-8


class CompressibleMHDPrimitive(NamedTuple):
    """Primitive variables for 2D compressible MHD."""

    density: Array
    velocity_x: Array
    velocity_y: Array
    pressure: Array
    magnetic_x: Array
    magnetic_y: Array
    dye: Array


class CompressibleMHDTrajectory(NamedTuple):
    """Saved compressible-MHD trajectory samples."""

    times: Array
    states: CompressibleMHDState


def conservative_from_primitive(
    primitive: CompressibleMHDPrimitive,
    params: CompressibleMHDParams,
) -> CompressibleMHDState:
    """Convert primitive variables to conservative variables."""
    density = jnp.maximum(primitive.density, params.density_floor)
    kinetic = 0.5 * density * (primitive.velocity_x**2 + primitive.velocity_y**2)
    magnetic = 0.5 * (primitive.magnetic_x**2 + primitive.magnetic_y**2)
    pressure = jnp.maximum(primitive.pressure, params.pressure_floor)
    total_energy = pressure / (params.gamma - 1.0) + kinetic + magnetic
    return CompressibleMHDState(
        density=density,
        momentum_x=density * primitive.velocity_x,
        momentum_y=density * primitive.velocity_y,
        total_energy=total_energy,
        magnetic_x=primitive.magnetic_x,
        magnetic_y=primitive.magnetic_y,
        dye_density=density * primitive.dye,
    )


def primitive_from_conservative(
    state: CompressibleMHDState,
    params: CompressibleMHDParams,
) -> CompressibleMHDPrimitive:
    """Convert conservative variables to primitive variables with floors."""
    density = jnp.maximum(state.density, params.density_floor)
    velocity_x = state.momentum_x / density
    velocity_y = state.momentum_y / density
    kinetic = 0.5 * density * (velocity_x**2 + velocity_y**2)
    magnetic = 0.5 * (state.magnetic_x**2 + state.magnetic_y**2)
    pressure = (params.gamma - 1.0) * (state.total_energy - kinetic - magnetic)
    pressure = jnp.maximum(pressure, params.pressure_floor)
    dye = state.dye_density / density
    return CompressibleMHDPrimitive(
        density=density,
        velocity_x=velocity_x,
        velocity_y=velocity_y,
        pressure=pressure,
        magnetic_x=state.magnetic_x,
        magnetic_y=state.magnetic_y,
        dye=dye,
    )


def compressible_mhd_rhs(
    state: CompressibleMHDState,
    params: CompressibleMHDParams,
    *,
    lengths: tuple[float, float],
) -> CompressibleMHDState:
    """Return the smooth conservative ideal-MHD RHS on a periodic domain."""
    primitive = primitive_from_conservative(state, params)
    flux_x, flux_y = compressible_mhd_fluxes(primitive, state)

    def divergence(x_flux, y_flux):
        return fft_derivative(x_flux, axis=0, length=lengths[0]) + fft_derivative(
            y_flux,
            axis=1,
            length=lengths[1],
        )

    return CompressibleMHDState(
        density=-divergence(flux_x.density, flux_y.density),
        momentum_x=-divergence(flux_x.momentum_x, flux_y.momentum_x),
        momentum_y=-divergence(flux_x.momentum_y, flux_y.momentum_y),
        total_energy=-divergence(flux_x.total_energy, flux_y.total_energy),
        magnetic_x=-divergence(flux_x.magnetic_x, flux_y.magnetic_x),
        magnetic_y=-divergence(flux_x.magnetic_y, flux_y.magnetic_y),
        dye_density=-divergence(flux_x.dye_density, flux_y.dye_density),
    )


def compressible_mhd_fluxes(
    primitive: CompressibleMHDPrimitive,
    state: CompressibleMHDState,
) -> tuple[CompressibleMHDState, CompressibleMHDState]:
    """Return ideal-MHD conservative fluxes in x and y."""
    velocity_dot_magnetic = (
        primitive.velocity_x * primitive.magnetic_x
        + primitive.velocity_y * primitive.magnetic_y
    )
    magnetic_pressure = 0.5 * (primitive.magnetic_x**2 + primitive.magnetic_y**2)
    total_pressure = primitive.pressure + magnetic_pressure
    total_enthalpy = state.total_energy + total_pressure
    induction_flux = primitive.velocity_x * primitive.magnetic_y - primitive.velocity_y * (
        primitive.magnetic_x
    )
    flux_x = CompressibleMHDState(
        density=state.momentum_x,
        momentum_x=state.momentum_x * primitive.velocity_x + total_pressure
        - primitive.magnetic_x**2,
        momentum_y=state.momentum_y * primitive.velocity_x
        - primitive.magnetic_x * primitive.magnetic_y,
        total_energy=total_enthalpy * primitive.velocity_x
        - velocity_dot_magnetic * primitive.magnetic_x,
        magnetic_x=jnp.zeros_like(primitive.magnetic_x),
        magnetic_y=induction_flux,
        dye_density=state.dye_density * primitive.velocity_x,
    )
    flux_y = CompressibleMHDState(
        density=state.momentum_y,
        momentum_x=state.momentum_x * primitive.velocity_y
        - primitive.magnetic_x * primitive.magnetic_y,
        momentum_y=state.momentum_y * primitive.velocity_y + total_pressure
        - primitive.magnetic_y**2,
        total_energy=total_enthalpy * primitive.velocity_y
        - velocity_dot_magnetic * primitive.magnetic_y,
        magnetic_x=-induction_flux,
        magnetic_y=jnp.zeros_like(primitive.magnetic_y),
        dye_density=state.dye_density * primitive.velocity_y,
    )
    return flux_x, flux_y


def uniform_compressible_mhd_state(
    *,
    shape: tuple[int, int],
    density: float = 1.0,
    velocity: tuple[float, float] = (0.0, 0.0),
    pressure: float = 1.0,
    magnetic: tuple[float, float] = (0.0, 0.0),
    dye: float = 0.0,
    params: CompressibleMHDParams | None = None,
) -> CompressibleMHDState:
    """Return a uniform conservative state for tests and sanity checks."""
    active_params = params or CompressibleMHDParams()
    primitive = CompressibleMHDPrimitive(
        density=jnp.full(shape, density),
        velocity_x=jnp.full(shape, velocity[0]),
        velocity_y=jnp.full(shape, velocity[1]),
        pressure=jnp.full(shape, pressure),
        magnetic_x=jnp.full(shape, magnetic[0]),
        magnetic_y=jnp.full(shape, magnetic[1]),
        dye=jnp.full(shape, dye),
    )
    return conservative_from_primitive(primitive, active_params)


KELVIN_HELMHOLTZ_VALIDATION_SCHEMA = "mhx.validation.kelvin_helmholtz.v1"


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


@dataclass(frozen=True)
class CompressibleKelvinHelmholtzConfig:
    """Configuration for a smooth low-Mach compressible-MHD KH tutorial."""

    shape: tuple[int, int] = (24, 48)
    lower: tuple[float, float] = (0.0, 0.0)
    upper: tuple[float, float] = (1.0, 2.0)
    gamma: float = 5.0 / 3.0
    dt: float = 5.0e-4
    t_end: float = 1.0e-2
    save_every: int = 10
    density: float = 1.0
    pressure: float = 10.0
    magnetic_field: tuple[float, float] = (0.1, 0.0)
    shear_width: float = 0.05
    perturbation_width: float = 0.2
    perturbation_amplitude: float = 1.0e-2
    flow_speed: float = 0.2
    y1: float = 0.5
    y2: float = 1.5

    @property
    def steps(self) -> int:
        """Number of RK4 steps implied by ``dt`` and ``t_end``."""
        return int(round(self.t_end / self.dt))

    def validated(self) -> CompressibleKelvinHelmholtzConfig:
        """Return ``self`` after validating smooth tutorial controls."""
        MeshConfig(shape=self.shape, lower=self.lower, upper=self.upper).validated()
        if self.gamma <= 1.0:
            raise ValueError("gamma must be greater than one")
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")
        if self.t_end <= 0.0:
            raise ValueError("t_end must be positive")
        if self.save_every < 1:
            raise ValueError("save_every must be >= 1")
        if self.steps < 1:
            raise ValueError("configuration must advance at least one step")
        if self.density <= 0.0:
            raise ValueError("density must be positive")
        if self.pressure <= 0.0:
            raise ValueError("pressure must be positive")
        if self.shear_width <= 0.0:
            raise ValueError("shear_width must be positive")
        if self.perturbation_width <= 0.0:
            raise ValueError("perturbation_width must be positive")
        return self


@dataclass(frozen=True)
class CompressibleKelvinHelmholtzRunResult:
    """Result bundle for a smooth compressible-MHD Kelvin--Helmholtz run."""

    config: CompressibleKelvinHelmholtzConfig
    grid: CartesianGrid
    params: CompressibleMHDParams
    initial_state: CompressibleMHDState
    trajectory: CompressibleMHDTrajectory
    dye_entropy: Array
    density_min: Array
    pressure_min: Array

    @property
    def final_state(self) -> CompressibleMHDState:
        """Final saved conservative state."""
        return jax.tree_util.tree_map(lambda leaf: leaf[-1], self.trajectory.states)


@dataclass(frozen=True)
class KelvinHelmholtzValidationResult:
    """Gated validation bundle for Kelvin--Helmholtz tutorials and examples."""

    primary: KelvinHelmholtzRunResult
    comparison: KelvinHelmholtzRunResult
    compressible: CompressibleKelvinHelmholtzRunResult | None
    diagnostics: dict[str, Any]
    validation: dict[str, Any]


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


def compressible_kelvin_helmholtz_grid(
    config: CompressibleKelvinHelmholtzConfig,
) -> CartesianGrid:
    """Build the periodic grid for the compressible-MHD KH tutorial."""
    cfg = config.validated()
    return CartesianGrid.from_mesh_config(
        MeshConfig(shape=cfg.shape, lower=cfg.lower, upper=cfg.upper)
    )


def compressible_kelvin_helmholtz_initial_state(
    grid: CartesianGrid,
    config: CompressibleKelvinHelmholtzConfig,
) -> CompressibleMHDState:
    """Return a smooth low-Mach compressible-MHD KH initial condition."""
    cfg = config.validated()
    x, y = grid.mesh()
    length_x, _ = grid.lengths
    wavenumber_x = 2.0 * jnp.pi / length_x
    tanh1 = jnp.tanh((y - cfg.y1) / cfg.shear_width)
    tanh2 = jnp.tanh((y - cfg.y2) / cfg.shear_width)
    velocity_x = cfg.flow_speed * (tanh1 - tanh2 - 1.0)
    envelope1 = jnp.exp(-((y - cfg.y1) ** 2) / (cfg.perturbation_width**2))
    envelope2 = jnp.exp(-((y - cfg.y2) ** 2) / (cfg.perturbation_width**2))
    velocity_y = (
        cfg.perturbation_amplitude
        * jnp.sin(wavenumber_x * x)
        * (envelope1 + envelope2)
    )
    dye = 0.5 * (tanh2 - tanh1 + 2.0)
    primitive = CompressibleMHDPrimitive(
        density=jnp.full(grid.shape, cfg.density),
        velocity_x=velocity_x,
        velocity_y=velocity_y,
        pressure=jnp.full(grid.shape, cfg.pressure),
        magnetic_x=jnp.full(grid.shape, cfg.magnetic_field[0]),
        magnetic_y=jnp.full(grid.shape, cfg.magnetic_field[1]),
        dye=dye,
    )
    return conservative_from_primitive(
        primitive,
        CompressibleMHDParams(gamma=cfg.gamma),
    )


def compressible_kelvin_helmholtz_dye_entropy(
    state: CompressibleMHDState,
    grid: CartesianGrid,
    params: CompressibleMHDParams,
) -> Array:
    """Return passive-dye entropy for a compressible-MHD state."""
    primitive = primitive_from_conservative(state, params)
    return dye_entropy(primitive.dye, grid)


def run_compressible_kelvin_helmholtz(
    config: CompressibleKelvinHelmholtzConfig | None = None,
) -> CompressibleKelvinHelmholtzRunResult:
    """Run a smooth low-Mach compressible-MHD Kelvin--Helmholtz tutorial case."""
    cfg = (config or CompressibleKelvinHelmholtzConfig()).validated()
    grid = compressible_kelvin_helmholtz_grid(cfg)
    params = CompressibleMHDParams(gamma=cfg.gamma)
    initial_state = compressible_kelvin_helmholtz_initial_state(grid, cfg)

    def rhs(state: CompressibleMHDState) -> CompressibleMHDState:
        return compressible_mhd_rhs(state, params, lengths=grid.lengths)

    trajectory_raw = evolve_rk4(
        initial_state,
        rhs,
        dt=cfg.dt,
        steps=cfg.steps,
        save_every=cfg.save_every,
    )
    trajectory = CompressibleMHDTrajectory(
        times=trajectory_raw.times,
        states=trajectory_raw.states,
    )
    primitives = jax.vmap(lambda state: primitive_from_conservative(state, params))(
        trajectory.states
    )
    entropy_history = jax.vmap(lambda state: compressible_kelvin_helmholtz_dye_entropy(
        state,
        grid,
        params,
    ))(trajectory.states)
    return CompressibleKelvinHelmholtzRunResult(
        config=cfg,
        grid=grid,
        params=params,
        initial_state=initial_state,
        trajectory=trajectory,
        dye_entropy=entropy_history,
        density_min=jnp.min(primitives.density, axis=(1, 2)),
        pressure_min=jnp.min(primitives.pressure, axis=(1, 2)),
    )


def run_kelvin_helmholtz_validation(
    *,
    primary_config: KelvinHelmholtzConfig | None = None,
    comparison_config: KelvinHelmholtzConfig | None = None,
    compressible_config: CompressibleKelvinHelmholtzConfig | None = None,
    min_saved_samples: int = 5,
    min_entropy_gain: float = 1.0e-3,
    max_resolution_entropy_rdiff: float = 5.0e-3,
    max_dye_overshoot: float = 1.0e-2,
    min_compressible_density: float = 0.2,
    min_compressible_pressure: float = 0.2,
) -> KelvinHelmholtzValidationResult:
    """Run a gated, deterministic Kelvin--Helmholtz validation bundle.

    The default settings are intentionally validation-grade and CPU-friendly:
    they exercise nonlinear shear-layer roll-up diagnostics, resolution
    consistency, positivity checks for the smooth compressible tutorial, and IO
    schema generation.  Passing this gate does not make a shock-capturing or
    high-Reynolds-number production claim.
    """
    if min_saved_samples < 2:
        raise ValueError("min_saved_samples must be >= 2")
    if min_entropy_gain < 0.0:
        raise ValueError("min_entropy_gain must be non-negative")
    if max_resolution_entropy_rdiff < 0.0:
        raise ValueError("max_resolution_entropy_rdiff must be non-negative")
    if max_dye_overshoot < 0.0:
        raise ValueError("max_dye_overshoot must be non-negative")
    if min_compressible_density <= 0.0:
        raise ValueError("min_compressible_density must be positive")
    if min_compressible_pressure <= 0.0:
        raise ValueError("min_compressible_pressure must be positive")

    primary_cfg = (
        primary_config
        or KelvinHelmholtzConfig(shape=(32, 64), dt=2.0e-3, t_end=0.4, save_every=20)
    ).validated()
    comparison_cfg = (
        comparison_config
        or KelvinHelmholtzConfig(
            shape=(16, 32),
            lower=primary_cfg.lower,
            upper=primary_cfg.upper,
            viscosity=primary_cfg.viscosity,
            resistivity=primary_cfg.resistivity,
            dt=primary_cfg.dt,
            t_end=primary_cfg.t_end,
            save_every=primary_cfg.save_every,
            shear_width=primary_cfg.shear_width,
            perturbation_width=primary_cfg.perturbation_width,
            perturbation_amplitude=primary_cfg.perturbation_amplitude,
            flow_speed=primary_cfg.flow_speed,
            y1=primary_cfg.y1,
            y2=primary_cfg.y2,
        )
    ).validated()

    primary = run_kelvin_helmholtz_dye(primary_cfg)
    comparison = run_kelvin_helmholtz_dye(comparison_cfg)
    compressible = (
        run_compressible_kelvin_helmholtz(compressible_config)
        if compressible_config is not None
        else run_compressible_kelvin_helmholtz(
            CompressibleKelvinHelmholtzConfig(shape=(16, 32), t_end=0.02)
        )
    )

    primary_history = _incompressible_history_with_initial(primary)
    comparison_history = _incompressible_history_with_initial(comparison)
    compressible_history = _compressible_history_with_initial(compressible)

    entropy_gain = float(primary_history["entropy"][-1] - primary_history["entropy"][0])
    comparison_entropy_gain = float(
        comparison_history["entropy"][-1] - comparison_history["entropy"][0]
    )
    entropy_scale = max(abs(float(primary_history["entropy"][-1])), np.finfo(np.float64).tiny)
    resolution_entropy_rdiff = float(
        abs(primary_history["entropy"][-1] - comparison_history["entropy"][-1])
        / entropy_scale
    )
    dye_min = float(np.min(primary_history["dye"]))
    dye_max = float(np.max(primary_history["dye"]))
    dye_overshoot = max(0.0, -dye_min, dye_max - 1.0)
    primary_vorticity_linf_growth = float(
        np.max(np.abs(primary_history["omega"][-1])) - np.max(np.abs(primary_history["omega"][0]))
    )
    compressible_density_min = float(np.min(compressible_history["density_min"]))
    compressible_pressure_min = float(np.min(compressible_history["pressure_min"]))

    checks = {
        "finite_primary_histories": _finite_arrays(
            primary_history["time"],
            primary_history["dye"],
            primary_history["omega"],
            primary_history["entropy"],
        ),
        "finite_comparison_histories": _finite_arrays(
            comparison_history["time"],
            comparison_history["dye"],
            comparison_history["omega"],
            comparison_history["entropy"],
        ),
        "enough_saved_samples": bool(primary_history["time"].size >= min_saved_samples),
        "entropy_gain_observed": entropy_gain >= min_entropy_gain,
        "comparison_entropy_gain_observed": comparison_entropy_gain >= min_entropy_gain,
        "resolution_entropy_consistent": (
            resolution_entropy_rdiff <= max_resolution_entropy_rdiff
        ),
        "dye_bounds_controlled": dye_overshoot <= max_dye_overshoot,
        "finite_compressible_histories": _finite_arrays(
            compressible_history["time"],
            compressible_history["dye_entropy"],
            compressible_history["density_min"],
            compressible_history["pressure_min"],
        ),
        "compressible_density_positive": compressible_density_min >= min_compressible_density,
        "compressible_pressure_positive": compressible_pressure_min >= min_compressible_pressure,
    }
    diagnostics = {
        "schema": KELVIN_HELMHOLTZ_VALIDATION_SCHEMA,
        "primary_config": _kh_config_dict(primary_cfg),
        "comparison_config": _kh_config_dict(comparison_cfg),
        "compressible_config": _compressible_kh_config_dict(compressible.config),
        "primary_samples": int(primary_history["time"].size),
        "comparison_samples": int(comparison_history["time"].size),
        "primary_initial_entropy": float(primary_history["entropy"][0]),
        "primary_final_entropy": float(primary_history["entropy"][-1]),
        "primary_entropy_gain": entropy_gain,
        "comparison_initial_entropy": float(comparison_history["entropy"][0]),
        "comparison_final_entropy": float(comparison_history["entropy"][-1]),
        "comparison_entropy_gain": comparison_entropy_gain,
        "resolution_entropy_relative_difference": resolution_entropy_rdiff,
        "primary_dye_min": dye_min,
        "primary_dye_max": dye_max,
        "primary_dye_overshoot": dye_overshoot,
        "primary_vorticity_linf_growth": primary_vorticity_linf_growth,
        "compressible_min_density": compressible_density_min,
        "compressible_min_pressure": compressible_pressure_min,
        "claim_boundary": (
            "Validation artifact for smooth periodic incompressible/passive-dye "
            "and low-Mach compressible-MHD Kelvin--Helmholtz examples. It checks "
            "finite histories, entropy response, resolution consistency, dye "
            "bounds, and positivity, but it is not a shock-capturing production "
            "compressible-MHD benchmark."
        ),
        "references": {
            "lecoanet_2016": "https://doi.org/10.1093/mnras/stv2564",
        },
    }
    validation = {
        "schema": "mhx.validation.kelvin_helmholtz.gates.v1",
        "passed": all(checks.values()),
        "checks": checks,
        "thresholds": {
            "min_saved_samples": min_saved_samples,
            "min_entropy_gain": min_entropy_gain,
            "max_resolution_entropy_rdiff": max_resolution_entropy_rdiff,
            "max_dye_overshoot": max_dye_overshoot,
            "min_compressible_density": min_compressible_density,
            "min_compressible_pressure": min_compressible_pressure,
        },
        "diagnostics": diagnostics,
    }
    return KelvinHelmholtzValidationResult(
        primary=primary,
        comparison=comparison,
        compressible=compressible,
        diagnostics=diagnostics,
        validation=validation,
    )


def write_kelvin_helmholtz_validation(
    outdir: str | Path,
    *,
    movies: bool = False,
    **kwargs: Any,
) -> tuple[Path, dict[str, Any]]:
    """Write Kelvin--Helmholtz validation JSON, NPZ, figures, GIFs, and manifest."""
    output_dir = Path(outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    result = run_kelvin_helmholtz_validation(**kwargs)
    primary_history = _incompressible_history_with_initial(result.primary)
    comparison_history = _incompressible_history_with_initial(result.comparison)
    compressible_history = (
        _compressible_history_with_initial(result.compressible)
        if result.compressible is not None
        else None
    )

    diagnostics_path = output_dir / "diagnostics.json"
    validation_path = output_dir / "validation.json"
    primary_history_path = output_dir / "kelvin_helmholtz_incompressible.npz"
    comparison_history_path = output_dir / "kelvin_helmholtz_resolution_comparison.npz"
    diagnostics_path.write_text(
        json.dumps(result.diagnostics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    validation_path.write_text(
        json.dumps(result.validation, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    np.savez_compressed(
        primary_history_path,
        schema=KELVIN_HELMHOLTZ_VALIDATION_SCHEMA,
        time=primary_history["time"],
        dye=primary_history["dye"],
        omega=primary_history["omega"],
        entropy=primary_history["entropy"],
    )
    np.savez_compressed(
        comparison_history_path,
        schema=KELVIN_HELMHOLTZ_VALIDATION_SCHEMA,
        time=comparison_history["time"],
        entropy=comparison_history["entropy"],
    )

    entropy_path = _write_entropy_figure(
        primary_history,
        comparison_history,
        figure_dir / "kelvin_helmholtz_entropy.png",
    )
    snapshot_path = _write_incompressible_snapshot_figure(
        primary_history,
        figure_dir / "kelvin_helmholtz_snapshots.png",
    )
    outputs: dict[str, str] = {
        "diagnostics": diagnostics_path.name,
        "validation": validation_path.name,
        "incompressible_history": primary_history_path.name,
        "resolution_comparison": comparison_history_path.name,
        "entropy_figure": str(entropy_path.relative_to(output_dir)),
        "snapshot_figure": str(snapshot_path.relative_to(output_dir)),
    }
    if compressible_history is not None:
        compressible_history_path = output_dir / "kelvin_helmholtz_compressible_mhd.npz"
        np.savez_compressed(
            compressible_history_path,
            schema=KELVIN_HELMHOLTZ_VALIDATION_SCHEMA,
            time=compressible_history["time"],
            dye_entropy=compressible_history["dye_entropy"],
            density_min=compressible_history["density_min"],
            pressure_min=compressible_history["pressure_min"],
        )
        compressible_path = _write_compressible_minima_figure(
            compressible_history,
            figure_dir / "kelvin_helmholtz_compressible_minima.png",
        )
        outputs.update(
            {
                "compressible_history": compressible_history_path.name,
                "compressible_minima_figure": str(compressible_path.relative_to(output_dir)),
            }
        )
    if movies:
        dye_movie_path = _write_dye_movie(
            primary_history,
            figure_dir / "kelvin_helmholtz_dye.gif",
        )
        outputs["dye_movie"] = str(dye_movie_path.relative_to(output_dir))

    manifest_path = output_dir / "manifest.json"
    write_manifest(
        manifest_path,
        config=result.diagnostics,
        outputs=outputs,
        claim_level="validation",
        claim_scope=(
            "Standalone smooth Kelvin--Helmholtz validation example with "
            "resolution, entropy, positivity, and visualization gates."
        ),
    )
    return manifest_path, result.validation


def _kh_config_dict(config: KelvinHelmholtzConfig) -> dict[str, Any]:
    return {
        "shape": list(config.shape),
        "lower": list(config.lower),
        "upper": list(config.upper),
        "viscosity": config.viscosity,
        "resistivity": config.resistivity,
        "dt": config.dt,
        "t_end": config.t_end,
        "save_every": config.save_every,
        "shear_width": config.shear_width,
        "perturbation_width": config.perturbation_width,
        "perturbation_amplitude": config.perturbation_amplitude,
        "flow_speed": config.flow_speed,
        "y1": config.y1,
        "y2": config.y2,
    }


def _compressible_kh_config_dict(config: CompressibleKelvinHelmholtzConfig) -> dict[str, Any]:
    return {
        "shape": list(config.shape),
        "lower": list(config.lower),
        "upper": list(config.upper),
        "gamma": config.gamma,
        "dt": config.dt,
        "t_end": config.t_end,
        "save_every": config.save_every,
        "density": config.density,
        "pressure": config.pressure,
        "magnetic_field": list(config.magnetic_field),
        "shear_width": config.shear_width,
        "perturbation_width": config.perturbation_width,
        "perturbation_amplitude": config.perturbation_amplitude,
        "flow_speed": config.flow_speed,
        "y1": config.y1,
        "y2": config.y2,
    }


def _incompressible_history_with_initial(
    result: KelvinHelmholtzRunResult,
) -> dict[str, np.ndarray]:
    return {
        "time": np.concatenate(
            ([0.0], np.asarray(result.trajectory.times, dtype=np.float64))
        ),
        "dye": np.concatenate(
            (
                np.asarray(result.initial_state.dye, dtype=np.float64)[None, ...],
                np.asarray(result.trajectory.states.dye, dtype=np.float64),
            ),
            axis=0,
        ),
        "omega": np.concatenate(
            (
                np.asarray(result.initial_state.mhd.omega, dtype=np.float64)[None, ...],
                np.asarray(result.trajectory.states.mhd.omega, dtype=np.float64),
            ),
            axis=0,
        ),
        "entropy": np.concatenate(
            (
                [float(dye_entropy(result.initial_state.dye, result.grid))],
                np.asarray(result.entropy, dtype=np.float64),
            )
        ),
    }


def _compressible_history_with_initial(
    result: CompressibleKelvinHelmholtzRunResult,
) -> dict[str, np.ndarray]:
    initial_primitive = primitive_from_conservative(result.initial_state, result.params)
    initial_entropy = float(
        compressible_kelvin_helmholtz_dye_entropy(
            result.initial_state,
            result.grid,
            result.params,
        )
    )
    return {
        "time": np.concatenate(
            ([0.0], np.asarray(result.trajectory.times, dtype=np.float64))
        ),
        "dye_entropy": np.concatenate(
            ([initial_entropy], np.asarray(result.dye_entropy, dtype=np.float64))
        ),
        "density_min": np.concatenate(
            (
                [float(jnp.min(initial_primitive.density))],
                np.asarray(result.density_min, dtype=np.float64),
            )
        ),
        "pressure_min": np.concatenate(
            (
                [float(jnp.min(initial_primitive.pressure))],
                np.asarray(result.pressure_min, dtype=np.float64),
            )
        ),
    }


def _finite_arrays(*arrays: np.ndarray) -> bool:
    return all(np.isfinite(np.asarray(array)).all() for array in arrays)


def _write_entropy_figure(
    primary_history: dict[str, np.ndarray],
    comparison_history: dict[str, np.ndarray],
    path: Path,
) -> Path:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 4.0), constrained_layout=True)
    ax.plot(primary_history["time"], primary_history["entropy"], "o-", label="primary")
    ax.plot(
        comparison_history["time"],
        comparison_history["entropy"],
        "s--",
        label="resolution comparison",
    )
    ax.set_xlabel("time")
    ax.set_ylabel(r"$\int -c\log c\,dA$")
    ax.set_title("Kelvin--Helmholtz passive-dye entropy")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def _write_incompressible_snapshot_figure(
        history: dict[str, np.ndarray],
        path: Path,
) -> Path:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    frame_indices = [0, history["dye"].shape[0] // 2, history["dye"].shape[0] - 1]
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 7.0), constrained_layout=True)

    # Calculate physical aspect ratio assuming square grid cells (Ny / Nx)
    # history["dye"] shape is (frames, nx, ny)
    nx, ny = history["dye"].shape[1], history["dye"].shape[2]
    aspect = ny / nx
    extent = (0.0, 1.0, 0.0, 1.0)

    omega_max = max(float(np.max(np.abs(history["omega"]))), np.finfo(np.float64).eps)

    for column, frame_index in enumerate(frame_indices):
        dye_image = axes[0, column].imshow(
            history["dye"][frame_index].T,
            origin="lower",
            cmap="RdBu_r",
            vmin=0.0,
            vmax=1.0,
            extent=extent,
            aspect=aspect
        )
        axes[0, column].set_title(f"Dye Concentration (t={history['time'][frame_index]:.2f})")
        axes[0, column].set_xlabel("x / Lx")
        axes[0, column].set_ylabel("z / Lz")
        fig.colorbar(dye_image, ax=axes[0, column], shrink=0.72, label="Concentration (c)")

        omega_image = axes[1, column].imshow(
            history["omega"][frame_index].T,
            origin="lower",
            cmap="RdBu_r",
            vmin=-omega_max,
            vmax=omega_max,
            extent=extent,
            aspect=aspect
        )
        axes[1, column].set_title(f"Vorticity (t={history['time'][frame_index]:.2f})")
        axes[1, column].set_xlabel("x / Lx")
        axes[1, column].set_ylabel("z / Lz")
        fig.colorbar(omega_image, ax=axes[1, column], shrink=0.72, label="Vorticity")

    fig.suptitle("Smooth periodic Kelvin--Helmholtz validation snapshots")
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def _write_compressible_minima_figure(history: dict[str, np.ndarray], path: Path) -> Path:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), constrained_layout=True)
    axes[0].plot(history["time"], history["dye_entropy"], "o-")
    axes[0].set_title("Compressible dye entropy")
    axes[0].set_xlabel("time")
    axes[0].set_ylabel(r"$\int -c\log c\,dA$")
    axes[0].grid(True, alpha=0.25)
    axes[1].plot(history["time"], history["density_min"], "o-", label=r"$\min \rho$")
    axes[1].plot(history["time"], history["pressure_min"], "s--", label=r"$\min p$")
    axes[1].set_title("Low-Mach positivity checks")
    axes[1].set_xlabel("time")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(frameon=False)
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def _write_dye_movie(
    history: dict[str, np.ndarray],
    path: Path,
    *,
    max_frames: int = 36,
) -> Path:
    import imageio.v2 as imageio
    from matplotlib import colormaps

    path.parent.mkdir(parents=True, exist_ok=True)
    indices = _sample_indices(history["dye"].shape[0], max_frames)
    values = history["dye"][indices]
    vmin = 0.0 # Force pure white bounds
    vmax = 1.0 # Force pure red/blue bounds
    colormap = colormaps["RdBu_r"]
    frames = []
    for field in values:
        normalized = np.clip((field.T - vmin) / (vmax - vmin), 0.0, 1.0)
        frames.append((255.0 * colormap(normalized)[..., :3]).astype(np.uint8))
    imageio.mimsave(path, frames, duration=90, loop=0, palettesize=48)
    return path


def _sample_indices(frame_count: int, max_frames: int) -> np.ndarray:
    if frame_count <= max_frames:
        return np.arange(frame_count)
    return np.unique(np.linspace(0, frame_count - 1, max_frames, dtype=int))


# ---------------------------------------------------------------------------
# Kelvin--Helmholtz passive-dye benchmark, FAST incompressible run
#
# The runnable section below reuses the helpers above to execute the same
# smooth double-shear passive-dye benchmark (Lecoanet et al. style) in the
# reduced-MHD hydrodynamic limit (``psi=0``).  FAST defaults are chosen for
# a local laptop run; ``PRODUCTION_HINT`` documents production-scale
# settings but is not the default.
# ---------------------------------------------------------------------------


def _run_incompressible_default() -> None:
    import os

    import matplotlib.pyplot as plt

    from mhx.runtime import configure_jax

    configure_jax(enable_x64=True)

    output_root = Path(os.environ.get("MHX_EXAMPLE_OUTDIR_ROOT", "outputs/examples"))
    outdir = output_root / "kelvin_helmholtz_incompressible"
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"Writing outputs to {outdir.resolve()}")

    # Inputs: FAST defaults small enough to run locally.  For a larger
    # exploratory run, change ``shape``, ``t_end``, and ``save_every``; for a
    # production-quality hydrodynamic benchmark use much higher resolution
    # and document convergence separately.
    fast_config = KelvinHelmholtzConfig(
        shape=(32, 64),
        dt=2.0e-3,
        t_end=0.2,
        save_every=20,
        viscosity=1.0e-3,
        perturbation_amplitude=1.0e-2,
    )
    production_hint = KelvinHelmholtzConfig(  # noqa: F841 -- documented reference
        shape=(512, 1024),
        dt=1.0e-3,
        t_end=10.0,
        save_every=200,
        viscosity=1.0e-6,
        perturbation_amplitude=1.0e-2,
    )
    config = fast_config
    print(config)

    # Inspect the initial condition and RHS.  The example exposes the main
    # building blocks rather than hiding everything behind one driver call.
    grid = kelvin_helmholtz_grid(config)
    state0 = kelvin_helmholtz_initial_state_from_config(grid, config)
    warmup_result = run_kelvin_helmholtz_dye(config)
    rhs0 = kelvin_helmholtz_dye_rhs(
        state0,
        params=warmup_result.params,
        lengths=grid.lengths,
    )
    print("grid shape:", grid.shape)
    print("initial dye entropy:", float(dye_entropy(state0.dye, grid)))
    omega0 = np.asarray(state0.mhd.omega)
    print("initial omega range:", float(np.min(omega0)), float(np.max(omega0)))
    print("RHS dye max norm:", float(np.max(np.abs(np.asarray(rhs0.dye)))))

    # Run the simulation.
    result = run_kelvin_helmholtz_dye(config)
    result.trajectory.times.block_until_ready()
    print("saved times:", np.asarray(result.trajectory.times))
    print("entropy history:", np.asarray(result.entropy))
    final_dye_array = np.asarray(result.final_state.dye)
    print("final dye range:", float(np.min(final_dye_array)), float(np.max(final_dye_array)))

    # Plot outputs: initial/final dye, final vorticity, and entropy history.
    extent = (config.lower[0], config.upper[0], config.lower[1], config.upper[1])
    initial_dye = np.asarray(state0.dye)
    final_dye = np.asarray(result.final_state.dye)
    final_omega = np.asarray(result.final_state.mhd.omega)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    images = [
        axes[0].imshow(
            initial_dye.T, origin="lower", extent=extent, cmap="RdBu_r", vmin=0.0, vmax=1.0
        ),
        axes[1].imshow(
            final_dye.T, origin="lower", extent=extent, cmap="RdBu_r", vmin=0.0, vmax=1.0
        ),
    ]
    axes[0].set_title(f"Initial Dye Concentration, t={float(result.trajectory.times[0]):.3f}")
    axes[1].set_title(f"Final Dye Concentration, t={float(result.trajectory.times[-1]):.3f}")
    for ax, image in zip(axes[:2], images, strict=True):
        ax.set_xlabel("x/Lx")
        ax.set_ylabel("y/Ly")
        fig.colorbar(image, ax=ax, shrink=0.8, label="Concentration (c)")
    vort_im = axes[2].imshow(final_omega.T, origin="lower", extent=extent, cmap="magma")
    axes[2].set_title(f"Final Vorticity, t={float(result.trajectory.times[-1]):.3f}")
    axes[2].set_xlabel("x/Lx")
    axes[2].set_ylabel("y/Ly")
    fig.colorbar(vort_im, ax=axes[2], shrink=0.8, label="Vorticity")
    fig.savefig(outdir / "kh_incompressible_snapshots.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 3), constrained_layout=True)
    ax.plot(np.asarray(result.trajectory.times), np.asarray(result.entropy), marker="o")
    ax.set_xlabel("time")
    ax.set_ylabel("dye entropy")
    ax.set_title("Kelvin--Helmholtz dye entropy")
    ax.grid(alpha=0.3)
    fig.savefig(outdir / "kh_incompressible_entropy.png", dpi=180)
    plt.close(fig)

    print("wrote", outdir / "kh_incompressible_snapshots.png")
    print("wrote", outdir / "kh_incompressible_entropy.png")


if __name__ == "__main__":
    _run_incompressible_default()
