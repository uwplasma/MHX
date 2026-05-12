"""Configurable reduced-MHD equilibrium and initial-condition builders."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, ClassVar, Protocol

import jax.numpy as jnp

from mhx.grids import CartesianGrid
from mhx.state import ReducedMHDState


class Equilibrium(Protocol):
    """Stable protocol for reduced-MHD initial-condition builders."""

    name: ClassVar[str]
    description: ClassVar[str]

    def initial_state(self, grid: CartesianGrid) -> ReducedMHDState:
        """Return the initial reduced-MHD state on ``grid``."""


@dataclass(frozen=True)
class EquilibriumMetadata:
    """Serializable metadata for an equilibrium builder."""

    name: str
    description: str
    parameters: dict[str, float]


@dataclass(frozen=True)
class CosineTearingEquilibrium:
    r"""Periodic current-sheet-like equilibrium ``ψ₀=cos(y)+ε cos(x)cos(y)``."""

    perturbation_amplitude: float = 1.0e-3

    name: ClassVar[str] = "cosine_tearing"
    description: ClassVar[str] = "Periodic cosine current sheet with a small tearing perturbation."

    def initial_state(self, grid: CartesianGrid) -> ReducedMHDState:
        x, y = grid.mesh()
        length_x, length_y = grid.lengths
        psi_equilibrium = jnp.cos(2.0 * jnp.pi * y / length_y)
        perturbation = self.perturbation_amplitude * jnp.cos(
            2.0 * jnp.pi * x / length_x
        ) * jnp.cos(2.0 * jnp.pi * y / length_y)
        omega = jnp.zeros_like(psi_equilibrium)
        return ReducedMHDState(psi=psi_equilibrium + perturbation, omega=omega)


@dataclass(frozen=True)
class PeriodicDoubleHarrisEquilibrium:
    r"""Periodic double-Harris current sheet with an optional tearing seed.

    The reconnecting field is

    ``B_y = A[tanh((x-L_x/4)/a) - tanh((x-3L_x/4)/a) - 1]``.

    The corresponding flux is shifted to zero mean and remains periodic to
    exponentially small boundary error when ``a << L_x``.  This is the
    periodic spectral analogue of the Harris-sheet geometry used by the direct
    tearing eigenvalue benchmarks; it is intended for nonlinear growth gates and
    production-campaign initial data, not as a replacement for the finite-domain
    FKR/Harris eigenproblem.
    """

    width: float = 0.4
    amplitude: float = 1.0
    perturbation_amplitude: float = 0.0
    perturbation_mode: tuple[int, int] = (0, 1)

    name: ClassVar[str] = "periodic_double_harris"
    description: ClassVar[str] = "Periodic double-Harris current sheet with optional seed."

    def initial_state(self, grid: CartesianGrid) -> ReducedMHDState:
        if self.width <= 0.0:
            raise ValueError("width must be positive")
        if self.amplitude == 0.0:
            raise ValueError("amplitude must be nonzero")
        x, y = grid.mesh()
        length_x, length_y = grid.lengths
        sheet_left = 0.25 * length_x
        sheet_right = 0.75 * length_x
        flux = self.amplitude * self.width * (
            jnp.log(jnp.cosh((x - sheet_left) / self.width))
            - jnp.log(jnp.cosh((x - sheet_right) / self.width))
        ) - self.amplitude * x
        flux = flux - jnp.mean(flux)
        if self.perturbation_amplitude != 0.0:
            mode_x, mode_y = self.perturbation_mode
            perturbation = self.perturbation_amplitude * jnp.cos(
                2.0 * jnp.pi * mode_x * x / length_x
            ) * jnp.cos(2.0 * jnp.pi * mode_y * y / length_y)
            flux = flux + perturbation
        return ReducedMHDState(psi=flux, omega=jnp.zeros_like(flux))


@dataclass(frozen=True)
class ZeroEquilibrium:
    """Zero-field reduced-MHD initial condition for plugin/unit tests."""

    name: ClassVar[str] = "zero"
    description: ClassVar[str] = "Zero magnetic flux and zero vorticity."

    def initial_state(self, grid: CartesianGrid) -> ReducedMHDState:
        zeros = jnp.zeros(grid.shape)
        return ReducedMHDState(psi=zeros, omega=zeros)


EquilibriumFactory = Callable[[Mapping[str, Any]], Equilibrium]


class EquilibriumRegistry:
    """Registry for named reduced-MHD equilibrium builders."""

    def __init__(self) -> None:
        self._factories: dict[str, EquilibriumFactory] = {}

    def register(self, name: str, factory: EquilibriumFactory) -> None:
        """Register a factory under a stable equilibrium name."""
        if not name:
            raise ValueError("equilibrium name must be non-empty")
        self._factories[name] = factory

    def create(self, name: str, parameters: Mapping[str, Any] | None = None) -> Equilibrium:
        """Create an equilibrium from a registered factory."""
        if name not in self._factories:
            known = ", ".join(sorted(self._factories))
            raise KeyError(f"unknown equilibrium {name!r}; known equilibria: {known}")
        return self._factories[name](parameters or {})

    def names(self) -> tuple[str, ...]:
        """Return registered equilibrium names in deterministic order."""
        return tuple(sorted(self._factories))

    def metadata(self) -> tuple[EquilibriumMetadata, ...]:
        """Return metadata for all registered equilibria."""
        items = []
        for name in self.names():
            equilibrium = self.create(name)
            items.append(
                EquilibriumMetadata(
                    name=name,
                    description=equilibrium.description,
                    parameters=_equilibrium_parameters(equilibrium),
                )
            )
        return tuple(items)


def default_equilibrium_registry() -> EquilibriumRegistry:
    """Return the built-in reduced-MHD equilibrium registry."""
    registry = EquilibriumRegistry()
    registry.register("cosine_tearing", _cosine_tearing_factory)
    registry.register("periodic_double_harris", _periodic_double_harris_factory)
    registry.register("zero", _zero_factory)
    return registry


def build_equilibrium(
    name: str,
    parameters: Mapping[str, Any] | None = None,
    *,
    registry: EquilibriumRegistry | None = None,
) -> Equilibrium:
    """Build a configured equilibrium by name."""
    active_registry = registry or default_equilibrium_registry()
    return active_registry.create(name, parameters)


def _cosine_tearing_factory(parameters: Mapping[str, Any]) -> CosineTearingEquilibrium:
    return CosineTearingEquilibrium(
        perturbation_amplitude=float(parameters.get("perturbation_amplitude", 1.0e-3))
    )


def _periodic_double_harris_factory(
    parameters: Mapping[str, Any],
) -> PeriodicDoubleHarrisEquilibrium:
    mode = parameters.get("perturbation_mode", (0, 1))
    return PeriodicDoubleHarrisEquilibrium(
        width=float(parameters.get("width", 0.4)),
        amplitude=float(parameters.get("amplitude", 1.0)),
        perturbation_amplitude=float(parameters.get("perturbation_amplitude", 0.0)),
        perturbation_mode=tuple(int(item) for item in mode),
    )


def _zero_factory(parameters: Mapping[str, Any]) -> ZeroEquilibrium:
    del parameters
    return ZeroEquilibrium()


def _equilibrium_parameters(equilibrium: Equilibrium) -> dict[str, float]:
    if isinstance(equilibrium, CosineTearingEquilibrium):
        return {"perturbation_amplitude": equilibrium.perturbation_amplitude}
    if isinstance(equilibrium, PeriodicDoubleHarrisEquilibrium):
        return {
            "width": equilibrium.width,
            "amplitude": equilibrium.amplitude,
            "perturbation_amplitude": equilibrium.perturbation_amplitude,
            "perturbation_mode_x": float(equilibrium.perturbation_mode[0]),
            "perturbation_mode_y": float(equilibrium.perturbation_mode[1]),
        }
    return {}
