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


def _zero_factory(parameters: Mapping[str, Any]) -> ZeroEquilibrium:
    del parameters
    return ZeroEquilibrium()


def _equilibrium_parameters(equilibrium: Equilibrium) -> dict[str, float]:
    if isinstance(equilibrium, CosineTearingEquilibrium):
        return {"perturbation_amplitude": equilibrium.perturbation_amplitude}
    return {}
