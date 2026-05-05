"""Versioned physics-term plugin API."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, ClassVar, Protocol

import jax.numpy as jnp

from mhx.numerics.spectral import laplacian
from mhx.state import ReducedMHDParams, ReducedMHDState

PHYSICS_API_VERSION = "mhx.physics.v1"


class PhysicsTerm(Protocol):
    """Stable v1 protocol for reduced-MHD RHS extension terms."""

    name: ClassVar[str]
    api_version: ClassVar[str]
    description: ClassVar[str]

    def rhs_addition(
        self,
        state: ReducedMHDState,
        params: ReducedMHDParams,
        *,
        lengths: tuple[float, float],
    ) -> ReducedMHDState:
        """Return additive RHS contributions for ``psi`` and ``omega``."""


@dataclass(frozen=True)
class PhysicsTermMetadata:
    """Serializable metadata for a registered physics term."""

    name: str
    api_version: str
    description: str
    parameters: dict[str, float]


@dataclass(frozen=True)
class HyperResistivityTerm:
    r"""Fourth-order diffusion term ``-η₄∇⁴ψ`` and ``-ν₄∇⁴ω``."""

    eta4: float = 0.0
    nu4: float = 0.0

    name: ClassVar[str] = "hyper_resistivity"
    api_version: ClassVar[str] = PHYSICS_API_VERSION
    description: ClassVar[str] = "Adds fourth-order hyper-resistive/hyper-viscous damping."

    def rhs_addition(
        self,
        state: ReducedMHDState,
        params: ReducedMHDParams,
        *,
        lengths: tuple[float, float],
    ) -> ReducedMHDState:
        del params
        biharmonic_psi = laplacian(laplacian(state.psi, lengths=lengths), lengths=lengths)
        biharmonic_omega = laplacian(laplacian(state.omega, lengths=lengths), lengths=lengths)
        return ReducedMHDState(
            psi=-self.eta4 * biharmonic_psi,
            omega=-self.nu4 * biharmonic_omega,
        )


@dataclass(frozen=True)
class VorticityDragTerm:
    """Linear vorticity drag ``-αω``."""

    rate: float = 0.0

    name: ClassVar[str] = "vorticity_drag"
    api_version: ClassVar[str] = PHYSICS_API_VERSION
    description: ClassVar[str] = "Adds linear damping to the vorticity equation."

    def rhs_addition(
        self,
        state: ReducedMHDState,
        params: ReducedMHDParams,
        *,
        lengths: tuple[float, float],
    ) -> ReducedMHDState:
        del params, lengths
        return ReducedMHDState(
            psi=jnp.zeros_like(state.psi),
            omega=-self.rate * state.omega,
        )


TermFactory = Callable[[Mapping[str, Any]], PhysicsTerm]


class PhysicsRegistry:
    """Registry for versioned physics-term factories."""

    def __init__(self) -> None:
        self._factories: dict[str, TermFactory] = {}

    def register(self, name: str, factory: TermFactory) -> None:
        """Register a factory under a stable term name."""
        if not name:
            raise ValueError("physics term name must be non-empty")
        self._factories[name] = factory

    def create(self, name: str, parameters: Mapping[str, Any] | None = None) -> PhysicsTerm:
        """Create a term from a registered factory."""
        if name not in self._factories:
            known = ", ".join(sorted(self._factories))
            raise KeyError(f"unknown physics term {name!r}; known terms: {known}")
        return self._factories[name](parameters or {})

    def names(self) -> tuple[str, ...]:
        """Return registered names in deterministic order."""
        return tuple(sorted(self._factories))

    def metadata(self) -> tuple[PhysicsTermMetadata, ...]:
        """Return metadata for all registered terms."""
        items = []
        for name in self.names():
            term = self.create(name)
            items.append(
                PhysicsTermMetadata(
                    name=name,
                    api_version=term.api_version,
                    description=term.description,
                    parameters=_term_parameters(term),
                )
            )
        return tuple(items)


def default_physics_registry() -> PhysicsRegistry:
    """Return the built-in physics-term registry."""
    registry = PhysicsRegistry()
    registry.register("hyper_resistivity", _hyper_resistivity_factory)
    registry.register("vorticity_drag", _vorticity_drag_factory)
    return registry


def build_physics_terms(
    names: tuple[str, ...],
    term_parameters: Mapping[str, Mapping[str, Any]],
    *,
    registry: PhysicsRegistry | None = None,
) -> tuple[PhysicsTerm, ...]:
    """Build configured terms from names and per-term parameter mappings."""
    active_registry = registry or default_physics_registry()
    return tuple(active_registry.create(name, term_parameters.get(name, {})) for name in names)


def apply_physics_terms(
    base_rhs: ReducedMHDState,
    terms: tuple[PhysicsTerm, ...],
    state: ReducedMHDState,
    params: ReducedMHDParams,
    *,
    lengths: tuple[float, float],
) -> ReducedMHDState:
    """Add all configured physics-term RHS contributions to a base RHS."""
    psi = base_rhs.psi
    omega = base_rhs.omega
    for term in terms:
        addition = term.rhs_addition(state, params, lengths=lengths)
        psi = psi + addition.psi
        omega = omega + addition.omega
    return ReducedMHDState(psi=psi, omega=omega)


def _hyper_resistivity_factory(parameters: Mapping[str, Any]) -> HyperResistivityTerm:
    return HyperResistivityTerm(
        eta4=float(parameters.get("eta4", 0.0)),
        nu4=float(parameters.get("nu4", 0.0)),
    )


def _vorticity_drag_factory(parameters: Mapping[str, Any]) -> VorticityDragTerm:
    return VorticityDragTerm(rate=float(parameters.get("rate", 0.0)))


def _term_parameters(term: PhysicsTerm) -> dict[str, float]:
    if isinstance(term, HyperResistivityTerm):
        return {"eta4": term.eta4, "nu4": term.nu4}
    if isinstance(term, VorticityDragTerm):
        return {"rate": term.rate}
    return {}

