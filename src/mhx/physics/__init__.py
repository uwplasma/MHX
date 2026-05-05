"""Composable physics closures and source terms."""

from mhx.physics.equilibria import (
    CosineTearingEquilibrium,
    Equilibrium,
    EquilibriumMetadata,
    EquilibriumRegistry,
    ZeroEquilibrium,
    build_equilibrium,
    default_equilibrium_registry,
)
from mhx.physics.terms import (
    PHYSICS_API_VERSION,
    HyperResistivityTerm,
    PhysicsRegistry,
    PhysicsTerm,
    PhysicsTermMetadata,
    VorticityDragTerm,
    apply_physics_terms,
    build_physics_terms,
    default_physics_registry,
)

__all__ = [
    "CosineTearingEquilibrium",
    "Equilibrium",
    "EquilibriumMetadata",
    "EquilibriumRegistry",
    "PHYSICS_API_VERSION",
    "HyperResistivityTerm",
    "PhysicsRegistry",
    "PhysicsTerm",
    "PhysicsTermMetadata",
    "ZeroEquilibrium",
    "VorticityDragTerm",
    "build_equilibrium",
    "apply_physics_terms",
    "build_physics_terms",
    "default_equilibrium_registry",
    "default_physics_registry",
]
