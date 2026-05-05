"""Composable physics closures and source terms."""

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
    "PHYSICS_API_VERSION",
    "HyperResistivityTerm",
    "PhysicsRegistry",
    "PhysicsTerm",
    "PhysicsTermMetadata",
    "VorticityDragTerm",
    "apply_physics_terms",
    "build_physics_terms",
    "default_physics_registry",
]
