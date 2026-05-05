"""Example third-party physics term for MHX."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import jax.numpy as jnp

from mhx.physics import PHYSICS_API_VERSION, PhysicsRegistry
from mhx.state import ReducedMHDParams, ReducedMHDState


@dataclass(frozen=True)
class TemplateFluxSink:
    r"""Linear flux sink ``\partial_t\psi \leftarrow -\alpha\psi``."""

    rate: float = 0.0

    name: ClassVar[str] = "template_flux_sink"
    api_version: ClassVar[str] = PHYSICS_API_VERSION
    description: ClassVar[str] = "Example third-party linear magnetic-flux sink."

    def rhs_addition(
        self,
        state: ReducedMHDState,
        params: ReducedMHDParams,
        *,
        lengths: tuple[float, float],
    ) -> ReducedMHDState:
        del params, lengths
        return ReducedMHDState(
            psi=-self.rate * state.psi,
            omega=jnp.zeros_like(state.omega),
        )


def _factory(parameters: dict[str, Any]) -> TemplateFluxSink:
    return TemplateFluxSink(rate=float(parameters.get("rate", 0.0)))


def register_physics(registry: PhysicsRegistry) -> None:
    """Register this package's physics terms with MHX."""
    registry.register("template_flux_sink", _factory)
