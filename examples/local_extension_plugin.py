"""Minimal local physics and diagnostics plugin used by documentation examples."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import jax.numpy as jnp

from mhx.diagnostics import DiagnosticContext, DiagnosticSpec, DiagnosticsRegistry
from mhx.physics import PHYSICS_API_VERSION, PhysicsRegistry
from mhx.state import ReducedMHDParams, ReducedMHDState


@dataclass(frozen=True)
class FluxDriveTerm:
    r"""Small deterministic source ``A cos(x) cos(y)`` for extension demos."""

    amplitude: float = 0.0

    name: ClassVar[str] = "example_flux_drive"
    api_version: ClassVar[str] = PHYSICS_API_VERSION
    description: ClassVar[str] = "Adds a small cos(x)cos(y) flux drive for plugin demos."

    def rhs_addition(
        self,
        state: ReducedMHDState,
        params: ReducedMHDParams,
        *,
        lengths: tuple[float, float],
    ) -> ReducedMHDState:
        del params
        nx, ny = state.psi.shape
        x = jnp.arange(nx, dtype=state.psi.dtype) * lengths[0] / nx
        y = jnp.arange(ny, dtype=state.psi.dtype) * lengths[1] / ny
        drive = self.amplitude * jnp.cos(2.0 * jnp.pi * x[:, None] / lengths[0]) * jnp.cos(
            2.0 * jnp.pi * y[None, :] / lengths[1]
        )
        return ReducedMHDState(psi=drive, omega=jnp.zeros_like(state.omega))


def _flux_drive_factory(parameters) -> FluxDriveTerm:
    return FluxDriveTerm(amplitude=float(parameters.get("amplitude", 0.0)))


def _final_flux_l2(context: DiagnosticContext) -> dict[str, float]:
    psi = context.trajectory.states.psi[-1]
    return {"final_flux_l2": float(jnp.sqrt(jnp.mean(psi**2)))}


def register_physics(registry: PhysicsRegistry) -> None:
    """Register the example RHS term with an MHX physics registry."""
    registry.register("example_flux_drive", _flux_drive_factory)


def register_diagnostics(registry: DiagnosticsRegistry) -> None:
    """Register the example scalar diagnostic with an MHX diagnostics registry."""
    registry.register(
        DiagnosticSpec(
            name="final_flux_l2",
            description="Root-mean-square magnetic flux at the final saved frame.",
            output_keys=("final_flux_l2",),
            compute=_final_flux_l2,
        )
    )
