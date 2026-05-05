"""Example third-party diagnostic for MHX."""

from __future__ import annotations

import jax.numpy as jnp

from mhx.diagnostics import DiagnosticContext, DiagnosticSpec, DiagnosticsRegistry


def _final_psi_mean(context: DiagnosticContext) -> dict[str, float]:
    return {"template_final_psi_mean": float(jnp.mean(context.trajectory.states.psi[-1]))}


def register_diagnostics(registry: DiagnosticsRegistry) -> None:
    """Register this package's diagnostics with MHX."""
    registry.register(
        DiagnosticSpec(
            name="template_final_psi_mean",
            description="Example third-party final magnetic-flux mean diagnostic.",
            output_keys=("template_final_psi_mean",),
            compute=_final_psi_mean,
        )
    )
