from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp

from mhx.diagnostics import default_diagnostics_registry
from mhx.physics import build_physics_terms, default_physics_registry
from mhx.state import ReducedMHDParams, ReducedMHDState


def test_plugin_template_registers_without_mhx_source_changes(monkeypatch) -> None:
    template_src = Path("examples/plugin_template/src").resolve()
    monkeypatch.syspath_prepend(str(template_src))

    from mhx_example_plugin.diagnostics import register_diagnostics
    from mhx_example_plugin.physics import register_physics

    physics_registry = default_physics_registry()
    register_physics(physics_registry)
    term = build_physics_terms(
        ("template_flux_sink",),
        {"template_flux_sink": {"rate": 0.25}},
        registry=physics_registry,
    )[0]
    state = ReducedMHDState(
        psi=jnp.ones((4, 4)),
        omega=jnp.zeros((4, 4)),
    )
    addition = term.rhs_addition(
        state,
        ReducedMHDParams(resistivity=0.0, viscosity=0.0),
        lengths=(1.0, 1.0),
    )
    assert float(jnp.mean(addition.psi)) == -0.25
    assert addition.omega.shape == state.omega.shape

    diagnostics_registry = default_diagnostics_registry()
    register_diagnostics(diagnostics_registry)
    metadata = diagnostics_registry.get("template_final_psi_mean")
    assert metadata.output_keys == ("template_final_psi_mean",)
