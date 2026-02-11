from __future__ import annotations

import jax.numpy as jnp

from mhx.solver.diagnostics import estimate_growth_rate


def test_estimate_growth_rate_exponential():
    ts = jnp.linspace(0.0, 1.0, 50)
    gamma_true = 2.0
    amp = jnp.exp(gamma_true * ts)

    gamma_fit, _, _ = estimate_growth_rate(
        ts,
        amp,
        w0=amp[0],
        lower_factor=1.0,
        upper_frac_of_max=1.0,
        min_points=5,
    )

    assert abs(float(gamma_fit) - gamma_true) < 0.2
