from __future__ import annotations

import numpy as np

from mhx.solver.diagnostics import estimate_growth_rate, plasmoid_complexity_metric


def test_gamma_fit_linear_growth_window():
    # Synthetic exponential growth
    ts = np.linspace(0.0, 1.0, 50)
    gamma = 2.5
    A0 = 1e-6
    amps = A0 * np.exp(gamma * ts)
    gamma_fit, lnA_fit, mask = estimate_growth_rate(ts, amps, w0=amps[0])
    assert np.isfinite(gamma_fit)
    assert gamma_fit > 0
    # Should be close to true gamma in synthetic case
    assert abs(gamma_fit - gamma) / gamma < 0.3


def test_plasmoid_complexity_trivial_profile():
    # Constant signal should have minimal complexity
    y = np.ones(128)
    c = plasmoid_complexity_metric(y)
    assert np.isfinite(c)
    assert c >= 0
