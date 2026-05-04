from __future__ import annotations

import math

from mhx.solver.tearing import sweet_parker_metrics


def test_sweet_parker_delta_scaling():
    eta1 = 1e-3
    eta2 = 4e-3
    sp1 = sweet_parker_metrics(B0=1.0, a=0.25, eta=eta1)
    sp2 = sweet_parker_metrics(B0=1.0, a=0.25, eta=eta2)

    ratio = sp2["delta_SP"] / sp1["delta_SP"]
    expected = math.sqrt(eta2 / eta1)
    assert abs(ratio / expected - 1.0) < 0.25
