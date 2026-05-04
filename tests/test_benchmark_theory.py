from __future__ import annotations

import pytest

from mhx.benchmarks import (
    fkr_constant_psi_estimate,
    harris_sheet_delta_prime,
    ideal_tearing_aspect_ratio,
    loureiro_plasmoid_estimate,
)


def test_harris_delta_prime_sign_and_validation() -> None:
    assert harris_sheet_delta_prime(0.5) == pytest.approx(3.0)
    assert harris_sheet_delta_prime(1.0) == pytest.approx(0.0)
    with pytest.raises(ValueError, match="ka"):
        harris_sheet_delta_prime(0.0)


def test_fkr_constant_psi_scaling() -> None:
    estimate = fkr_constant_psi_estimate(lundquist=1.0e6, ka=0.5)
    higher_s = fkr_constant_psi_estimate(lundquist=1.0e7, ka=0.5)
    assert estimate.delta_prime_a == pytest.approx(3.0)
    assert higher_s.gamma_tau_a < estimate.gamma_tau_a
    assert estimate.to_dict()["ka"] == pytest.approx(0.5)
    with pytest.raises(ValueError, match="lundquist"):
        fkr_constant_psi_estimate(lundquist=0.0, ka=0.5)
    with pytest.raises(ValueError, match="delta_prime"):
        fkr_constant_psi_estimate(lundquist=1.0e6, ka=1.1)


def test_plasmoid_and_ideal_tearing_scalings() -> None:
    plasmoid = loureiro_plasmoid_estimate(1.0e4)
    assert plasmoid.gamma_tau_a == pytest.approx(10.0)
    assert plasmoid.fastest_mode_k_l == pytest.approx(1.0e4 ** (3.0 / 8.0))
    assert ideal_tearing_aspect_ratio(1.0e6) == pytest.approx(0.01)
    with pytest.raises(ValueError, match="lundquist"):
        loureiro_plasmoid_estimate(0.0)
    with pytest.raises(ValueError, match="lundquist"):
        ideal_tearing_aspect_ratio(0.0)
