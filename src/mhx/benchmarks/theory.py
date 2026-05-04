"""Analytic scaling estimates used by benchmark reports."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class FKRConstantPsiEstimate:
    """Constant-psi tearing scaling estimate for a Harris-like sheet."""

    lundquist: float
    ka: float
    delta_prime_a: float
    gamma_tau_a: float
    inner_width_a: float

    def to_dict(self) -> dict[str, float]:
        """Return JSON-compatible values."""
        return asdict(self)


@dataclass(frozen=True)
class PlasmoidScalingEstimate:
    """Loureiro-type Sweet-Parker plasmoid scaling estimate."""

    lundquist: float
    gamma_tau_a: float
    fastest_mode_k_l: float

    def to_dict(self) -> dict[str, float]:
        """Return JSON-compatible values."""
        return asdict(self)


def harris_sheet_delta_prime(ka: float) -> float:
    r"""Return the Harris-sheet outer-region proxy ``Δ' a = 2(1/(ka) - ka)``."""
    if ka <= 0.0:
        raise ValueError("ka must be positive")
    return 2.0 * (1.0 / ka - ka)


def fkr_constant_psi_estimate(lundquist: float, ka: float) -> FKRConstantPsiEstimate:
    r"""Return a dimensionless FKR constant-psi tearing scaling estimate.

    The estimate intentionally omits order-unity coefficients:

    ``γ τ_a ~ S_a^(-3/5) (ka)^(2/5) (Δ'a)^(4/5)``.
    """
    if lundquist <= 0.0:
        raise ValueError("lundquist must be positive")
    delta_prime_a = harris_sheet_delta_prime(ka)
    if delta_prime_a <= 0.0:
        raise ValueError("constant-psi tearing estimate requires positive delta_prime_a")
    gamma_tau_a = (lundquist ** (-3.0 / 5.0)) * (ka ** (2.0 / 5.0)) * (
        delta_prime_a ** (4.0 / 5.0)
    )
    inner_width_a = (lundquist ** (-2.0 / 5.0)) * (ka ** (-2.0 / 5.0)) * (
        delta_prime_a ** (1.0 / 5.0)
    )
    return FKRConstantPsiEstimate(
        lundquist=lundquist,
        ka=ka,
        delta_prime_a=delta_prime_a,
        gamma_tau_a=gamma_tau_a,
        inner_width_a=inner_width_a,
    )


def loureiro_plasmoid_estimate(lundquist: float) -> PlasmoidScalingEstimate:
    r"""Return Sweet-Parker plasmoid estimates ``γτ_A ~ S^(1/4)``, ``kL ~ S^(3/8)``."""
    if lundquist <= 0.0:
        raise ValueError("lundquist must be positive")
    return PlasmoidScalingEstimate(
        lundquist=lundquist,
        gamma_tau_a=lundquist ** (1.0 / 4.0),
        fastest_mode_k_l=lundquist ** (3.0 / 8.0),
    )


def ideal_tearing_aspect_ratio(lundquist: float) -> float:
    r"""Return Pucci-Velli ideal-tearing scaling ``a/L ~ S^(-1/3)``."""
    if lundquist <= 0.0:
        raise ValueError("lundquist must be positive")
    return lundquist ** (-1.0 / 3.0)

