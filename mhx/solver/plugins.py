from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Dict, List, Tuple

import jax.numpy as jnp

Array = jnp.ndarray


class PhysicsTerm(Protocol):
    """Physics plugin interface.

    Implementations must return additive contributions to (dv_hat, dB_hat).
    """

    name: str

    def rhs_additions(
        self,
        *,
        t: float,
        v_hat: Array,
        B_hat: Array,
        kx: Array,
        ky: Array,
        kz: Array,
        k2: Array,
        mask_dealias: Array,
    ) -> Tuple[Array, Array]:
        ...


_REGISTRY: Dict[str, PhysicsTerm] = {}


def register_term(term: PhysicsTerm) -> None:
    _REGISTRY[term.name] = term


def get_term(name: str) -> PhysicsTerm:
    return _REGISTRY[name]


def list_terms() -> List[str]:
    return sorted(_REGISTRY.keys())


def apply_terms(
    terms: List[PhysicsTerm],
    *,
    t: float,
    v_hat: Array,
    B_hat: Array,
    kx: Array,
    ky: Array,
    kz: Array,
    k2: Array,
    mask_dealias: Array,
) -> Tuple[Array, Array]:
    dv = jnp.zeros_like(v_hat)
    dB = jnp.zeros_like(B_hat)
    for term in terms:
        dv_add, dB_add = term.rhs_additions(
            t=t,
            v_hat=v_hat,
            B_hat=B_hat,
            kx=kx,
            ky=ky,
            kz=kz,
            k2=k2,
            mask_dealias=mask_dealias,
        )
        dv = dv + dv_add
        dB = dB + dB_add
    return dv, dB


@dataclass(frozen=True)
class HyperResistivityTerm:
    """Adds -eta4 * k^4 * B_hat to induction equation."""

    eta4: float
    name: str = "hyper_resistivity"

    def rhs_additions(self, *, t: float, v_hat: Array, B_hat: Array, kx: Array, ky: Array, kz: Array, k2: Array, mask_dealias: Array) -> Tuple[Array, Array]:
        _ = t
        k4 = k2 * k2
        dB = -self.eta4 * k4 * B_hat
        dv = jnp.zeros_like(v_hat)
        return dv, dB


@dataclass(frozen=True)
class LinearDragTerm:
    """Adds -mu * v_hat to momentum equation."""

    mu: float
    name: str = "linear_drag"

    def rhs_additions(self, *, t: float, v_hat: Array, B_hat: Array, kx: Array, ky: Array, kz: Array, k2: Array, mask_dealias: Array) -> Tuple[Array, Array]:
        _ = (t, B_hat, kx, ky, kz, k2, mask_dealias)
        dv = -self.mu * v_hat
        dB = jnp.zeros_like(B_hat)
        return dv, dB
