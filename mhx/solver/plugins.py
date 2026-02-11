from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Dict, List, Tuple, Callable

import jax.numpy as jnp

Array = jnp.ndarray

@dataclass(frozen=True)
class PhysicsAPI:
    version: str
    term_signature: str
    notes: str = ""


PHYSICS_API = PhysicsAPI(
    version="1",
    term_signature="rhs_additions(*, t, v_hat, B_hat, kx, ky, kz, k2, mask_dealias) -> (dv_hat, dB_hat)",
    notes="Return additive spectral RHS contributions; inputs are dealised and div-free.",
)

API_VERSION = PHYSICS_API.version
SUPPORTED_API: Dict[str, PhysicsAPI] = {PHYSICS_API.version: PHYSICS_API}


class PhysicsTerm(Protocol):
    """Physics plugin interface.

    Implementations must return additive contributions to (dv_hat, dB_hat).
    """

    name: str
    api_version: str

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
_FACTORIES: Dict[str, Callable[..., PhysicsTerm]] = {}


def register_term(term: PhysicsTerm) -> None:
    api_version = getattr(term, "api_version", None)
    if api_version not in SUPPORTED_API:
        raise ValueError(
            f"PhysicsTerm {term.name} has incompatible api_version {api_version}. "
            f"Supported: {sorted(SUPPORTED_API.keys())}"
        )
    _REGISTRY[term.name] = term


def register_factory(name: str, factory: Callable[..., PhysicsTerm]) -> None:
    _FACTORIES[name] = factory


def get_term(name: str) -> PhysicsTerm:
    return _REGISTRY[name]


def list_terms() -> List[str]:
    return sorted(_REGISTRY.keys())


def build_terms(names: List[str], params: Dict[str, Dict[str, float]] | None = None) -> List[PhysicsTerm]:
    params = params or {}
    terms = []
    for name in names:
        if name in _FACTORIES:
            term = _FACTORIES[name](**params.get(name, {}))
            terms.append(term)
        elif name in _REGISTRY:
            terms.append(_REGISTRY[name])
        else:
            raise KeyError(f"Unknown physics term: {name}")
    return terms


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
    api_version: str = API_VERSION

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
    api_version: str = API_VERSION

    def rhs_additions(self, *, t: float, v_hat: Array, B_hat: Array, kx: Array, ky: Array, kz: Array, k2: Array, mask_dealias: Array) -> Tuple[Array, Array]:
        _ = (t, B_hat, kx, ky, kz, k2, mask_dealias)
        dv = -self.mu * v_hat
        dB = jnp.zeros_like(B_hat)
        return dv, dB


register_factory("hyper_resistivity", lambda eta4=1e-4: HyperResistivityTerm(eta4=eta4))
register_factory("linear_drag", lambda mu=0.1: LinearDragTerm(mu=mu))
