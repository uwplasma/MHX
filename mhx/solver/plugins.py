from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Dict, List, Tuple, Callable
import inspect

import jax.numpy as jnp

from mhx.version import PHYSICS_API_VERSION

Array = jnp.ndarray

REQUIRED_TERM_KWARGS = {"t", "v_hat", "B_hat", "kx", "ky", "kz", "k2", "mask_dealias"}

@dataclass(frozen=True)
class PhysicsAPI:
    version: str
    term_signature: str
    notes: str = ""


PHYSICS_API = PhysicsAPI(
    version=PHYSICS_API_VERSION,
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


def list_factories() -> List[str]:
    return sorted(_FACTORIES.keys())


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


def validate_term(term: PhysicsTerm) -> List[str]:
    errors: List[str] = []
    if not getattr(term, "name", None):
        errors.append("missing name attribute")
    if not getattr(term, "api_version", None):
        errors.append("missing api_version attribute")
    if getattr(term, "api_version", None) not in SUPPORTED_API:
        errors.append(f"unsupported api_version {getattr(term, 'api_version', None)}")
    fn = getattr(term, "rhs_additions", None)
    if fn is None:
        errors.append("missing rhs_additions method")
        return errors
    sig = inspect.signature(fn)
    params = {p.name for p in sig.parameters.values() if p.name != "self"}
    missing = REQUIRED_TERM_KWARGS - params
    if missing:
        errors.append(f"rhs_additions missing kwargs: {sorted(missing)}")
    return errors


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


@dataclass(frozen=True)
class HallTerm:
    """Hall-like term: dB ~ -d_h * k^2 * curl(B) in spectral form."""

    d_h: float
    name: str = "hall"
    api_version: str = API_VERSION

    def rhs_additions(self, *, t: float, v_hat: Array, B_hat: Array, kx: Array, ky: Array, kz: Array, k2: Array, mask_dealias: Array) -> Tuple[Array, Array]:
        _ = (t, v_hat, mask_dealias)
        Bx_hat, By_hat, Bz_hat = B_hat[0], B_hat[1], B_hat[2]
        Jx_hat = 1j * (ky * Bz_hat - kz * By_hat)
        Jy_hat = 1j * (kz * Bx_hat - kx * Bz_hat)
        Jz_hat = 1j * (kx * By_hat - ky * Bx_hat)
        J_hat = jnp.stack([Jx_hat, Jy_hat, Jz_hat], axis=0)
        dB = -self.d_h * k2 * J_hat
        dv = jnp.zeros_like(v_hat)
        return dv, dB


@dataclass(frozen=True)
class AnisotropicPressureTerm:
    """Toy anisotropic pressure: damp parallel velocity via -chi * k_parallel^2 * v_hat."""

    chi: float
    name: str = "anisotropic_pressure"
    api_version: str = API_VERSION

    def rhs_additions(self, *, t: float, v_hat: Array, B_hat: Array, kx: Array, ky: Array, kz: Array, k2: Array, mask_dealias: Array) -> Tuple[Array, Array]:
        _ = (t, B_hat, kx, ky, k2, mask_dealias)
        kpar2 = kz * kz
        dv = -self.chi * kpar2 * v_hat
        dB = jnp.zeros_like(B_hat)
        return dv, dB


register_factory("hall", lambda d_h=1e-2: HallTerm(d_h=d_h))
register_factory("hall_toy", lambda d_h=1e-2: HallTerm(d_h=d_h))
register_factory("anisotropic_pressure", lambda chi=1e-2: AnisotropicPressureTerm(chi=chi))


@dataclass(frozen=True)
class ElectronPressureTensorTerm:
    """Toy electron pressure tensor term: proxy via -pe_coef * k^2 * J_hat."""

    pe_coef: float
    name: str = "electron_pressure_tensor"
    api_version: str = API_VERSION

    def rhs_additions(self, *, t: float, v_hat: Array, B_hat: Array, kx: Array, ky: Array, kz: Array, k2: Array, mask_dealias: Array) -> Tuple[Array, Array]:
        _ = (t, v_hat, mask_dealias)
        Bx_hat, By_hat, Bz_hat = B_hat[0], B_hat[1], B_hat[2]
        Jx_hat = 1j * (ky * Bz_hat - kz * By_hat)
        Jy_hat = 1j * (kz * Bx_hat - kx * Bz_hat)
        Jz_hat = 1j * (kx * By_hat - ky * Bx_hat)
        J_hat = jnp.stack([Jx_hat, Jy_hat, Jz_hat], axis=0)
        dB = -self.pe_coef * k2 * J_hat
        dv = jnp.zeros_like(v_hat)
        return dv, dB


@dataclass(frozen=True)
class TwoFluidOhmTerm:
    """Toy two-fluid Ohm's law: Hall + electron pressure proxy."""

    d_h: float
    pe_coef: float
    name: str = "two_fluid_ohm"
    api_version: str = API_VERSION

    def rhs_additions(self, *, t: float, v_hat: Array, B_hat: Array, kx: Array, ky: Array, kz: Array, k2: Array, mask_dealias: Array) -> Tuple[Array, Array]:
        _ = (t, v_hat, mask_dealias)
        Bx_hat, By_hat, Bz_hat = B_hat[0], B_hat[1], B_hat[2]
        Jx_hat = 1j * (ky * Bz_hat - kz * By_hat)
        Jy_hat = 1j * (kz * Bx_hat - kx * Bz_hat)
        Jz_hat = 1j * (kx * By_hat - ky * Bx_hat)
        J_hat = jnp.stack([Jx_hat, Jy_hat, Jz_hat], axis=0)
        dB_hall = -self.d_h * k2 * J_hat
        dB_pe = -self.pe_coef * k2 * J_hat
        dB = dB_hall + dB_pe
        dv = jnp.zeros_like(v_hat)
        return dv, dB


register_factory("electron_pressure_tensor", lambda pe_coef=1e-2: ElectronPressureTensorTerm(pe_coef=pe_coef))
register_factory("two_fluid_ohm", lambda d_h=1e-2, pe_coef=1e-2: TwoFluidOhmTerm(d_h=d_h, pe_coef=pe_coef))
