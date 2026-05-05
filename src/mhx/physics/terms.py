"""Versioned physics-term plugin API."""

from __future__ import annotations

import importlib
import importlib.util
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Protocol

import jax.numpy as jnp

from mhx.numerics.spectral import fft_derivative, gradient, laplacian
from mhx.state import ReducedMHDParams, ReducedMHDState

PHYSICS_API_VERSION = "mhx.physics.v1"


class PhysicsTerm(Protocol):
    """Stable v1 protocol for reduced-MHD RHS extension terms."""

    name: ClassVar[str]
    api_version: ClassVar[str]
    description: ClassVar[str]

    def rhs_addition(
        self,
        state: ReducedMHDState,
        params: ReducedMHDParams,
        *,
        lengths: tuple[float, float],
    ) -> ReducedMHDState:
        """Return additive RHS contributions for ``psi`` and ``omega``."""


@dataclass(frozen=True)
class PhysicsTermMetadata:
    """Serializable metadata for a registered physics term."""

    name: str
    api_version: str
    description: str
    parameters: dict[str, float]


@dataclass(frozen=True)
class HyperResistivityTerm:
    r"""Fourth-order diffusion term ``-η₄∇⁴ψ`` and ``-ν₄∇⁴ω``."""

    eta4: float = 0.0
    nu4: float = 0.0

    name: ClassVar[str] = "hyper_resistivity"
    api_version: ClassVar[str] = PHYSICS_API_VERSION
    description: ClassVar[str] = "Adds fourth-order hyper-resistive/hyper-viscous damping."

    def rhs_addition(
        self,
        state: ReducedMHDState,
        params: ReducedMHDParams,
        *,
        lengths: tuple[float, float],
    ) -> ReducedMHDState:
        del params
        biharmonic_psi = laplacian(laplacian(state.psi, lengths=lengths), lengths=lengths)
        biharmonic_omega = laplacian(laplacian(state.omega, lengths=lengths), lengths=lengths)
        return ReducedMHDState(
            psi=-self.eta4 * biharmonic_psi,
            omega=-self.nu4 * biharmonic_omega,
        )


@dataclass(frozen=True)
class VorticityDragTerm:
    """Linear vorticity drag ``-αω``."""

    rate: float = 0.0

    name: ClassVar[str] = "vorticity_drag"
    api_version: ClassVar[str] = PHYSICS_API_VERSION
    description: ClassVar[str] = "Adds linear damping to the vorticity equation."

    def rhs_addition(
        self,
        state: ReducedMHDState,
        params: ReducedMHDParams,
        *,
        lengths: tuple[float, float],
    ) -> ReducedMHDState:
        del params, lengths
        return ReducedMHDState(
            psi=jnp.zeros_like(state.psi),
            omega=-self.rate * state.omega,
        )


@dataclass(frozen=True)
class ElectronPressureTensorTerm:
    r"""Toy pressure-tensor Ohm's-law closure using anisotropic current smoothing."""

    chi_x: float = 0.0
    chi_y: float = 0.0

    name: ClassVar[str] = "electron_pressure_tensor"
    api_version: ClassVar[str] = PHYSICS_API_VERSION
    description: ClassVar[str] = (
        "Adds an anisotropic electron-pressure-tensor closure to the flux equation."
    )

    def rhs_addition(
        self,
        state: ReducedMHDState,
        params: ReducedMHDParams,
        *,
        lengths: tuple[float, float],
    ) -> ReducedMHDState:
        del params
        current = -laplacian(state.psi, lengths=lengths)
        pressure_divergence = self.chi_x * fft_derivative(
            current,
            axis=0,
            length=lengths[0],
            order=2,
        ) + self.chi_y * fft_derivative(
            current,
            axis=1,
            length=lengths[1],
            order=2,
        )
        return ReducedMHDState(
            psi=pressure_divergence,
            omega=jnp.zeros_like(state.omega),
        )


@dataclass(frozen=True)
class ToyHallOhmTerm:
    r"""Reduced-state toy Hall Ohm's-law term ``d_i [j,\psi]``."""

    ion_skin_depth: float = 0.0

    name: ClassVar[str] = "toy_hall_ohm"
    api_version: ClassVar[str] = PHYSICS_API_VERSION
    description: ClassVar[str] = "Adds a reduced-state toy Hall bracket to the flux equation."

    def rhs_addition(
        self,
        state: ReducedMHDState,
        params: ReducedMHDParams,
        *,
        lengths: tuple[float, float],
    ) -> ReducedMHDState:
        del params
        current = -laplacian(state.psi, lengths=lengths)
        return ReducedMHDState(
            psi=self.ion_skin_depth * _poisson_bracket(current, state.psi, lengths=lengths),
            omega=jnp.zeros_like(state.omega),
        )


TermFactory = Callable[[Mapping[str, Any]], PhysicsTerm]


class PhysicsRegistry:
    """Registry for versioned physics-term factories."""

    def __init__(self) -> None:
        self._factories: dict[str, TermFactory] = {}

    def register(self, name: str, factory: TermFactory) -> None:
        """Register a factory under a stable term name."""
        if not name:
            raise ValueError("physics term name must be non-empty")
        self._factories[name] = factory

    def create(self, name: str, parameters: Mapping[str, Any] | None = None) -> PhysicsTerm:
        """Create a term from a registered factory."""
        if name not in self._factories:
            known = ", ".join(sorted(self._factories))
            raise KeyError(f"unknown physics term {name!r}; known terms: {known}")
        return self._factories[name](parameters or {})

    def names(self) -> tuple[str, ...]:
        """Return registered names in deterministic order."""
        return tuple(sorted(self._factories))

    def metadata(self) -> tuple[PhysicsTermMetadata, ...]:
        """Return metadata for all registered terms."""
        items = []
        for name in self.names():
            term = self.create(name)
            items.append(
                PhysicsTermMetadata(
                    name=name,
                    api_version=term.api_version,
                    description=term.description,
                    parameters=_term_parameters(term),
                )
            )
        return tuple(items)


def default_physics_registry() -> PhysicsRegistry:
    """Return the built-in physics-term registry."""
    registry = PhysicsRegistry()
    registry.register("electron_pressure_tensor", _electron_pressure_tensor_factory)
    registry.register("hyper_resistivity", _hyper_resistivity_factory)
    registry.register("toy_hall_ohm", _toy_hall_ohm_factory)
    registry.register("vorticity_drag", _vorticity_drag_factory)
    return registry


def load_physics_plugin_modules(
    registry: PhysicsRegistry,
    module_names: tuple[str, ...],
) -> PhysicsRegistry:
    """Load user physics plugins that expose ``register_physics(registry)``."""
    for module_name in module_names:
        module = _import_user_module(module_name)
        register = getattr(module, "register_physics", None)
        if register is None:
            raise AttributeError(
                f"physics plugin module {module_name!r} must define register_physics(registry)"
            )
        register(registry)
    return registry


def _import_user_module(module_name: str):
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError:
        cwd = str(Path.cwd())
        if cwd not in sys.path:
            sys.path.insert(0, cwd)
        try:
            return importlib.import_module(module_name)
        except ModuleNotFoundError:
            module_path = Path.cwd().joinpath(*module_name.split(".")).with_suffix(".py")
            if not module_path.exists():
                raise
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            if spec is None or spec.loader is None:
                raise
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            return module


def build_physics_terms(
    names: tuple[str, ...],
    term_parameters: Mapping[str, Mapping[str, Any]],
    *,
    registry: PhysicsRegistry | None = None,
    plugin_modules: tuple[str, ...] = (),
) -> tuple[PhysicsTerm, ...]:
    """Build configured terms from names and per-term parameter mappings."""
    active_registry = registry or default_physics_registry()
    if plugin_modules:
        load_physics_plugin_modules(active_registry, plugin_modules)
    return tuple(active_registry.create(name, term_parameters.get(name, {})) for name in names)


def apply_physics_terms(
    base_rhs: ReducedMHDState,
    terms: tuple[PhysicsTerm, ...],
    state: ReducedMHDState,
    params: ReducedMHDParams,
    *,
    lengths: tuple[float, float],
) -> ReducedMHDState:
    """Add all configured physics-term RHS contributions to a base RHS."""
    psi = base_rhs.psi
    omega = base_rhs.omega
    for term in terms:
        addition = term.rhs_addition(state, params, lengths=lengths)
        psi = psi + addition.psi
        omega = omega + addition.omega
    return ReducedMHDState(psi=psi, omega=omega)


def _hyper_resistivity_factory(parameters: Mapping[str, Any]) -> HyperResistivityTerm:
    return HyperResistivityTerm(
        eta4=float(parameters.get("eta4", 0.0)),
        nu4=float(parameters.get("nu4", 0.0)),
    )


def _vorticity_drag_factory(parameters: Mapping[str, Any]) -> VorticityDragTerm:
    return VorticityDragTerm(rate=float(parameters.get("rate", 0.0)))


def _electron_pressure_tensor_factory(
    parameters: Mapping[str, Any],
) -> ElectronPressureTensorTerm:
    return ElectronPressureTensorTerm(
        chi_x=float(parameters.get("chi_x", 0.0)),
        chi_y=float(parameters.get("chi_y", 0.0)),
    )


def _toy_hall_ohm_factory(parameters: Mapping[str, Any]) -> ToyHallOhmTerm:
    return ToyHallOhmTerm(ion_skin_depth=float(parameters.get("ion_skin_depth", 0.0)))


def _poisson_bracket(
    a: jnp.ndarray,
    b: jnp.ndarray,
    *,
    lengths: tuple[float, float],
) -> jnp.ndarray:
    da_dx, da_dy = gradient(a, lengths=lengths)
    db_dx, db_dy = gradient(b, lengths=lengths)
    return da_dx * db_dy - da_dy * db_dx


def _term_parameters(term: PhysicsTerm) -> dict[str, float]:
    if isinstance(term, HyperResistivityTerm):
        return {"eta4": term.eta4, "nu4": term.nu4}
    if isinstance(term, VorticityDragTerm):
        return {"rate": term.rate}
    if isinstance(term, ElectronPressureTensorTerm):
        return {"chi_x": term.chi_x, "chi_y": term.chi_y}
    if isinstance(term, ToyHallOhmTerm):
        return {"ion_skin_depth": term.ion_skin_depth}
    return {}
