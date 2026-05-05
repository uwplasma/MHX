"""Diagnostics for reduced-MHD states and trajectories."""

from __future__ import annotations

import importlib
import importlib.util
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax.numpy as jnp
from jaxtyping import Array

from mhx.equations.reduced_mhd import stream_function
from mhx.numerics.spectral import gradient, spectral_wavenumbers
from mhx.state import ReducedMHDState, ReducedMHDTrajectory


@dataclass(frozen=True)
class DiagnosticContext:
    """Inputs shared by reduced-MHD trajectory diagnostics."""

    trajectory: ReducedMHDTrajectory
    initial_state: ReducedMHDState
    lengths: tuple[float, float]
    mode: tuple[int, int]
    fit_time_window: tuple[float, float] | None


@dataclass(frozen=True)
class DiagnosticSpec:
    """Metadata and callable for one named reduced-MHD diagnostic."""

    name: str
    description: str
    output_keys: tuple[str, ...]
    compute: Callable[[DiagnosticContext], dict[str, Any]]


class DiagnosticsRegistry:
    """Registry for config-selectable reduced-MHD diagnostics."""

    def __init__(self) -> None:
        self._items: dict[str, DiagnosticSpec] = {}

    def register(self, spec: DiagnosticSpec) -> None:
        """Register a diagnostic specification by name."""
        if spec.name in self._items:
            raise ValueError(f"diagnostic already registered: {spec.name}")
        self._items[spec.name] = spec

    def get(self, name: str) -> DiagnosticSpec:
        """Return a diagnostic specification, raising a clear error if absent."""
        try:
            return self._items[name]
        except KeyError as exc:
            available = ", ".join(sorted(self._items))
            raise KeyError(f"unknown diagnostic {name!r}; available: {available}") from exc

    def names(self) -> tuple[str, ...]:
        """Return registered diagnostic names in deterministic order."""
        return tuple(sorted(self._items))

    def metadata(self) -> tuple[dict[str, Any], ...]:
        """Return JSON-compatible diagnostic metadata."""
        return tuple(
            {
                "name": spec.name,
                "description": spec.description,
                "output_keys": list(spec.output_keys),
            }
            for spec in (self._items[name] for name in self.names())
        )

    def compute(
        self,
        names: tuple[str, ...],
        context: DiagnosticContext,
    ) -> dict[str, Any]:
        """Compute selected diagnostics and merge their output dictionaries."""
        diagnostics: dict[str, Any] = {"diagnostic_quantities": list(names)}
        for name in names:
            diagnostics.update(self.get(name).compute(context))
        return diagnostics


def magnetic_energy(state: ReducedMHDState, *, lengths: tuple[float, float]) -> Array:
    r"""Return mean magnetic perturbation energy ``0.5 <|∇ψ|²>``."""
    grad_psi = gradient(state.psi, lengths=lengths)
    return 0.5 * sum(jnp.mean(component**2) for component in grad_psi)


def kinetic_energy(state: ReducedMHDState, *, lengths: tuple[float, float]) -> Array:
    r"""Return mean kinetic energy ``0.5 <|∇φ|²>``."""
    phi = stream_function(state.omega, lengths=lengths)
    grad_phi = gradient(phi, lengths=lengths)
    return 0.5 * sum(jnp.mean(component**2) for component in grad_phi)


def total_energy(state: ReducedMHDState, *, lengths: tuple[float, float]) -> Array:
    """Return reduced-MHD magnetic plus kinetic energy."""
    return magnetic_energy(state, lengths=lengths) + kinetic_energy(state, lengths=lengths)


def magnetic_divergence_linf(state: ReducedMHDState, *, lengths: tuple[float, float]) -> Array:
    r"""Return ``||∇·B_\perp||_∞`` for ``B_\perp=(∂_yψ,-∂_xψ)``."""
    psi = jnp.asarray(state.psi)
    kx = spectral_wavenumbers(psi.shape[0], lengths[0]).reshape((-1, 1))
    ky = spectral_wavenumbers(psi.shape[1], lengths[1]).reshape((1, -1))
    psi_hat = jnp.fft.fftn(psi)
    div_hat = (1j * kx) * (1j * ky) * psi_hat - (1j * ky) * (1j * kx) * psi_hat
    div_b = jnp.fft.ifftn(div_hat)
    return jnp.max(jnp.abs(div_b))


def mode_amplitude(state: ReducedMHDState, *, mode: tuple[int, int]) -> Array:
    """Return the absolute normalized Fourier amplitude of ``psi`` for a mode."""
    psi_hat = jnp.fft.fftn(state.psi) / state.psi.size
    return jnp.abs(psi_hat[mode[0] % state.psi.shape[0], mode[1] % state.psi.shape[1]])


def trajectory_mode_amplitude(
    trajectory: ReducedMHDTrajectory,
    *,
    mode: tuple[int, int],
) -> Array:
    """Return a saved trajectory's mode-amplitude time series."""
    return jnp.asarray(
        [mode_amplitude(state, mode=mode) for state in _iter_states(trajectory.states)]
    )


def fit_exponential_growth(times: Array, amplitudes: Array) -> Array:
    r"""Fit ``amplitude ≈ A exp(γ t)`` and return ``γ``."""
    if times.shape[0] < 2:
        raise ValueError("at least two samples are required for a growth-rate fit")
    log_amplitudes = jnp.log(jnp.maximum(amplitudes, jnp.finfo(amplitudes.dtype).tiny))
    centered_time = times - jnp.mean(times)
    centered_log = log_amplitudes - jnp.mean(log_amplitudes)
    denominator = jnp.sum(centered_time**2)
    return jnp.where(
        denominator == 0.0,
        jnp.nan,
        jnp.sum(centered_time * centered_log) / denominator,
    )


def select_fit_window(
    times: Array,
    amplitudes: Array,
    *,
    window: tuple[float, float] | None,
) -> tuple[Array, Array]:
    """Select samples inside an inclusive fit-time window."""
    if window is None:
        return times, amplitudes
    start, stop = window
    if stop <= start:
        raise ValueError("fit window stop must exceed start")
    mask = (times >= start) & (times <= stop)
    return times[mask], amplitudes[mask]


def trajectory_energies(
    trajectory: ReducedMHDTrajectory,
    *,
    lengths: tuple[float, float],
) -> dict[str, Array]:
    """Return energy time series for a saved trajectory."""
    magnetic = jnp.asarray(
        [magnetic_energy(state, lengths=lengths) for state in _iter_states(trajectory.states)]
    )
    kinetic = jnp.asarray(
        [kinetic_energy(state, lengths=lengths) for state in _iter_states(trajectory.states)]
    )
    return {
        "time": trajectory.times,
        "magnetic": magnetic,
        "kinetic": kinetic,
        "total": magnetic + kinetic,
    }


def compute_reduced_mhd_diagnostics(
    trajectory: ReducedMHDTrajectory,
    *,
    initial_state: ReducedMHDState,
    lengths: tuple[float, float],
    quantities: tuple[str, ...],
    mode: tuple[int, int],
    fit_time_window: tuple[float, float] | None,
    registry: DiagnosticsRegistry | None = None,
    plugin_modules: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Compute selected reduced-MHD diagnostics through the registry API."""
    diagnostic_registry = registry or default_diagnostics_registry()
    if plugin_modules:
        load_diagnostics_plugin_modules(diagnostic_registry, plugin_modules)
    context = DiagnosticContext(
        trajectory=trajectory,
        initial_state=initial_state,
        lengths=lengths,
        mode=mode,
        fit_time_window=fit_time_window,
    )
    return diagnostic_registry.compute(quantities, context)


def load_diagnostics_plugin_modules(
    registry: DiagnosticsRegistry,
    module_names: tuple[str, ...],
) -> DiagnosticsRegistry:
    """Load user diagnostics plugins exposing ``register_diagnostics(registry)``."""
    for module_name in module_names:
        module = _import_user_module(module_name)
        register = getattr(module, "register_diagnostics", None)
        if register is None:
            raise AttributeError(
                "diagnostics plugin module "
                f"{module_name!r} must define register_diagnostics(registry)"
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


def default_diagnostics_registry() -> DiagnosticsRegistry:
    """Return the built-in reduced-MHD diagnostics registry."""
    registry = DiagnosticsRegistry()
    registry.register(
        DiagnosticSpec(
            name="energy",
            description="Initial/final magnetic, kinetic, and total reduced-MHD energies.",
            output_keys=(
                "initial_total_energy",
                "final_total_energy",
                "final_magnetic_energy",
                "final_kinetic_energy",
            ),
            compute=_energy_diagnostics,
        )
    )
    registry.register(
        DiagnosticSpec(
            name="mode_growth",
            description="Fourier-mode amplitude and exponential growth/decay fit.",
            output_keys=(
                "diagnostic_mode",
                "fit_time_window",
                "fit_sample_count",
                "initial_mode_amplitude",
                "final_mode_amplitude",
                "gamma_fit",
            ),
            compute=_mode_growth_diagnostics,
        )
    )
    registry.register(
        DiagnosticSpec(
            name="divergence_error",
            description="Final spectral divergence error of B_perp = (d_y psi, -d_x psi).",
            output_keys=("final_magnetic_divergence_linf",),
            compute=_divergence_error_diagnostics,
        )
    )
    return registry


def _energy_diagnostics(context: DiagnosticContext) -> dict[str, Any]:
    energies = trajectory_energies(context.trajectory, lengths=context.lengths)
    return {
        "initial_total_energy": float(total_energy(context.initial_state, lengths=context.lengths)),
        "final_total_energy": float(energies["total"][-1]),
        "final_magnetic_energy": float(energies["magnetic"][-1]),
        "final_kinetic_energy": float(energies["kinetic"][-1]),
    }


def _mode_growth_diagnostics(context: DiagnosticContext) -> dict[str, Any]:
    amplitudes = trajectory_mode_amplitude(context.trajectory, mode=context.mode)
    fit_times, fit_amplitudes = select_fit_window(
        context.trajectory.times,
        amplitudes,
        window=context.fit_time_window,
    )
    return {
        "diagnostic_mode": list(context.mode),
        "fit_time_window": (
            None if context.fit_time_window is None else list(context.fit_time_window)
        ),
        "fit_sample_count": float(fit_times.shape[0]),
        "initial_mode_amplitude": float(amplitudes[0]),
        "final_mode_amplitude": float(amplitudes[-1]),
        "gamma_fit": float(fit_exponential_growth(fit_times, fit_amplitudes)),
    }


def _divergence_error_diagnostics(context: DiagnosticContext) -> dict[str, Any]:
    final_state = ReducedMHDState(
        psi=context.trajectory.states.psi[-1],
        omega=context.trajectory.states.omega[-1],
    )
    return {
        "final_magnetic_divergence_linf": float(
            magnetic_divergence_linf(final_state, lengths=context.lengths)
        )
    }


def _iter_states(states: ReducedMHDState):
    for index in range(states.psi.shape[0]):
        yield ReducedMHDState(psi=states.psi[index], omega=states.omega[index])
