"""Paper-facing diagnostics for periodic incompressible 3D MHD."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import jax
import numpy as np

from mhx.equations import mhd3d
from mhx.state.mhd3d import MHD3DState, MHD3DTrajectory


def _state_at(trajectory: MHD3DTrajectory, index: int) -> MHD3DState:
    return jax.tree.map(lambda leaf: leaf[index], trajectory.states)


def shell_spectra(state: MHD3DState, *, shape: tuple[int, int, int]) -> dict[str, np.ndarray]:
    """Return isotropic integer-shell kinetic, magnetic, and total spectra.

    Shell ``q`` contains modes satisfying ``q - 1/2 <= |k| < q + 1/2``.
    The half-spectrum conjugate weights and unnormalized FFT convention are
    included, so summing a spectrum reproduces the corresponding mean energy.
    """
    nx, ny, nz = shape
    kx = np.fft.fftfreq(nx, d=1.0 / nx)[:, None, None]
    ky = np.fft.fftfreq(ny, d=1.0 / ny)[None, :, None]
    kz = np.fft.rfftfreq(nz, d=1.0 / nz)[None, None, :]
    shells = np.floor(np.sqrt(kx * kx + ky * ky + kz * kz) + 0.5).astype(int)

    weights = 2.0 * np.ones(nz // 2 + 1)
    weights[0] = 1.0
    if nz % 2 == 0:
        weights[-1] = 1.0
    weights = weights[None, None, :]
    normalization = float(nx * ny * nz) ** 2

    def spectrum(field_hat) -> np.ndarray:
        density = (
            0.5
            * weights
            * np.sum(np.abs(np.asarray(field_hat)) ** 2, axis=0)
            / normalization
        )
        return np.bincount(
            shells.ravel(),
            weights=density.ravel(),
            minlength=int(shells.max()) + 1,
        )

    kinetic = spectrum(state.v_hat)
    magnetic = spectrum(state.b_hat)
    return {
        "k": np.arange(kinetic.size),
        "kinetic": kinetic,
        "magnetic": magnetic,
        "total": kinetic + magnetic,
    }


def signed_mode_coefficient(
    field_hat,
    mode: tuple[int, int, int],
    *,
    shape: tuple[int, int, int],
) -> np.ndarray:
    """Return a normalized vector coefficient from an RFFT half-spectrum.

    Negative last-axis modes are recovered using the real-field conjugacy
    relation ``f(-k) = conj(f(k))``.
    """
    kx, ky, kz = mode
    nx, ny, nz = shape
    if abs(kz) > nz // 2:
        raise ValueError(f"mode {mode} is outside the represented z spectrum")
    values = np.asarray(field_hat)
    if kz < 0:
        coefficient = np.conj(values[:, (-kx) % nx, (-ky) % ny, -kz])
    else:
        coefficient = values[:, kx % nx, ky % ny, kz]
    return coefficient / float(nx * ny * nz)


def trajectory_bulk_diagnostics(
    trajectory: MHD3DTrajectory,
    *,
    shape: tuple[int, int, int],
    viscosity: float,
    resistivity: float,
    lengths: tuple[float, float, float] = (2.0 * np.pi,) * 3,
) -> dict[str, np.ndarray]:
    """Return energy and Laplacian dissipation histories for a trajectory."""
    k = mhd3d.wavevectors(shape, lengths)
    kinetic = []
    magnetic = []
    dissipation = []
    for index in range(len(trajectory.times)):
        state = _state_at(trajectory, index)
        energy = mhd3d.energies(state, shape=shape)
        vorticity = MHD3DState(
            v_hat=mhd3d.curl_hat(state.v_hat, k),
            b_hat=mhd3d.curl_hat(state.b_hat, k),
        )
        squared_curls = mhd3d.energies(vorticity, shape=shape)
        kinetic.append(float(energy["kinetic"]))
        magnetic.append(float(energy["magnetic"]))
        dissipation.append(
            2.0 * viscosity * float(squared_curls["kinetic"])
            + 2.0 * resistivity * float(squared_curls["magnetic"])
        )
    return {
        "time": np.asarray(trajectory.times),
        "kinetic": np.asarray(kinetic),
        "magnetic": np.asarray(magnetic),
        "total": np.asarray(kinetic) + np.asarray(magnetic),
        "dissipation": np.asarray(dissipation),
    }


def peak_window_spectra(
    trajectory: MHD3DTrajectory,
    *,
    shape: tuple[int, int, int],
    dissipation: Sequence[float],
    half_width: float = 0.25,
) -> dict[str, np.ndarray | float | int]:
    """Average shell spectra around peak dissipation as in Lee et al. (2010)."""
    times = np.asarray(trajectory.times)
    values = np.asarray(dissipation)
    peak_index = int(np.argmax(values))
    peak_time = float(times[peak_index])
    indices = np.flatnonzero(np.abs(times - peak_time) <= half_width)
    if indices.size == 0:
        indices = np.asarray([peak_index])
    spectra = [shell_spectra(_state_at(trajectory, int(i)), shape=shape) for i in indices]
    return {
        "k": spectra[0]["k"],
        "kinetic": np.mean([item["kinetic"] for item in spectra], axis=0),
        "magnetic": np.mean([item["magnetic"] for item in spectra], axis=0),
        "total": np.mean([item["total"] for item in spectra], axis=0),
        "peak_index": peak_index,
        "peak_time": peak_time,
        "sample_count": int(indices.size),
    }


def collision_mode_histories(
    trajectory: MHD3DTrajectory,
    *,
    shape: tuple[int, int, int],
) -> dict[str, np.ndarray]:
    """Return the Howes--Nielson primary, secondary, and tertiary histories."""
    modes = {
        "primary_plus": (1, 0, -1),
        "primary_minus": (0, 1, 1),
        "secondary": (1, 1, 0),
        "tertiary_plus": (2, 1, -1),
        "tertiary_minus": (1, 2, 1),
    }
    result: dict[str, np.ndarray] = {"time": np.asarray(trajectory.times)}
    for name, mode in modes.items():
        velocity = []
        magnetic = []
        z_plus = []
        z_minus = []
        for index in range(len(trajectory.times)):
            state = _state_at(trajectory, index)
            v = signed_mode_coefficient(state.v_hat, mode, shape=shape)
            b = signed_mode_coefficient(state.b_hat, mode, shape=shape)
            velocity.append(np.linalg.norm(v))
            magnetic.append(np.linalg.norm(b))
            z_plus.append(np.linalg.norm(v + b))
            z_minus.append(np.linalg.norm(v - b))
        result[f"{name}_velocity"] = np.asarray(velocity)
        result[f"{name}_magnetic"] = np.asarray(magnetic)
        result[f"{name}_z_plus"] = np.asarray(z_plus)
        result[f"{name}_z_minus"] = np.asarray(z_minus)
    return result


def alfven_collision_reference(
    times: Sequence[float],
    *,
    amplitude_plus: float,
    amplitude_minus: float,
    alfven_speed: float = 1.0,
    k_perpendicular: float = 1.0,
    k_parallel: float = 1.0,
) -> Mapping[str, np.ndarray]:
    """Howes--Nielson magnetic-mode amplitudes through O(epsilon^3).

    Implements the Fourier coefficients of their equations (36) and (40) for
    the key ``(1,1,0)``, ``(2,1,-1)``, and ``(1,2,1)`` modes.
    """
    time = np.asarray(times, dtype=float)
    omega = k_parallel * alfven_speed

    secondary_prefactor = (
        amplitude_plus
        * amplitude_minus
        * k_perpendicular
        / (16.0 * alfven_speed**2 * k_parallel)
    )
    secondary_scalar = secondary_prefactor * (
        np.exp(-2j * omega * time) - 1.0
    )
    secondary = np.sqrt(2.0) * np.abs(secondary_scalar)

    plus_prefactor = (
        amplitude_plus**2
        * amplitude_minus
        * k_perpendicular**2
        / (640.0 * alfven_speed**3 * k_parallel**2)
    )
    plus_scalar = plus_prefactor * (
        4j * omega * time * np.exp(-1j * omega * time)
        + 1.5 * np.exp(1j * omega * time)
        - 5.0 * np.exp(-1j * omega * time)
        + 3.5 * np.exp(-3j * omega * time)
    )
    tertiary_plus = np.sqrt(5.0) * np.abs(plus_scalar)

    minus_prefactor = (
        amplitude_plus
        * amplitude_minus**2
        * k_perpendicular**2
        / (640.0 * alfven_speed**3 * k_parallel**2)
    )
    minus_scalar = minus_prefactor * (
        -4j * omega * time * np.exp(-1j * omega * time)
        + 1.5 * np.exp(1j * omega * time)
        - np.exp(-1j * omega * time)
        - 0.5 * np.exp(-3j * omega * time)
    )
    tertiary_minus = np.sqrt(5.0) * np.abs(minus_scalar)

    # Equation (40) contains bounded oscillatory terms plus a secular term.
    # These amplitudes isolate the latter; their squared sum follows Eq. (41).
    secular_plus = np.sqrt(5.0) * 4.0 * omega * time * abs(plus_prefactor)
    secular_minus = np.sqrt(5.0) * 4.0 * omega * time * abs(minus_prefactor)
    return {
        "time": time,
        "secondary_magnetic": secondary,
        "tertiary_plus_magnetic": tertiary_plus,
        "tertiary_minus_magnetic": tertiary_minus,
        "tertiary_plus_secular_magnetic": secular_plus,
        "tertiary_minus_secular_magnetic": secular_minus,
    }


__all__ = [
    "alfven_collision_reference",
    "collision_mode_histories",
    "peak_window_spectra",
    "shell_spectra",
    "signed_mode_coefficient",
    "trajectory_bulk_diagnostics",
]
