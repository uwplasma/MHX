"""Run and render the smooth periodic Kelvin--Helmholtz example."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, NamedTuple

try:
    from .common import (
        encode_frames,
        figure_frame,
        load_source_metadata,
        require_source,
        sample_indices,
        source_dir,
        validate_array_shape,
        write_source_metadata,
    )
except ImportError:  # Direct script execution.
    from common import (
        encode_frames,
        figure_frame,
        load_source_metadata,
        require_source,
        sample_indices,
        source_dir,
        validate_array_shape,
        write_source_metadata,
    )

CASE = "kelvin_helmholtz"
PRESETS = {
    "preview": {
        "shape": (32, 64),
        "dt": 2.0e-3,
        "t_end": 0.2,
        "save_every": 20,
        "viscosity": 1.0e-3,
    },
    "final": {
        "shape": (128, 256),
        "dt": 1.0e-3,
        "t_end": 4.0,
        "save_every": 100,
        "viscosity": 2.5e-4,
    },
}


class KelvinHelmholtzState(NamedTuple):
    """Reduced-MHD hydrodynamic state plus a passive dye."""

    mhd: Any
    dye: Any


def _initial_state(grid: Any) -> KelvinHelmholtzState:
    import jax.numpy as jnp

    from mhx.state import ReducedMHDState

    x, y = grid.mesh()
    shear_width = 0.05
    perturbation_width = 0.2
    amplitude = 1.0e-2
    flow_speed = 1.0
    y1, y2 = 0.5, 1.5
    wavenumber_x = 2.0 * jnp.pi / grid.lengths[0]
    sech1 = 1.0 / jnp.cosh((y - y1) / shear_width)
    sech2 = 1.0 / jnp.cosh((y - y2) / shear_width)
    dy_velocity_x = (flow_speed / shear_width) * (sech1**2 - sech2**2)
    envelope = jnp.exp(-((y - y1) ** 2) / perturbation_width**2) + jnp.exp(
        -((y - y2) ** 2) / perturbation_width**2
    )
    dx_velocity_y = amplitude * wavenumber_x * jnp.cos(wavenumber_x * x) * envelope
    omega = dx_velocity_y - dy_velocity_x
    dye = 0.5 * (
        jnp.tanh((y - y2) / shear_width)
        - jnp.tanh((y - y1) / shear_width)
        + 2.0
    )
    return KelvinHelmholtzState(
        mhd=ReducedMHDState(psi=jnp.zeros_like(x), omega=omega),
        dye=dye,
    )


def simulate_config(*, config: dict[str, Any], outdir: Path, preset: str) -> Path:
    """Run one explicit incompressible passive-dye configuration."""
    import jax.numpy as jnp
    import numpy as np

    from mhx.config import MeshConfig
    from mhx.equations.reduced_mhd import poisson_bracket, reduced_mhd_rhs, stream_function
    from mhx.grids import CartesianGrid
    from mhx.numerics.spectral import laplacian
    from mhx.state import ReducedMHDParams
    from mhx.time_integrators import evolve_rk4

    grid = CartesianGrid.from_mesh_config(
        MeshConfig(shape=config["shape"], lower=(0.0, 0.0), upper=(1.0, 2.0))
    )
    params = ReducedMHDParams(resistivity=0.0, viscosity=config["viscosity"])
    initial = _initial_state(grid)

    def rhs(state: KelvinHelmholtzState) -> KelvinHelmholtzState:
        mhd_rhs = reduced_mhd_rhs(state.mhd, params, lengths=grid.lengths)
        phi = stream_function(state.mhd.omega, lengths=grid.lengths)
        dye_rhs = -poisson_bracket(phi, state.dye, lengths=grid.lengths) + (
            params.viscosity * laplacian(state.dye, lengths=grid.lengths)
        )
        return KelvinHelmholtzState(mhd=mhd_rhs, dye=dye_rhs)

    steps = int(round(config["t_end"] / config["dt"]))
    trajectory = evolve_rk4(
        initial,
        rhs,
        dt=config["dt"],
        steps=steps,
        save_every=config["save_every"],
    )
    time = np.concatenate(([0.0], np.asarray(trajectory.times, dtype=float)))
    dye = np.concatenate(
        (np.asarray(initial.dye)[None, ...], np.asarray(trajectory.states.dye)), axis=0
    )
    omega = np.concatenate(
        (
            np.asarray(initial.mhd.omega)[None, ...],
            np.asarray(trajectory.states.mhd.omega),
        ),
        axis=0,
    )
    dye_safe = jnp.clip(jnp.asarray(dye), 1.0e-12, 1.0)
    entropy = np.asarray(jnp.mean(-dye_safe * jnp.log(dye_safe), axis=(1, 2)) * 2.0)
    outdir.mkdir(parents=True, exist_ok=True)
    source = outdir / "kelvin_helmholtz.npz"
    np.savez_compressed(source, time=time, dye=dye, omega=omega, entropy=entropy)
    write_source_metadata(
        outdir,
        {
            "case": CASE,
            "preset": preset,
            **config,
            "domain": [0.0, 1.0, 0.0, 2.0],
            "trajectory": source.name,
            "claim_scope": "smooth periodic hydrodynamic-limit Kelvin--Helmholtz demonstration",
        },
    )
    return source


def simulate(*, preset: str, outdir: Path) -> Path:
    """Run the notebook-derived incompressible passive-dye example."""
    return simulate_config(config=PRESETS[preset], outdir=outdir, preset=preset)


def render(*, source: Path, maximum_frames: int = 40) -> dict[str, Path]:
    """Render passive dye with vorticity contours and physical axes."""
    import matplotlib.pyplot as plt
    import numpy as np

    source = require_source(source, CASE)
    metadata = load_source_metadata(source)
    with np.load(source, allow_pickle=False) as data:
        time = np.asarray(data["time"], dtype=float)
        dye = np.asarray(data["dye"], dtype=float)
        omega = np.asarray(data["omega"], dtype=float)
    if metadata.get("shape"):
        validate_array_shape(dye.shape[1:], metadata["shape"], label=CASE)

    indices = sample_indices(len(time), maximum_frames)
    omega_limit = max(float(np.percentile(np.abs(omega[indices]), 99.5)), np.finfo(float).eps)
    x = np.linspace(0.0, 1.0, dye.shape[1], endpoint=False)
    y = np.linspace(0.0, 2.0, dye.shape[2], endpoint=False)
    contour_levels = np.linspace(-omega_limit, omega_limit, 13)
    frames = []
    for index in indices:
        figure, axis = plt.subplots(figsize=(5.3, 7.0), dpi=100, constrained_layout=True)
        image = axis.imshow(
            dye[index].T,
            origin="lower",
            extent=(0.0, 1.0, 0.0, 2.0),
            cmap="RdBu_r",
            vmin=0.0,
            vmax=1.0,
            aspect="equal",
        )
        axis.contour(
            x,
            y,
            omega[index].T,
            levels=contour_levels,
            colors="white",
            linewidths=0.45,
            alpha=0.7,
        )
        axis.set_xlabel(r"$x/L_x$")
        axis.set_ylabel(r"$y/L_x$")
        axis.set_title(f"Kelvin--Helmholtz passive-dye roll-up, t = {time[index]:.2f}")
        figure.colorbar(image, ax=axis, label="dye concentration c", shrink=0.82)
        frames.append(figure_frame(figure))
        plt.close(figure)

    return encode_frames(
        frames,
        stem="kelvin_helmholtz",
        source=source,
        source_metadata=metadata,
        times=time[indices],
        write_gif=True,
        write_mp4=True,
        fps=10,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    simulate_parser = subparsers.add_parser("simulate")
    simulate_parser.add_argument("--preset", choices=PRESETS, default="preview")
    simulate_parser.add_argument("--outdir", type=Path)
    render_parser = subparsers.add_parser("render")
    render_parser.add_argument("--source", type=Path)
    render_parser.add_argument("--preset", choices=PRESETS, default="preview")
    render_parser.add_argument("--max-frames", type=int, default=40)
    args = parser.parse_args()
    if args.command == "simulate":
        outdir = args.outdir or source_dir(CASE, args.preset)
        print(simulate(preset=args.preset, outdir=outdir))
    else:
        source = args.source or source_dir(CASE, args.preset) / "kelvin_helmholtz.npz"
        for path in render(source=source, maximum_frames=args.max_frames).values():
            print(path)


if __name__ == "__main__":
    main()
