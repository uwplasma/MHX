"""The compressible run path behind ``mhx.Simulation(equations="compressible")``."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from rich.console import Console
from rich.table import Table

from mhx.equations import mhd3d
from mhx.equations.compressible import (
    CompressibleParams,
    CompressibleState,
    compressible_nonlinear,
)
from mhx.state.mhd3d import MHD3DTrajectory
from mhx.time_integrators.low_storage import evolve_if_rk3


@dataclass(frozen=True)
class CompressibleResult:
    """Trajectory, diagnostics, and timings from a compressible run."""

    trajectory: MHD3DTrajectory
    shape: tuple[int, int, int]
    parameters: CompressibleParams
    config: dict[str, object]
    diagnostics: dict[str, object]
    compile_seconds: float
    run_seconds: float
    device_count: int

    @property
    def final_state(self) -> CompressibleState:
        """Return the last saved spectral state."""
        return jax.tree.map(lambda leaf: leaf[-1], self.trajectory.states)

    @property
    def final_time(self) -> float:
        """Return the time of the last saved state."""
        return float(self.trajectory.times[-1])

    def print_summary(self, *, console: Console | None = None) -> None:
        """Print the main run results as a compact table."""
        output = console or Console()
        table = Table(title="MHX compressible result", show_header=False)
        table.add_column("Quantity", style="cyan")
        table.add_column("Value", justify="right")
        table.add_row("Final time", f"{self.final_time:.6g}")
        table.add_row("Saved states", str(self.trajectory.times.shape[0]))
        table.add_row("Devices", str(self.device_count))
        table.add_row("Compile time", f"{self.compile_seconds:.3f} s")
        table.add_row("Run time", f"{self.run_seconds:.3f} s")
        table.add_row(
            "Density spread",
            f"{float(self.diagnostics['final_density_relative_spread']):.3e}",
        )
        table.add_row(
            "Max |div B|",
            f"{float(self.diagnostics['final_magnetic_divergence_linf']):.3e}",
        )
        output.print(table)

    def save(self, path: str | Path) -> Path:
        """Save the real-space field history and metadata to one NPZ file."""
        output = Path(path)
        if output.suffix != ".npz":
            output = output / "trajectory.npz"
        output.parent.mkdir(parents=True, exist_ok=True)
        frames = int(self.trajectory.times.shape[0])

        def frame(transformer, index):
            return np.asarray(transformer(index))

        density = np.stack(
            [
                np.exp(
                    np.asarray(
                        mhd3d.to_physical(
                            self.trajectory.states.lnrho_hat[i][None],
                            shape=self.shape,
                        )[0]
                    )
                )
                for i in range(frames)
            ]
        )
        velocity = np.stack(
            [
                np.asarray(
                    mhd3d.to_physical(
                        self.trajectory.states.u_hat[i], shape=self.shape
                    )
                )
                for i in range(frames)
            ]
        )
        magnetic = np.stack(
            [
                np.asarray(
                    mhd3d.to_physical(
                        self.trajectory.states.b_hat[i], shape=self.shape
                    )
                )
                for i in range(frames)
            ]
        )
        np.savez_compressed(
            output,
            schema="mhx.compressible_trajectory.v1",
            dimension=3,
            times=np.asarray(self.trajectory.times),
            density=density,
            velocity=velocity,
            magnetic=magnetic,
            config=np.asarray(str(self.config)),
            diagnostics=np.asarray(str(self.diagnostics)),
        )
        return output

    def plot(self, path: str | Path) -> Path:
        """Write a four-panel summary: density, |j|, energies, density spread."""
        import matplotlib.pyplot as plt

        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        k = mhd3d.wavevectors(self.shape, (2.0 * jnp.pi,) * 3)
        final = self.final_state
        density = np.exp(
            np.asarray(
                mhd3d.to_physical(final.lnrho_hat[None], shape=self.shape)[0]
            )
        )
        current = np.asarray(
            mhd3d.to_physical(mhd3d.curl_hat(final.b_hat, k), shape=self.shape)
        )
        mid = self.shape[2] // 2
        current_slice = np.sqrt(np.sum(current * current, axis=0))[:, :, mid]

        times = np.asarray(self.trajectory.times)
        spreads = [
            float(
                np.std(
                    np.asarray(
                        mhd3d.to_physical(
                            self.trajectory.states.lnrho_hat[i][None],
                            shape=self.shape,
                        )[0]
                    )
                )
            )
            for i in range(len(times))
        ]

        figure, axes = plt.subplots(2, 2, figsize=(9.0, 7.0), constrained_layout=True)
        image = axes[0, 0].imshow(density[:, :, mid].T, origin="lower", cmap="viridis")
        axes[0, 0].set_title("density, midplane")
        figure.colorbar(image, ax=axes[0, 0], shrink=0.8)
        image = axes[0, 1].imshow(current_slice.T, origin="lower", cmap="inferno")
        axes[0, 1].set_title("|j|, midplane")
        figure.colorbar(image, ax=axes[0, 1], shrink=0.8)
        axes[1, 0].plot(times, spreads)
        axes[1, 0].set_xlabel("time")
        axes[1, 0].set_title("log-density spread")
        axes[1, 1].axis("off")
        figure.savefig(output, dpi=180)
        plt.close(figure)
        return output


def run_compressible(simulation) -> CompressibleResult:
    """Execute the compressible path for a configured :class:`mhx.Simulation`."""
    shape = simulation.shape
    lengths = tuple(
        upper - lower
        for lower, upper in zip(simulation.lower, simulation.upper, strict=True)
    )
    equilibrium = simulation.equilibrium
    if not hasattr(equilibrium, "initial_fields"):
        raise TypeError(
            "equations='compressible' needs a 3D equilibrium with "
            "initial_fields, for example mhx.OrszagTang3DEquilibrium()"
        )

    params = CompressibleParams(
        sound_speed=simulation.sound_speed,
        viscosity=simulation.viscosity,
        bulk_viscosity=simulation.bulk_viscosity,
        resistivity=simulation.resistivity,
        guide_field=simulation.guide_field,
    )
    k = mhd3d.wavevectors(shape, lengths)
    mask = mhd3d.two_thirds_mask_rfft(shape)

    velocity, magnetic = equilibrium.initial_fields(shape)
    spectral = mhd3d.spectral_shape(shape)
    state0 = CompressibleState(
        lnrho_hat=jnp.zeros(spectral, dtype=complex),
        u_hat=mhd3d.project(mhd3d.to_spectral(velocity), k),
        b_hat=mhd3d.project(mhd3d.to_spectral(magnetic), k),
    )
    zero_decay = CompressibleState(
        lnrho_hat=jnp.zeros(spectral),
        u_hat=jnp.zeros((3, *spectral)),
        b_hat=jnp.zeros((3, *spectral)),
    )

    def nonlinear(state: CompressibleState) -> CompressibleState:
        return compressible_nonlinear(state, params, shape=shape, k=k, mask=mask)

    def program(initial: CompressibleState) -> MHD3DTrajectory:
        return evolve_if_rk3(
            initial,
            nonlinear,
            zero_decay,
            dt=simulation.dt,
            steps=simulation.steps,
            save_every=simulation.save_every,
        )

    compiled = jax.jit(program)
    start = time.perf_counter()
    lowered = compiled.lower(state0).compile()
    compile_seconds = time.perf_counter() - start
    start = time.perf_counter()
    trajectory = lowered(state0)
    jax.block_until_ready(trajectory)
    run_seconds = time.perf_counter() - start

    final = jax.tree.map(lambda leaf: leaf[-1], trajectory.states)
    lnrho = mhd3d.to_physical(final.lnrho_hat[None], shape=shape)[0]
    diagnostics: dict[str, object] = {
        "final_density_relative_spread": float(jnp.std(jnp.exp(lnrho))),
        "final_magnetic_divergence_linf": float(
            mhd3d.divergence_linf(final.b_hat, k)
        ),
        "final_velocity_max": float(
            jnp.max(jnp.abs(mhd3d.to_physical(final.u_hat, shape=shape)))
        ),
    }
    config = {
        "equations": "compressible",
        "shape": list(shape),
        "sound_speed": float(simulation.sound_speed),
        "viscosity": float(simulation.viscosity),
        "bulk_viscosity": float(simulation.bulk_viscosity),
        "resistivity": float(simulation.resistivity),
        "dt": float(simulation.dt),
        "t_end": float(simulation.t_end),
        "equilibrium": getattr(type(equilibrium), "name", type(equilibrium).__name__),
    }
    return CompressibleResult(
        trajectory=MHD3DTrajectory(times=trajectory.times, states=trajectory.states),
        shape=shape,
        parameters=params,
        config=config,
        diagnostics=diagnostics,
        compile_seconds=compile_seconds,
        run_seconds=run_seconds,
        device_count=1,
    )
