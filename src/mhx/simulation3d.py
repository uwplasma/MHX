"""The 3D run path behind ``mhx.Simulation(equations="mhd3d")``."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh
from jaxtyping import Array
from rich.console import Console
from rich.table import Table

from mhx.equations import mhd3d
from mhx.numerics.spectral.pfft import shard_spectral
from mhx.state.mhd3d import MHD3DParams, MHD3DState, MHD3DTrajectory
from mhx.time_integrators.exponential import evolve_etdrk4
from mhx.time_integrators.low_storage import evolve_if_rk3


class Equilibrium3D(Protocol):
    """Objects that build component-first real ``(v, b)`` fields."""

    def initial_fields(
        self, shape: tuple[int, int, int]
    ) -> tuple[Array, Array]: ...


@dataclass(frozen=True)
class MHD3DResult:
    """Trajectory, diagnostics, and timings from one 3D simulation."""

    trajectory: MHD3DTrajectory
    shape: tuple[int, int, int]
    parameters: MHD3DParams
    config: dict[str, object]
    diagnostics: dict[str, object]
    compile_seconds: float
    run_seconds: float
    device_count: int

    @property
    def final_state(self) -> MHD3DState:
        """Return the last saved spectral state."""
        return jax.tree.map(lambda leaf: leaf[-1], self.trajectory.states)

    @property
    def final_time(self) -> float:
        """Return the time of the last saved state."""
        return float(self.trajectory.times[-1])

    def print_summary(self, *, console: Console | None = None) -> None:
        """Print the main run results as a compact table."""
        output = console or Console()
        table = Table(title="MHX 3D result", show_header=False)
        table.add_column("Quantity", style="cyan")
        table.add_column("Value", justify="right")
        table.add_row("Final time", f"{self.final_time:.6g}")
        table.add_row("Saved states", str(self.trajectory.times.shape[0]))
        table.add_row("Devices", str(self.device_count))
        table.add_row("Compile time", f"{self.compile_seconds:.3f} s")
        table.add_row("Run time", f"{self.run_seconds:.3f} s")
        table.add_row(
            "Initial energy", f"{float(self.diagnostics['initial_total_energy']):.6e}"
        )
        table.add_row(
            "Final energy", f"{float(self.diagnostics['final_total_energy']):.6e}"
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
        # Transform one saved frame at a time: transforming the whole
        # trajectory at once exhausts device memory after large runs.
        frames = int(self.trajectory.times.shape[0])
        velocity = np.stack(
            [
                np.asarray(
                    mhd3d.to_physical(
                        self.trajectory.states.v_hat[i], shape=self.shape
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
            schema="mhx.mhd3d_trajectory.v1",
            dimension=3,
            times=np.asarray(self.trajectory.times),
            velocity=velocity,
            magnetic=magnetic,
            config=np.asarray(str(self.config)),
            diagnostics=np.asarray(str(self.diagnostics)),
        )
        return output

    def plot(self, path: str | Path) -> Path:
        """Write a four-panel summary: |j| midplane, |v| midplane, energies."""
        import matplotlib.pyplot as plt

        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)

        plt.rcParams.update(
            {
                "axes.spines.top": False,
                "axes.spines.right": False,
                "font.size": 10,
            }
        )

        k = mhd3d.wavevectors(self.shape, (2.0 * jnp.pi,) * 3)
        final = self.final_state
        current = mhd3d.to_physical(
            mhd3d.curl_hat(final.b_hat, k), shape=self.shape
        )
        velocity = mhd3d.to_physical(final.v_hat, shape=self.shape)
        mid = self.shape[2] // 2
        current_slice = np.asarray(
            jnp.sqrt(jnp.sum(current * current, axis=0))[:, :, mid]
        )
        speed_slice = np.asarray(
            jnp.sqrt(jnp.sum(velocity * velocity, axis=0))[:, :, mid]
        )

        history = [
            mhd3d.energies(
                jax.tree.map(lambda leaf, i=i: leaf[i], self.trajectory.states),
                shape=self.shape,
            )
            for i in range(len(self.trajectory.times))
        ]
        times = np.asarray(self.trajectory.times)

        figure, axes = plt.subplots(2, 2, figsize=(11.0, 7.5), constrained_layout=True)
        for axis, field, title, clabel in (
            (axes[0, 0], current_slice, r"$|\mathbf{j}|$, midplane", r"$|\mathbf{j}|$"),
            (axes[0, 1], speed_slice, r"$|\mathbf{v}|$, midplane", r"$|\mathbf{v}|$"),
        ):
            image = axis.imshow(field.T, origin="lower", cmap="jet")
            axis.set_title(title)
            axis.set_xlabel("grid x")
            axis.set_ylabel("grid y")
            cbar = figure.colorbar(image, ax=axis, shrink=0.82)
            cbar.set_label(clabel)

        axes[1, 0].plot(times, [float(h["kinetic"]) for h in history], label="kinetic")
        axes[1, 0].plot(times, [float(h["magnetic"]) for h in history], label="magnetic")
        axes[1, 0].plot(times, [float(h["total"]) for h in history], label="total")
        axes[1, 0].set_xlabel("time")
        axes[1, 0].set_title("Mean energy")
        axes[1, 0].grid(True, alpha=0.25)
        axes[1, 0].legend(frameon=False)

        axes[1, 1].plot(times, [float(h["cross_helicity"]) for h in history])
        axes[1, 1].set_xlabel("time")
        axes[1, 1].set_title("Cross helicity")
        axes[1, 1].grid(True, alpha=0.25)

        figure.suptitle(
            f"MHX 3D incompressible MHD  —  "
            f"{getattr(type(self), '__name__', 'run')}",
            fontsize=13,
        )
        figure.savefig(output, dpi=200)
        plt.close(figure)
        return output


def run_mhd3d(simulation) -> MHD3DResult:
    """Execute the 3D path for a configured :class:`mhx.Simulation`."""
    shape = simulation.shape
    lengths = tuple(
        upper - lower
        for lower, upper in zip(simulation.lower, simulation.upper, strict=True)
    )
    equilibrium = simulation.equilibrium
    if not hasattr(equilibrium, "initial_fields"):
        raise TypeError(
            "equations='mhd3d' needs a 3D equilibrium with initial_fields, "
            "for example mhx.OrszagTang3DEquilibrium()"
        )

    mesh: Mesh | None = None
    device_count = simulation.device_count or 1
    if device_count > 1:
        devices = jax.devices()[:device_count]
        if shape[0] % device_count:
            raise ValueError("shape[0] must divide evenly by device_count")
        mesh = Mesh(np.asarray(devices), axis_names=("x",))

    params = MHD3DParams(
        viscosity=simulation.viscosity,
        resistivity=simulation.resistivity,
        guide_field=getattr(simulation, "guide_field", (0.0, 0.0, 0.0)),
        dissipation_order=getattr(simulation, "dissipation_order", 1),
    )
    k = mhd3d.wavevectors(shape, lengths)
    mask = mhd3d.two_thirds_mask_rfft(shape)

    velocity, magnetic = equilibrium.initial_fields(shape)
    state0 = MHD3DState(
        v_hat=mhd3d.project(mhd3d.to_spectral(velocity, mesh=mesh), k),
        b_hat=mhd3d.project(mhd3d.to_spectral(magnetic, mesh=mesh), k),
    )
    state0 = jax.tree.map(lambda leaf: shard_spectral(leaf, mesh), state0)
    decay = mhd3d.decay_rates(params, k)

    def nonlinear(state: MHD3DState) -> MHD3DState:
        return mhd3d.mhd3d_nonlinear(
            state, params, shape=shape, k=k, mask=mask, mesh=mesh
        )

    forcing = getattr(simulation, "forcing", None)
    if forcing is not None:

        def forced_nonlinear(state: MHD3DState) -> MHD3DState:
            return mhd3d.mhd3d_nonlinear(
                state, params, shape=shape, k=k, mask=mask, mesh=mesh
            )

        def nonlinear(state: MHD3DState) -> MHD3DState:
            return jax.tree.map(
                lambda r, f: r + f, forced_nonlinear(state), forcing
            )

    evolve = {"if_rk3": evolve_if_rk3, "etdrk4": evolve_etdrk4}[
        simulation.integrator
    ]
    steps = simulation.steps

    def program(initial: MHD3DState) -> MHD3DTrajectory:
        return evolve(
            initial,
            nonlinear,
            decay,
            dt=simulation.dt,
            steps=steps,
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

    initial_energy = mhd3d.energies(state0, shape=shape)
    final_state = jax.tree.map(lambda leaf: leaf[-1], trajectory.states)
    final_energy = mhd3d.energies(final_state, shape=shape)
    diagnostics: dict[str, object] = {
        "initial_total_energy": float(initial_energy["total"]),
        "final_total_energy": float(final_energy["total"]),
        "final_kinetic_energy": float(final_energy["kinetic"]),
        "final_magnetic_energy": float(final_energy["magnetic"]),
        "final_cross_helicity": float(final_energy["cross_helicity"]),
        "final_magnetic_divergence_linf": float(
            mhd3d.divergence_linf(final_state.b_hat, k)
        ),
        "final_velocity_divergence_linf": float(
            mhd3d.divergence_linf(final_state.v_hat, k)
        ),
    }
    config = {
        "equations": "mhd3d",
        "shape": list(shape),
        "viscosity": float(simulation.viscosity),
        "resistivity": float(simulation.resistivity),
        "dt": float(simulation.dt),
        "t_end": float(simulation.t_end),
        "save_every": int(simulation.save_every),
        "integrator": simulation.integrator,
        "equilibrium": getattr(type(equilibrium), "name", type(equilibrium).__name__),
    }
    return MHD3DResult(
        trajectory=trajectory,
        shape=shape,
        parameters=params,
        config=config,
        diagnostics=diagnostics,
        compile_seconds=compile_seconds,
        run_seconds=run_seconds,
        device_count=device_count,
    )
