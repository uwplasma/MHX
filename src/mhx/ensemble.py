"""Independent reduced-MHD ensembles on one or more JAX processes."""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec
from rich.console import Console
from rich.table import Table

from mhx.equations.reduced_mhd import reduced_mhd_rhs_spectral
from mhx.grids import CartesianGrid
from mhx.io import write_reduced_mhd_trajectory_npz
from mhx.parallel import make_device_mesh, shard_batch
from mhx.physics import Equilibrium
from mhx.state import ReducedMHDParams, ReducedMHDState, ReducedMHDTrajectory
from mhx.time_integrators import evolve_rk4

if TYPE_CHECKING:
    from mhx.simulation import Simulation


def _host_array(array: jax.Array) -> np.ndarray:
    """Gather a global JAX array when a result spans several processes."""
    if array.is_fully_addressable:
        return np.asarray(array)
    from jax.experimental import multihost_utils

    return np.asarray(multihost_utils.process_allgather(array, tiled=True))


@dataclass(frozen=True)
class EnsembleResult:
    """Trajectories and timings for an independent simulation ensemble.

    The leading field axis selects the case; the next axis selects saved time.
    """

    trajectory: ReducedMHDTrajectory
    grid: CartesianGrid
    config: dict[str, object]
    compile_seconds: float
    run_seconds: float
    device_count: int

    @property
    def case_count(self) -> int:
        """Return the number of independent simulations."""
        return int(self.trajectory.states.psi.shape[0])

    @property
    def final_states(self) -> ReducedMHDState:
        """Return the final magnetic flux and vorticity for every case."""
        return ReducedMHDState(
            psi=self.trajectory.states.psi[:, -1],
            omega=self.trajectory.states.omega[:, -1],
        )

    def print_summary(self, *, console: Console | None = None) -> None:
        """Print ensemble size, timing, and throughput."""
        output = console or Console()
        table = Table(title="MHX ensemble", show_header=False)
        table.add_column("Quantity", style="cyan")
        table.add_column("Value", justify="right")
        table.add_row("Cases", str(self.case_count))
        table.add_row("Devices", str(self.device_count))
        table.add_row("Compile time", f"{self.compile_seconds:.3f} s")
        table.add_row("Run time", f"{self.run_seconds:.3f} s")
        table.add_row("Throughput", f"{self.case_count / self.run_seconds:.2f} cases/s")
        output.print(table)

    def save(self, path: str | Path) -> Path:
        """Save each case as one standard MHX trajectory file."""
        output = Path(path)
        psi = _host_array(self.trajectory.states.psi)
        omega = _host_array(self.trajectory.states.omega)
        times = _host_array(self.trajectory.times)
        if jax.process_index() != 0:
            return output
        output.mkdir(parents=True, exist_ok=True)
        for case in range(self.case_count):
            trajectory = ReducedMHDTrajectory(
                times=times,
                states=ReducedMHDState(psi=psi[case], omega=omega[case]),
            )
            write_reduced_mhd_trajectory_npz(
                output / f"case_{case:03d}.npz",
                trajectory=trajectory,
                config={**self.config, "case": case},
                diagnostics={"case": case},
            )
        return output

    def plot(self, path: str | Path) -> Path:
        """Plot the final magnetic flux for up to eight ensemble cases."""
        import matplotlib.pyplot as plt

        output = Path(path)
        final_flux = _host_array(self.final_states.psi)
        if jax.process_index() != 0:
            return output
        output.parent.mkdir(parents=True, exist_ok=True)

        shown_cases = min(self.case_count, 8)
        columns = min(shown_cases, 4)
        rows = math.ceil(shown_cases / columns)
        figure, axes = plt.subplots(
            rows,
            columns,
            figsize=(3.2 * columns, 2.8 * rows),
            squeeze=False,
            constrained_layout=True,
        )
        extent = (
            self.grid.lower[0],
            self.grid.upper[0],
            self.grid.lower[1],
            self.grid.upper[1],
        )
        for case, axis in enumerate(axes.flat):
            if case >= shown_cases:
                axis.set_visible(False)
                continue
            image = axis.imshow(
                final_flux[case].T,
                origin="lower",
                extent=extent,
                aspect="auto",
                cmap="viridis",
            )
            axis.set_title(f"Case {case}")
            axis.set_xlabel("x")
            axis.set_ylabel("y")
            figure.colorbar(image, ax=axis, shrink=0.75)
        figure.savefig(output, dpi=180)
        plt.close(figure)
        return output


def run_ensemble(
    simulation: Simulation,
    equilibria: Sequence[Equilibrium],
) -> EnsembleResult:
    """Run independent initial conditions over a JAX device mesh."""
    if simulation.integrator != "rk4":
        raise ValueError("run_ensemble currently supports the RK4 integrator")
    cases = tuple(equilibria)
    if not cases:
        raise ValueError("equilibria must contain at least one case")

    device_count = simulation.device_count or 1
    if len(cases) % device_count:
        raise ValueError("ensemble size must be divisible by device_count")
    if jax.process_count() > 1 and device_count != jax.device_count():
        raise ValueError("a distributed ensemble must use every global JAX device")
    if len(cases) % jax.process_count():
        raise ValueError("ensemble size must divide evenly over JAX processes")

    grid = CartesianGrid(
        shape=simulation.shape,
        lower=simulation.lower,
        upper=simulation.upper,
    )
    parameters = ReducedMHDParams(
        resistivity=simulation.resistivity,
        viscosity=simulation.viscosity,
    )
    mesh = make_device_mesh(device_count)
    local_count = len(cases) // jax.process_count()
    start = jax.process_index() * local_count
    local_cases = cases[start : start + local_count]
    local_fields = []
    for case in local_cases:
        state = case.initial_state(grid)
        local_fields.append(
            np.stack((np.asarray(state.psi), np.asarray(state.omega)))
        )
    host_fields = np.stack(local_fields)
    sharding = NamedSharding(
        mesh,
        PartitionSpec("device", None, None, None),
    )
    if jax.process_count() == 1:
        initial_fields = jax.device_put(host_fields, sharding)
    else:
        initial_fields = jax.make_array_from_process_local_data(
            sharding,
            host_fields,
            global_shape=(len(cases), 2, *simulation.shape),
        )

    def single_rhs(field_pair: jax.Array) -> jax.Array:
        rhs = reduced_mhd_rhs_spectral(
            ReducedMHDState(psi=field_pair[0], omega=field_pair[1]),
            parameters,
            lengths=grid.lengths,
            terms=simulation.terms,
            dealiasing=simulation.dealiasing,
        )
        return jnp.stack((rhs.psi, rhs.omega))

    batched_rhs = jax.vmap(single_rhs)

    def integrate_local(fields: jax.Array) -> jax.Array:
        fields_hat = jnp.fft.fftn(fields, axes=(-2, -1))
        trajectory_hat = evolve_rk4(
            fields_hat,
            batched_rhs,
            dt=simulation.dt,
            steps=simulation.steps,
            save_every=simulation.save_every,
        )
        physical = jnp.fft.ifftn(
            trajectory_hat.states,
            axes=(-2, -1),
        ).real
        return jnp.moveaxis(physical, 1, 0)

    integrate = shard_batch(
        integrate_local,
        mesh=mesh,
        input_rank=4,
        output_rank=5,
    )
    compile_start = time.perf_counter()
    executable = jax.jit(integrate).lower(initial_fields).compile()
    compile_seconds = time.perf_counter() - compile_start

    run_start = time.perf_counter()
    fields = executable(initial_fields)
    jax.block_until_ready(fields)
    run_seconds = time.perf_counter() - run_start

    stride = min(simulation.save_every, simulation.steps)
    step_numbers = list(range(stride, simulation.steps + 1, stride))
    if step_numbers[-1] != simulation.steps:
        step_numbers.append(simulation.steps)
    trajectory = ReducedMHDTrajectory(
        times=jnp.asarray(step_numbers) * simulation.dt,
        states=ReducedMHDState(psi=fields[:, :, 0], omega=fields[:, :, 1]),
    )
    result = EnsembleResult(
        trajectory=trajectory,
        grid=grid,
        config={
            **simulation._config(device_count),
            "ensemble_size": len(cases),
            "parallel_axis": "case",
        },
        compile_seconds=compile_seconds,
        run_seconds=run_seconds,
        device_count=device_count,
    )
    if simulation.verbose and jax.process_index() == 0:
        result.print_summary()
    return result
