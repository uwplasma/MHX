"""A small public API for running reduced-MHD simulations."""

from __future__ import annotations

import math
import time
import warnings
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import jax
import numpy as np
from rich.console import Console
from rich.table import Table

from mhx.diagnostics import compute_reduced_mhd_diagnostics, trajectory_energies
from mhx.ensemble import EnsembleResult, run_ensemble
from mhx.equations.reduced_mhd import (
    current_density,
    reduced_mhd_rhs,
    reduced_mhd_rhs_spectral,
    to_physical_trajectory,
    to_spectral_state,
)
from mhx.grids import CartesianGrid
from mhx.io import write_reduced_mhd_trajectory_npz
from mhx.numerics import spectral_diffusion_preconditioner
from mhx.parallel import make_spatial_sharding, shard_state
from mhx.physics import Equilibrium, PeriodicDoubleHarrisEquilibrium, PhysicsTerm
from mhx.state import ReducedMHDParams, ReducedMHDState, ReducedMHDTrajectory
from mhx.time_integrators import evolve_backward_euler, evolve_rk4

Integrator = Literal["rk4", "backward_euler"]


@dataclass(frozen=True)
class SimulationResult:
    """Fields, diagnostics, and timing data from one simulation.

    Use :meth:`print_summary`, :meth:`plot`, and :meth:`save` after a run.
    The full saved field history is available through :attr:`trajectory`.
    """

    trajectory: ReducedMHDTrajectory
    initial_state: ReducedMHDState
    grid: CartesianGrid
    parameters: ReducedMHDParams
    config: dict[str, object]
    diagnostics: dict[str, object]
    compile_seconds: float
    run_seconds: float
    device_count: int

    @property
    def final_state(self) -> ReducedMHDState:
        """Return the last saved magnetic-flux and vorticity fields."""
        return ReducedMHDState(
            psi=self.trajectory.states.psi[-1],
            omega=self.trajectory.states.omega[-1],
        )

    @property
    def final_time(self) -> float:
        """Return the time of the last saved state."""
        return float(self.trajectory.times[-1])

    def print_summary(self, *, console: Console | None = None) -> None:
        """Print the main run results as a compact Rich table."""
        output = console or Console()
        table = Table(title="MHX result", show_header=False)
        table.add_column("Quantity", style="cyan")
        table.add_column("Value", justify="right")
        table.add_row("Final time", f"{self.final_time:.6g}")
        table.add_row("Saved states", str(self.trajectory.times.shape[0]))
        table.add_row("Devices", str(self.device_count))
        table.add_row("Compile time", f"{self.compile_seconds:.3f} s")
        table.add_row("Run time", f"{self.run_seconds:.3f} s")
        table.add_row(
            "Initial energy",
            f"{float(self.diagnostics['initial_total_energy']):.6e}",
        )
        table.add_row(
            "Final energy",
            f"{float(self.diagnostics['final_total_energy']):.6e}",
        )
        table.add_row(
            "Max |div B|",
            f"{float(self.diagnostics['final_magnetic_divergence_linf']):.3e}",
        )
        output.print(table)

    def save(self, path: str | Path) -> Path:
        """Save all retained fields and metadata to one compressed NPZ file.

        If ``path`` has no ``.npz`` suffix, MHX writes ``path/trajectory.npz``.
        """
        output = Path(path)
        if output.suffix != ".npz":
            output = output / "trajectory.npz"
        return write_reduced_mhd_trajectory_npz(
            output,
            trajectory=self.trajectory,
            config=self.config,
            diagnostics=self.diagnostics,
        )

    def plot(self, path: str | Path) -> Path:
        """Write a four-panel summary of the fields and energy history.

        The second panel shows the island flux, the deviation of the final
        flux from its y average. The total flux is visually dominated by the
        equilibrium, so the island view is the one that shows reconnection.
        """
        import matplotlib.pyplot as plt

        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        initial_flux = np.asarray(self.initial_state.psi)
        final_flux = np.asarray(self.final_state.psi)
        island_flux = final_flux - final_flux.mean(axis=1, keepdims=True)
        final_current = np.asarray(current_density(self.final_state.psi, lengths=self.grid.lengths))
        energies = trajectory_energies(self.trajectory, lengths=self.grid.lengths)
        extent = (
            self.grid.lower[0],
            self.grid.upper[0],
            self.grid.lower[1],
            self.grid.upper[1],
        )

        figure, axes = plt.subplots(2, 2, figsize=(9.0, 7.0), constrained_layout=True)
        initial_image = axes[0, 0].imshow(
            initial_flux.T,
            origin="lower",
            extent=extent,
            aspect="auto",
            cmap="viridis",
        )
        axes[0, 0].set_title("Initial magnetic flux")
        figure.colorbar(initial_image, ax=axes[0, 0], shrink=0.8)

        island_limit = max(float(np.max(np.abs(island_flux))), np.finfo(float).eps)
        island_image = axes[0, 1].imshow(
            island_flux.T,
            origin="lower",
            extent=extent,
            aspect="auto",
            cmap="RdBu_r",
            vmin=-island_limit,
            vmax=island_limit,
        )
        axes[0, 1].contour(
            np.linspace(extent[0], extent[1], final_flux.shape[0]),
            np.linspace(extent[2], extent[3], final_flux.shape[1]),
            final_flux.T,
            colors="black",
            linewidths=0.4,
            levels=12,
        )
        axes[0, 1].set_title(f"Island flux at t = {self.final_time:.3g}")
        figure.colorbar(island_image, ax=axes[0, 1], shrink=0.8)

        current_limit = max(float(np.max(np.abs(final_current))), np.finfo(float).eps)
        current_image = axes[1, 0].imshow(
            final_current.T,
            origin="lower",
            extent=extent,
            aspect="auto",
            cmap="RdBu_r",
            vmin=-current_limit,
            vmax=current_limit,
        )
        axes[1, 0].set_title("Final current density")
        figure.colorbar(current_image, ax=axes[1, 0], shrink=0.8)

        axes[1, 1].plot(energies["time"], energies["magnetic"], label="magnetic")
        axes[1, 1].plot(energies["time"], energies["kinetic"], label="kinetic")
        axes[1, 1].plot(energies["time"], energies["total"], label="total")
        axes[1, 1].set_title("Mean energy")
        axes[1, 1].set_xlabel("time")
        axes[1, 1].legend(frameon=False)

        for axis in axes.flat[:3]:
            axis.set_xlabel("x")
            axis.set_ylabel("y")
        figure.savefig(output, dpi=180)
        plt.close(figure)
        return output


@dataclass(frozen=True)
class Simulation:
    """Configure and run a periodic two-dimensional reduced-MHD model.

    The default initial condition is a seeded double-Harris current sheet.
    MHX builds the physics operators. SOLVAX supplies the nonlinear and Krylov
    solvers used by the backward-Euler option.

    Args:
        shape: Global grid points in the two periodic directions.
        lower: Lower coordinate bound in each direction.
        upper: Upper coordinate bound in each direction.
        equilibrium: Object that builds the initial state.
        resistivity: Magnetic diffusivity.
        viscosity: Viscous diffusivity.
        dt: Fixed time step.
        t_end: Final simulation time.
        save_every: Save one state after this many steps.
        integrator: ``"rk4"`` or ``"backward_euler"``.
        dealiasing: Spectral product filter. Use ``"two_thirds"`` for runs.
        device_count: Number of JAX devices. Leave unset for one device.
        verbose: Print configuration and timing information.
        terms: Extra MHX physics terms applied to the right-hand side.
    """

    shape: tuple[int, int] = (64, 64)
    lower: tuple[float, float] = (0.0, 0.0)
    upper: tuple[float, float] = (2.0 * math.pi, 2.0 * math.pi)
    equilibrium: Equilibrium = field(
        default_factory=lambda: PeriodicDoubleHarrisEquilibrium(
            width=0.4,
            perturbation_amplitude=1.0e-3,
            perturbation_mode=(2, 1),
        )
    )
    resistivity: float = 5.0e-3
    viscosity: float = 5.0e-3
    dt: float = 2.0e-2
    t_end: float = 2.0
    save_every: int = 10
    integrator: str = "rk4"
    dealiasing: str = "two_thirds"
    device_count: int | None = None
    verbose: bool = True
    terms: tuple[PhysicsTerm, ...] = ()
    equations: str = "reduced_mhd"
    guide_field: tuple[float, float, float] = (0.0, 0.0, 0.0)
    dissipation_order: int = 1
    forcing: object | None = None
    sound_speed: float = 1.0
    bulk_viscosity: float = 0.0

    def __post_init__(self) -> None:
        """Reject invalid settings before JAX starts a compilation."""
        if self.equations not in ("reduced_mhd", "mhd3d", "compressible"):
            raise ValueError(
                "equations must be 'reduced_mhd', 'mhd3d', or 'compressible'"
            )
        if self.equations in ("mhd3d", "compressible"):
            if len(self.shape) != 3 or any(points < 4 for points in self.shape):
                raise ValueError(
                    f"equations={self.equations!r} needs three grid sizes of "
                    "at least 4"
                )
            if self.integrator == "rk4":
                # The 3D production stepper; keeps the 2D default untouched.
                object.__setattr__(self, "integrator", "if_rk3")
            if self.integrator not in ("if_rk3", "etdrk4"):
                raise ValueError("3D integrator must be 'if_rk3' or 'etdrk4'")
            if len(self.lower) == 2:
                object.__setattr__(self, "lower", (*self.lower, 0.0))
                object.__setattr__(self, "upper", (*self.upper, 2.0 * math.pi))
            self._validate_time_settings()
            return
        if len(self.shape) != 2 or any(points < 4 for points in self.shape):
            raise ValueError("shape must contain two grid sizes of at least 4")
        if self.integrator not in ("rk4", "backward_euler"):
            raise ValueError("integrator must be 'rk4' or 'backward_euler'")
        self._validate_time_settings()

    def _validate_time_settings(self) -> None:
        """Checks shared by the 2D and 3D paths."""
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")
        if self.t_end <= 0.0:
            raise ValueError("t_end must be positive")
        if self.save_every < 1:
            raise ValueError("save_every must be at least 1")
        if self.dealiasing not in ("none", "two_thirds"):
            raise ValueError("dealiasing must be 'none' or 'two_thirds'")
        step_count = self.t_end / self.dt
        if not math.isclose(step_count, round(step_count), rel_tol=1.0e-10, abs_tol=1.0e-10):
            raise ValueError("t_end must be an integer multiple of dt")

    @property
    def steps(self) -> int:
        """Return the fixed number of time steps."""
        return int(round(self.t_end / self.dt))

    def run(self) -> SimulationResult:
        """Compile and run the configured simulation.

        Returns:
            A :class:`SimulationResult` with fields, diagnostics, and timings,
            or an :class:`mhx.simulation3d.MHD3DResult` for ``mhd3d`` runs.
        """
        if self.equations == "mhd3d":
            from mhx.simulation3d import run_mhd3d

            return run_mhd3d(self)
        if self.equations == "compressible":
            from mhx.simulation_compressible import run_compressible

            return run_compressible(self)
        if self.integrator == "backward_euler" and not jax.config.jax_enable_x64:
            warnings.warn(
                "backward_euler targets Newton tolerances near 1e-9, which "
                "float32 usually cannot reach, so the run may report "
                "implicit_converged=False. Set JAX_ENABLE_X64=1 for implicit "
                "runs.",
                UserWarning,
                stacklevel=2,
            )
        console = Console()
        grid = CartesianGrid(
            shape=self.shape,
            lower=self.lower,
            upper=self.upper,
        )
        parameters = ReducedMHDParams(
            resistivity=self.resistivity,
            viscosity=self.viscosity,
        )
        initial_state = self.equilibrium.initial_state(grid)
        effective_device_count = 1
        if self.device_count is not None:
            sharding_plan = make_spatial_sharding(self.shape, self.device_count)
            initial_state = shard_state(initial_state, sharding_plan)
            effective_device_count = sharding_plan.device_count

        if self.verbose:
            self._print_setup(console, effective_device_count)

        def rhs(state: ReducedMHDState) -> ReducedMHDState:
            return reduced_mhd_rhs(
                state,
                parameters,
                lengths=grid.lengths,
                terms=self.terms,
                dealiasing=self.dealiasing,
            )

        if self.integrator == "rk4":
            def integrate(state: ReducedMHDState):
                state_hat = to_spectral_state(state)

                def spectral_rhs(active_state_hat: ReducedMHDState) -> ReducedMHDState:
                    return reduced_mhd_rhs_spectral(
                        active_state_hat,
                        parameters,
                        lengths=grid.lengths,
                        terms=self.terms,
                        dealiasing=self.dealiasing,
                    )

                trajectory_hat = evolve_rk4(
                    state_hat,
                    spectral_rhs,
                    dt=self.dt,
                    steps=self.steps,
                    save_every=self.save_every,
                )
                return to_physical_trajectory(trajectory_hat)

        else:
            preconditioner = spectral_diffusion_preconditioner(
                parameters,
                lengths=grid.lengths,
                dt=self.dt,
            )

            def integrate(state: ReducedMHDState):
                return evolve_backward_euler(
                    state,
                    rhs,
                    dt=self.dt,
                    steps=self.steps,
                    save_every=self.save_every,
                    preconditioner=preconditioner,
                )

        if self.verbose:
            console.print("[cyan]Compile[/cyan] JAX program")
        compile_start = time.perf_counter()
        executable = jax.jit(integrate).lower(initial_state).compile()
        compile_seconds = time.perf_counter() - compile_start

        if self.verbose:
            console.print("[cyan]Run[/cyan] time integration")
        run_start = time.perf_counter()
        raw_result = executable(initial_state)
        jax.block_until_ready(raw_result)
        trajectory = raw_result if self.integrator == "rk4" else raw_result.trajectory
        run_seconds = time.perf_counter() - run_start
        diagnostics = compute_reduced_mhd_diagnostics(
            trajectory,
            initial_state=initial_state,
            lengths=grid.lengths,
            quantities=("energy", "divergence_error"),
            mode=(0, 1),
            fit_time_window=None,
        )
        if self.integrator == "backward_euler":
            diagnostics.update(
                {
                    "implicit_converged": bool(np.all(np.asarray(raw_result.converged))),
                    "implicit_linear_converged": bool(
                        np.all(np.asarray(raw_result.linear_converged))
                    ),
                    "implicit_nonlinear_iterations": int(
                        np.sum(np.asarray(raw_result.nonlinear_iterations))
                    ),
                    "implicit_linear_iterations": int(
                        np.sum(np.asarray(raw_result.linear_iterations))
                    ),
                }
            )
        config = self._config(effective_device_count)
        result = SimulationResult(
            trajectory=trajectory,
            initial_state=initial_state,
            grid=grid,
            parameters=parameters,
            config=config,
            diagnostics=diagnostics,
            compile_seconds=compile_seconds,
            run_seconds=run_seconds,
            device_count=effective_device_count,
        )
        if self.verbose:
            console.print(
                f"[green]Done[/green] {self.steps} steps in {run_seconds:.3f} s "
                f"after {compile_seconds:.3f} s compilation"
            )
        return result

    def run_ensemble(self, equilibria: Sequence[Equilibrium]) -> EnsembleResult:
        """Run independent initial conditions in parallel over devices."""
        return run_ensemble(self, equilibria)

    def _print_setup(self, console: Console, device_count: int) -> None:
        console.rule("[bold cyan]MHX reduced-MHD simulation")
        table = Table(show_header=False, box=None)
        table.add_column("Setting", style="cyan")
        table.add_column("Value")
        table.add_row("Grid", f"{self.shape[0]} x {self.shape[1]}")
        table.add_row("Integrator", self.integrator)
        table.add_row("Time", f"0 to {self.t_end:g} in {self.steps} steps")
        table.add_row("Resistivity", f"{self.resistivity:g}")
        table.add_row("Viscosity", f"{self.viscosity:g}")
        table.add_row("Devices", str(device_count))
        console.print(table)

    def _config(self, device_count: int) -> dict[str, object]:
        parameters = (
            asdict(self.equilibrium) if hasattr(self.equilibrium, "__dataclass_fields__") else {}
        )
        return {
            "model": "reduced_mhd",
            "shape": list(self.shape),
            "lower": list(self.lower),
            "upper": list(self.upper),
            "equilibrium": {
                "name": getattr(self.equilibrium, "name", type(self.equilibrium).__name__),
                "parameters": parameters,
            },
            "resistivity": self.resistivity,
            "viscosity": self.viscosity,
            "dt": self.dt,
            "t_end": self.t_end,
            "steps": self.steps,
            "save_every": self.save_every,
            "integrator": self.integrator,
            "dealiasing": self.dealiasing,
            "device_count": device_count,
            "platform": jax.default_backend(),
            "x64": bool(jax.config.x64_enabled),
        }
