"""Run one reconnection ensemble across JAX processes and devices.

Evolves an ensemble of 2 * (device count) double-Harris reconnection cases, split
over every JAX device in every process, and writes the final flux figure and the
per-case fields to OUTDIR. Launch it under Slurm or Open MPI (for example
`srun -n 2 python examples/gallery/07_multi_process.py`); it also runs as a single
process. Takes under a minute; raise SHAPE and T_END for a larger run.
"""

from pathlib import Path

from mhx.parallel import initialize_distributed

# Call this before querying devices. Slurm and Open MPI provide the process
# settings that JAX needs.
initialize_distributed()

import jax  # noqa: E402

import mhx  # noqa: E402

# --- Input parameters ------------------------------------------------------
OUTDIR = Path("outputs/gallery/multi_process")
SHAPE = (128, 128)  # grid points per side
DT = 2.0e-3  # time step
T_END = 4.0e-2  # end time
SAVE_EVERY = 5  # steps between saved frames
CASES_PER_DEVICE = 2  # ensemble members per device

# --- Run -------------------------------------------------------------------
case_count = CASES_PER_DEVICE * jax.device_count()

equilibria = tuple(
    mhx.PeriodicDoubleHarrisEquilibrium(
        width=0.4,
        perturbation_amplitude=1.0e-3 * (1.0 + 0.05 * case),
        perturbation_mode=(2, 1),
    )
    for case in range(case_count)
)

simulation = mhx.Simulation(
    shape=SHAPE,
    dt=DT,
    t_end=T_END,
    save_every=SAVE_EVERY,
    device_count=jax.device_count(),
    verbose=False,
)

if jax.process_index() == 0:
    print(
        f"Running {case_count} cases on {jax.device_count()} device(s) in "
        f"{jax.process_count()} process(es)...",
        flush=True,
    )
result = simulation.run_ensemble(equilibria)

# --- Results ---------------------------------------------------------------
if jax.process_index() == 0:
    result.print_summary()

# Every process enters these calls because gathering a global array is a
# collective operation. Only process 0 writes the shared files.
figure = result.plot(OUTDIR / "final_flux.png")
data = result.save(OUTDIR / "cases")

if jax.process_index() == 0:
    print(f"Figure: {figure}")
    print(f"Data:   {data}")
