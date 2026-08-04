"""Run one reconnection ensemble across JAX processes and devices."""

from pathlib import Path

from mhx.parallel import initialize_distributed

# Call this before querying devices. Slurm and Open MPI provide the process
# settings that JAX needs.
initialize_distributed()

import jax  # noqa: E402

import mhx  # noqa: E402

output = Path("outputs/gallery/multi_process")
case_count = 2 * jax.device_count()

equilibria = tuple(
    mhx.PeriodicDoubleHarrisEquilibrium(
        width=0.4,
        perturbation_amplitude=1.0e-3 * (1.0 + 0.05 * case),
        perturbation_mode=(2, 1),
    )
    for case in range(case_count)
)

simulation = mhx.Simulation(
    shape=(128, 128),
    dt=2.0e-3,
    t_end=4.0e-2,
    save_every=5,
    device_count=jax.device_count(),
    verbose=False,
)

result = simulation.run_ensemble(equilibria)

if jax.process_index() == 0:
    result.print_summary()

# Every process enters these calls because gathering a global array is a
# collective operation. Only process 0 writes the shared files.
figure = result.plot(output / "final_flux.png")
data = result.save(output / "cases")

if jax.process_index() == 0:
    print(f"Figure: {figure}")
    print(f"Data:   {data}")
