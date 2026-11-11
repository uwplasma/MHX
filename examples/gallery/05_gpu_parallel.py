"""Split one simulation across every visible GPU.

Runs a 256 x 256 reduced-MHD simulation sharded over all visible GPUs and writes a
summary figure and the saved fields to OUTDIR. Takes seconds on a GPU; raise SHAPE
and T_END for a larger run. Requires a GPU JAX wheel. Run from the repository root:

    python examples/gallery/05_gpu_parallel.py
"""

from pathlib import Path

import jax

import mhx

# --- Input parameters ------------------------------------------------------
OUTDIR = Path("outputs/gallery/gpu_parallel")
SHAPE = (256, 256)  # grid points per side
DT = 2.0e-3  # time step
T_END = 0.2  # end time
SAVE_EVERY = 10  # steps between saved frames

# --- Run -------------------------------------------------------------------
if jax.default_backend() != "gpu":
    raise SystemExit("Install a GPU JAX wheel, then run this script again.")

gpu_count = len(mhx.available_devices("gpu"))
print(f"Running on {gpu_count} GPU(s) (progress is printed during the run)...", flush=True)
simulation = mhx.Simulation(
    shape=SHAPE,
    dt=DT,
    t_end=T_END,
    save_every=SAVE_EVERY,
    device_count=gpu_count,
)

result = simulation.run()

# --- Results ---------------------------------------------------------------
result.print_summary()

figure = result.plot(OUTDIR / "summary.png")
data = result.save(OUTDIR)

print(f"Figure: {figure}")
print(f"Data:   {data}")
