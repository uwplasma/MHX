"""Split one simulation across four local CPU devices.

Runs a 128 x 128 reduced-MHD simulation with the grid sharded over four logical CPU
devices and writes a summary figure and the saved fields to OUTDIR. Takes under a
minute on a CPU; raise SHAPE and T_END for a larger run. Run from the repository
root:

    python examples/gallery/04_cpu_parallel.py
"""

import os
from pathlib import Path

# JAX must see this setting before MHX imports JAX (an XLA setting, not an input).
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

import mhx  # noqa: E402

# --- Input parameters ------------------------------------------------------
OUTDIR = Path("outputs/gallery/cpu_parallel")
SHAPE = (128, 128)  # grid points per side
DT = 5.0e-3  # time step
T_END = 0.5  # end time
SAVE_EVERY = 10  # steps between saved frames
DEVICE_COUNT = 4  # logical CPU devices to shard over

# --- Run -------------------------------------------------------------------
print(f"Running on {DEVICE_COUNT} CPU devices (progress is printed during the run)...", flush=True)
simulation = mhx.Simulation(
    shape=SHAPE,
    dt=DT,
    t_end=T_END,
    save_every=SAVE_EVERY,
    device_count=DEVICE_COUNT,
)

result = simulation.run()

# --- Results ---------------------------------------------------------------
result.print_summary()

figure = result.plot(OUTDIR / "summary.png")
data = result.save(OUTDIR)

print(f"Figure: {figure}")
print(f"Data:   {data}")
