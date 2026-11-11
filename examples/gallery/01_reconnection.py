"""Run a small periodic double-Harris reconnection model.

Evolves a 64 x 64 resistive reduced-MHD double current sheet and writes a summary
figure and the saved fields to OUTDIR. Takes a few seconds on a CPU; raise SHAPE and
T_END for a larger run. Run from the repository root:

    python examples/gallery/01_reconnection.py
"""

from pathlib import Path

import mhx

# --- Input parameters ------------------------------------------------------
OUTDIR = Path("outputs/gallery/reconnection")
SHAPE = (64, 64)  # grid points per side
WIDTH = 0.4  # current-sheet width
PERTURBATION_AMPLITUDE = 4.0e-3
PERTURBATION_MODE = (2, 1)
RESISTIVITY = 5.0e-3
VISCOSITY = 5.0e-3
DT = 2.0e-2  # time step
T_END = 2.0  # end time
SAVE_EVERY = 10  # steps between saved frames

# --- Run -------------------------------------------------------------------
print("Running reconnection (progress is printed during the run)...", flush=True)
simulation = mhx.Simulation(
    shape=SHAPE,
    equilibrium=mhx.PeriodicDoubleHarrisEquilibrium(
        width=WIDTH,
        perturbation_amplitude=PERTURBATION_AMPLITUDE,
        perturbation_mode=PERTURBATION_MODE,
    ),
    resistivity=RESISTIVITY,
    viscosity=VISCOSITY,
    dt=DT,
    t_end=T_END,
    save_every=SAVE_EVERY,
)

result = simulation.run()

# --- Results ---------------------------------------------------------------
result.print_summary()

figure = result.plot(OUTDIR / "summary.png")
data = result.save(OUTDIR)

print(f"Figure: {figure}")
print(f"Data:   {data}")
