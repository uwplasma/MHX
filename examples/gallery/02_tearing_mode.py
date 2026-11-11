"""Run a perturbed periodic current sheet.

Evolves a 64 x 64 cosine tearing equilibrium with a small flux perturbation and
writes a summary figure and the saved fields to OUTDIR. Takes a few seconds on a
CPU; raise SHAPE and T_END for a longer run. Run from the repository root:

    python examples/gallery/02_tearing_mode.py
"""

from pathlib import Path

import mhx

# --- Input parameters ------------------------------------------------------
OUTDIR = Path("outputs/gallery/tearing_mode")
SHAPE = (64, 64)  # grid points per side
PERTURBATION_AMPLITUDE = 1.0e-3
RESISTIVITY = 1.0e-2
VISCOSITY = 1.0e-2
DT = 5.0e-3  # time step
T_END = 0.5  # end time
SAVE_EVERY = 10  # steps between saved frames

# --- Run -------------------------------------------------------------------
print("Running tearing mode (progress is printed during the run)...", flush=True)
simulation = mhx.Simulation(
    shape=SHAPE,
    equilibrium=mhx.CosineTearingEquilibrium(
        perturbation_amplitude=PERTURBATION_AMPLITUDE,
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
