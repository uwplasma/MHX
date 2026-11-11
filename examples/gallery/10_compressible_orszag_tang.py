"""Run a subsonic compressible Orszag--Tang vortex in a thin 3D box.

Evolves a 64 x 64 x 4 compressible MHD Orszag--Tang vortex (Mach near 0.3) to t = 1
and writes a summary figure and the saved fields to OUTDIR. Takes about a minute on a
CPU; raise SHAPE and T_END for a larger run. Run from the repository root:

    python examples/gallery/10_compressible_orszag_tang.py
"""

from pathlib import Path

import mhx

# --- Input parameters ------------------------------------------------------
OUTDIR = Path("outputs/gallery/compressible_orszag_tang")
SHAPE = (64, 64, 4)  # grid points per side
BETA = 0.8
SOUND_SPEED = 5.0
VISCOSITY = 5.0e-3
BULK_VISCOSITY = 5.0e-3
RESISTIVITY = 5.0e-3
DT = 1.0e-3  # time step
T_END = 1.0  # end time
SAVE_EVERY = 100  # steps between saved frames

# --- Run -------------------------------------------------------------------
print("Running compressible Orszag--Tang (progress is printed during the run)...", flush=True)
simulation = mhx.Simulation(
    shape=SHAPE,
    equations="compressible",
    equilibrium=mhx.OrszagTang3DEquilibrium(beta=BETA),
    sound_speed=SOUND_SPEED,
    viscosity=VISCOSITY,
    bulk_viscosity=BULK_VISCOSITY,
    resistivity=RESISTIVITY,
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
