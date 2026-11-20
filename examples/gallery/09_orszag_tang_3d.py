"""Run the 3D incompressible Orszag--Tang vortex at laptop scale.

Evolves a 64^3 incompressible MHD Orszag--Tang vortex to t = 2 and writes a summary
figure and the saved fields to OUTDIR. Takes about a minute on a laptop CPU; lower
SHAPE for a quick check or raise it for a larger run. Run from the repository root:

    python examples/gallery/09_orszag_tang_3d.py
"""

from pathlib import Path

import mhx

# --- Input parameters ------------------------------------------------------
OUTDIR = Path("outputs/gallery/orszag_tang_3d")
SHAPE = (64, 64, 64)  # grid points per side
BETA = 0.8
VISCOSITY = 5.0e-3
RESISTIVITY = 5.0e-3
DT = 2.0e-3  # time step
T_END = 2.0  # end time
SAVE_EVERY = 100  # steps between saved frames

# --- Run -------------------------------------------------------------------
print("Running 3D Orszag--Tang (progress is printed during the run)...", flush=True)
simulation = mhx.Simulation(
    shape=SHAPE,
    equations="mhd3d",
    equilibrium=mhx.OrszagTang3DEquilibrium(beta=BETA),
    viscosity=VISCOSITY,
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
