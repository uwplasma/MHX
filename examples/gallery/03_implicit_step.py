"""Run backward Euler with SOLVAX Newton--Krylov solves.

Advances a 32 x 32 tearing equilibrium with the implicit backward-Euler integrator
and writes a summary figure and the saved fields to OUTDIR. Takes a few seconds on a
CPU; raise SHAPE and T_END for a larger run. Run from the repository root:

    python examples/gallery/03_implicit_step.py
"""

from pathlib import Path

import mhx

# --- Input parameters ------------------------------------------------------
OUTDIR = Path("outputs/gallery/implicit_step")
SHAPE = (32, 32)  # grid points per side
PERTURBATION_AMPLITUDE = 1.0e-3
RESISTIVITY = 2.0e-2
VISCOSITY = 2.0e-2
DT = 1.0e-2  # time step
T_END = 5.0e-2  # end time
SAVE_EVERY = 1  # steps between saved frames
INTEGRATOR = "backward_euler"

# --- Run -------------------------------------------------------------------
print("Running implicit steps (progress is printed during the run)...", flush=True)
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
    integrator=INTEGRATOR,
)

result = simulation.run()

# --- Results ---------------------------------------------------------------
result.print_summary()

figure = result.plot(OUTDIR / "summary.png")
data = result.save(OUTDIR)

print(f"Figure: {figure}")
print(f"Data:   {data}")
