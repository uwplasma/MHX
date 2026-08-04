"""Run backward Euler with SOLVAX Newton--Krylov solves."""

from pathlib import Path

import mhx

output = Path("outputs/gallery/implicit_step")

simulation = mhx.Simulation(
    shape=(32, 32),
    equilibrium=mhx.CosineTearingEquilibrium(
        perturbation_amplitude=1.0e-3,
    ),
    resistivity=2.0e-2,
    viscosity=2.0e-2,
    dt=1.0e-2,
    t_end=5.0e-2,
    save_every=1,
    integrator="backward_euler",
)

result = simulation.run()
result.print_summary()

figure = result.plot(output / "summary.png")
data = result.save(output)

print(f"Figure: {figure}")
print(f"Data:   {data}")
