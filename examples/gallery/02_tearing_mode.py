"""Run a perturbed periodic current sheet."""

from pathlib import Path

import mhx

output = Path("outputs/gallery/tearing_mode")

simulation = mhx.Simulation(
    shape=(64, 64),
    equilibrium=mhx.CosineTearingEquilibrium(
        perturbation_amplitude=1.0e-3,
    ),
    resistivity=1.0e-2,
    viscosity=1.0e-2,
    dt=5.0e-3,
    t_end=0.5,
    save_every=10,
)

result = simulation.run()
result.print_summary()

figure = result.plot(output / "summary.png")
data = result.save(output)

print(f"Figure: {figure}")
print(f"Data:   {data}")
