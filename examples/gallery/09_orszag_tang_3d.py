"""Run the 3D incompressible Orszag--Tang vortex at laptop scale."""

from pathlib import Path

import mhx

output = Path("outputs/gallery/orszag_tang_3d")

simulation = mhx.Simulation(
    shape=(64, 64, 64),
    equations="mhd3d",
    equilibrium=mhx.OrszagTang3DEquilibrium(beta=0.8),
    viscosity=5.0e-3,
    resistivity=5.0e-3,
    dt=2.0e-3,
    t_end=2.0,
    save_every=100,
)

result = simulation.run()
result.print_summary()

figure = result.plot(output / "summary.png")
data = result.save(output)

print(f"Figure: {figure}")
print(f"Data:   {data}")
