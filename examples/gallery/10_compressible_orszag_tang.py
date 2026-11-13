"""Run a subsonic compressible Orszag--Tang vortex in a thin 3D box."""

from pathlib import Path

import mhx

output = Path("outputs/gallery/compressible_orszag_tang")

simulation = mhx.Simulation(
    shape=(64, 64, 4),
    equations="compressible",
    equilibrium=mhx.OrszagTang3DEquilibrium(beta=0.8),
    sound_speed=5.0,
    viscosity=5.0e-3,
    bulk_viscosity=5.0e-3,
    resistivity=5.0e-3,
    dt=1.0e-3,
    t_end=1.0,
    save_every=100,
)

result = simulation.run()
result.print_summary()

figure = result.plot(output / "summary.png")
data = result.save(output)

print(f"Figure: {figure}")
print(f"Data:   {data}")
