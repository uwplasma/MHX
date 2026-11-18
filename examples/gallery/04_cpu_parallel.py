"""Split one simulation across four local CPU devices."""

import os
from pathlib import Path

# JAX must see this setting before MHX imports JAX.
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

import mhx  # noqa: E402

output = Path("outputs/gallery/cpu_parallel")

simulation = mhx.Simulation(
    shape=(128, 128),
    dt=5.0e-3,
    t_end=0.5,
    save_every=10,
    device_count=4,
)

result = simulation.run()
result.print_summary()

figure = result.plot(output / "summary.png")
data = result.save(output)

print(f"Figure: {figure}")
print(f"Data:   {data}")
