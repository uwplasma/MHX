"""Split one simulation across every visible GPU."""

from pathlib import Path

import jax

import mhx

if jax.default_backend() != "gpu":
    raise SystemExit("Install a GPU JAX wheel, then run this script again.")

output = Path("outputs/gallery/gpu_parallel")
gpu_count = len(mhx.available_devices("gpu"))

simulation = mhx.Simulation(
    shape=(256, 256),
    dt=2.0e-3,
    t_end=0.2,
    save_every=10,
    device_count=gpu_count,
)

result = simulation.run()
result.print_summary()

figure = result.plot(output / "summary.png")
data = result.save(output)

print(f"Figure: {figure}")
print(f"Data:   {data}")
