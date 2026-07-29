"""Measure fixed-size CPU strong scaling on one to four devices."""

import json
import math
import os
from pathlib import Path
from statistics import median

# JAX must create the logical CPU devices before MHX imports JAX.
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

import jax  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

import mhx  # noqa: E402

platform = jax.default_backend()
available_count = len(mhx.available_devices())
device_counts = tuple(count for count in (1, 2, 4) if count <= available_count)
shape = (1024, 1024) if platform == "gpu" else (256, 256)
steps = 50 if platform == "gpu" else 20
dt = 5.0e-5 if platform == "gpu" else 2.0e-3

output = Path(f"outputs/gallery/strong_scaling_{platform}")
output.mkdir(parents=True, exist_ok=True)

run_times = []

for device_count in device_counts:
    samples = []
    for _ in range(3):
        result = mhx.Simulation(
            shape=shape,
            dt=dt,
            t_end=steps * dt,
            save_every=10,
            device_count=device_count,
            verbose=False,
        ).run()
        if not math.isfinite(float(result.diagnostics["final_total_energy"])):
            raise RuntimeError("simulation produced a non-finite final energy")
        samples.append(result.run_seconds)
    run_times.append(median(samples))
    print(f"{device_count} device(s): {run_times[-1]:.4f} s")

result.print_summary()

speedup = [run_times[0] / run_time for run_time in run_times]
measurements = {
    "platform": platform,
    "devices": [str(device) for device in mhx.available_devices()],
    "jax_version": jax.__version__,
    "shape": list(shape),
    "steps": steps,
    "dt": dt,
    "samples_per_count": 3,
    "finite": True,
    "device_counts": list(device_counts),
    "run_seconds": run_times,
    "speedup": speedup,
}
(output / "measurements.json").write_text(
    json.dumps(measurements, indent=2) + "\n",
    encoding="utf-8",
)

figure, axis = plt.subplots(figsize=(6.0, 4.0), constrained_layout=True)
axis.plot(device_counts, speedup, "o-", label="measured")
axis.plot(device_counts, device_counts, "--", color="0.4", label="ideal")
axis.set_xticks(device_counts)
axis.set_xlabel("CPU devices")
axis.set_ylabel("speedup")
axis.set_title(f"MHX {platform.upper()} strong scaling at fixed {shape[0]} x {shape[1]} grid")
axis.grid(alpha=0.25)
axis.legend(frameon=False)
figure.savefig(output / "strong_scaling.png", dpi=180)
plt.close(figure)

print(f"Figure: {output / 'strong_scaling.png'}")
print(f"Data:   {output / 'measurements.json'}")
