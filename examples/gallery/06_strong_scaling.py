"""Measure strong scaling for a fixed reconnection ensemble.

Runs the same ensemble of CASES double-Harris reconnection cases on 1, 2 and 4
devices (as many as are visible and divide CASES), times each run SAMPLES times after
compilation, and writes measurements.json and strong_scaling.png to OUTDIR. On a CPU
it uses four logical devices and takes a few minutes; on a GPU it uses a larger grid.
Raise SIDE, STEPS and SAMPLES for steadier numbers. Run from the repository root:

    python examples/gallery/06_strong_scaling.py
"""

import json
import math
import os
from pathlib import Path
from statistics import median

# JAX must create the logical CPU devices before MHX imports JAX (an XLA setting,
# not an input).
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

import jax  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

import mhx  # noqa: E402

# --- Input parameters ------------------------------------------------------
SIDE = None  # grid points per side; None -> 1024 on GPU, 256 on CPU
STEPS = None  # time steps per run; None -> 50 on GPU, 20 on CPU
SAMPLES = 3  # timed repeats per device count
CASES = 4  # independent ensemble members (the parallel axis)
OUTDIR = None  # None -> outputs/gallery/strong_scaling_<platform>

# --- Run -------------------------------------------------------------------
platform = jax.default_backend()
available_count = len(mhx.available_devices())
device_counts = tuple(count for count in (1, 2, 4) if count <= available_count)
side = SIDE if SIDE is not None else (1024 if platform == "gpu" else 256)
shape = (side, side)
steps = STEPS if STEPS is not None else (50 if platform == "gpu" else 20)
sample_count = SAMPLES
case_count = CASES
device_counts = tuple(count for count in device_counts if case_count % count == 0)
dt = 5.0e-5 if platform == "gpu" else 2.0e-3
equilibria = tuple(
    mhx.PeriodicDoubleHarrisEquilibrium(
        width=0.4,
        perturbation_amplitude=1.0e-3 * (1.0 + 0.05 * case),
        perturbation_mode=(2, 1),
    )
    for case in range(case_count)
)

output = Path(OUTDIR) if OUTDIR is not None else Path(f"outputs/gallery/strong_scaling_{platform}")
output.mkdir(parents=True, exist_ok=True)

print(
    f"Strong scaling on {platform}: {case_count} cases of {shape[0]} x {shape[1]}, "
    f"{steps} steps, device counts {device_counts}, {sample_count} samples each",
    flush=True,
)

run_times = []
run_samples = []

for device_count in device_counts:
    samples = []
    for sample in range(sample_count):
        result = mhx.Simulation(
            shape=shape,
            dt=dt,
            t_end=steps * dt,
            save_every=10,
            device_count=device_count,
            verbose=False,
        ).run_ensemble(equilibria)
        if not math.isfinite(float(jax.numpy.max(jax.numpy.abs(result.final_states.psi)))):
            raise RuntimeError("ensemble produced a non-finite magnetic field")
        samples.append(result.run_seconds)
        print(
            f"  {device_count} device(s), sample {sample + 1}/{sample_count}: "
            f"{samples[-1]:.4f} s",
            flush=True,
        )
    run_samples.append(samples)
    run_times.append(median(samples))
    print(f"{device_count} device(s): {run_times[-1]:.4f} s (median)", flush=True)

# --- Results ---------------------------------------------------------------
result.print_summary()

speedup = [run_times[0] / run_time for run_time in run_times]
measurements = {
    "platform": platform,
    "devices": [str(device) for device in mhx.available_devices()],
    "jax_version": jax.__version__,
    "shape": list(shape),
    "case_count": case_count,
    "parallel_axis": "independent_case",
    "steps": steps,
    "dt": dt,
    "samples_per_count": sample_count,
    "finite": True,
    "device_counts": list(device_counts),
    "run_samples_seconds": run_samples,
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
axis.set_xlabel(f"{platform.upper()} devices")
axis.set_ylabel("speedup")
axis.set_title(f"MHX {platform.upper()}: {case_count} fixed {shape[0]} x {shape[1]} cases")
axis.grid(alpha=0.25)
axis.legend(frameon=False)
figure.savefig(output / "strong_scaling.png", dpi=180)
plt.close(figure)

print(f"Figure: {output / 'strong_scaling.png'}")
print(f"Data:   {output / 'measurements.json'}")
