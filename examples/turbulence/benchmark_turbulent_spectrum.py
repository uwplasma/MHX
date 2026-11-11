"""2D MHD turbulent cascade spectrum test.

Runs continually forced 2D reduced-MHD turbulence, keeps every saved frame, and
writes ``turbulent_spectrum.npz`` (time, psi, omega, current density, energies and
the 1D spectrum), ``diagnostics.json``, ``validation.json``, ``manifest.json`` and
a summary figure under ``figures/``. The defaults take several minutes on a CPU;
to scale up, raise ``NX``/``NY`` and ``T_END`` (use a GPU) and raise ``SAVE_EVERY``
so the saved file stays manageable. Run from the repository root:

    python examples/turbulence/benchmark_turbulent_spectrum.py
"""

import time
from pathlib import Path

import jax

from mhx.benchmarks.turbulent_spectrum import write_turbulent_spectrum_validation

# --- Input parameters ------------------------------------------------------
OUTDIR = Path("outputs/examples/turbulence_2d")
NX = 256  # grid points in x
NY = 256  # grid points in y
RESISTIVITY = 5.0e-5  # magnetic diffusivity eta
VISCOSITY = 5.0e-5  # kinematic viscosity nu
DT = 1.0e-3  # time step
T_END = 50.0  # simulation end time
SAVE_EVERY = 1000  # steps between saved frames

# --- Run -------------------------------------------------------------------
print("Running 2D MHD turbulent cascade spectrum test", flush=True)
print(f"Grid: {NX}x{NY} | dt: {DT} | t_end: {T_END} | steps: {round(T_END / DT)}", flush=True)
print(f"JAX devices: {jax.devices()}", flush=True)
print(f"Output directory: {OUTDIR.resolve()}", flush=True)
print("Compiling and running (no progress is printed inside the run)...", flush=True)

start = time.perf_counter()
manifest_path, validation = write_turbulent_spectrum_validation(
    outdir=OUTDIR,
    shape=(NX, NY),
    resistivity=RESISTIVITY,
    viscosity=VISCOSITY,
    dt=DT,
    t_end=T_END,
    save_every=SAVE_EVERY,
)
elapsed = time.perf_counter() - start

# --- Results ---------------------------------------------------------------
passed = validation.get("passed", False)
print(f"Run and write took {elapsed:.1f} s")
print(f"wrote {manifest_path}")
print(f"passed={passed}")
if not passed:
    print("Failed checks:")
    for check, passed_check in validation.get("checks", {}).items():
        if not passed_check:
            print(f"  - {check}")
