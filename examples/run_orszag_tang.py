"""Run the reduced-MHD Orszag--Tang nonlinear example with movies.

Evolves the viscoresistive Orszag--Tang vortex, checks it against the validation
gates in mhx.benchmarks, and writes diagnostics, figures, optional GIFs and a
manifest under OUTDIR. Takes about a minute on a laptop CPU; raise NX, NY and
T_END to scale up, or set MOVIES = False to skip the GIFs. Run from the
repository root:

    python examples/run_orszag_tang.py
"""

import time
from pathlib import Path

from mhx.benchmarks import write_orszag_tang_vortex_validation
from mhx.runtime import configure_jax

# --- Input parameters ------------------------------------------------------
OUTDIR = Path("outputs/examples/orszag_tang")
NX = 64  # grid points in x
NY = 64  # grid points in y
RESISTIVITY = 1.0e-2  # magnetic diffusivity eta
VISCOSITY = 1.0e-2  # kinematic viscosity nu
DT = 5.0e-3  # time step
T_END = 6.0  # end time
SAVE_EVERY = 40  # steps between saved states
MOVIES = True  # write GIFs of the flux, current and vorticity

# --- Run -------------------------------------------------------------------
configure_jax(enable_x64=True)
print(f"Orszag--Tang vortex: {NX}x{NY}, eta={RESISTIVITY:g}, nu={VISCOSITY:g}, dt={DT:g}, "
      f"t_end={T_END:g}, movies={MOVIES}", flush=True)
start = time.perf_counter()
manifest_path, validation = write_orszag_tang_vortex_validation(
    OUTDIR,
    shape=(NX, NY),
    resistivity=RESISTIVITY,
    viscosity=VISCOSITY,
    dt=DT,
    t_end=T_END,
    save_every=SAVE_EVERY,
    movies=MOVIES,
)
print(f"Run time (including compile and output): {time.perf_counter() - start:.2f} s")

# --- Results ---------------------------------------------------------------
print(f"wrote {manifest_path}")
print(f"passed={validation['passed']}")
