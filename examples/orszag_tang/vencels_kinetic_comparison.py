"""Vencels kinetic comparison for the nonlinear Orszag--Tang vortex.

Runs the incompressible reduced-MHD Orszag--Tang vortex on the 10 x 10 box of
the Vencels kinetic simulation and draws a three-panel comparison figure: the
energy partitioning against the Vencels magnetic, electron and ion curves, the
magnetic power spectrum at t = t_A against the Vencels spectrum, and the current
density j_z. It writes the figure and a manifest. The reference data ships with
the package (mhx/benchmarks/data/vencelsdata.csv). The default run takes seconds on a
laptop CPU; raise NX and NY (for example 256, with DT = 5.0e-4 to
stay below the diffusion limit) for a converged comparison. Run from the
repository root:

    python examples/orszag_tang/vencels_kinetic_comparison.py
"""

import time
from pathlib import Path

import jax

from mhx.benchmarks.vencels import VENCELS_CSV, VENCELS_DOMAIN_SIZE, run_orszag_tang
from mhx.io import write_manifest
from mhx.plotting import plot_kinetic_comparison

# --- Input parameters ------------------------------------------------------
OUTDIR = Path("outputs/examples/orszag_tang_kinetic")
NX = 24  # grid points in x
NY = 24  # grid points in y
RESISTIVITY = 1.0e-2  # magnetic diffusivity eta
VISCOSITY = 1.0e-2  # kinematic viscosity nu
DT = 5.0e-3  # time step
T_END = 30.0  # end time (the Alfven time is t_A = 10)
SAVE_EVERY = 20  # steps between saved states

# --- Run -------------------------------------------------------------------
if not VENCELS_CSV.exists():
    raise FileNotFoundError(f"Could not find Vencels data at {VENCELS_CSV}")
OUTDIR.mkdir(parents=True, exist_ok=True)

print(f"Orszag--Tang vs Vencels: {NX}x{NY} grid on a {VENCELS_DOMAIN_SIZE:g} x "
      f"{VENCELS_DOMAIN_SIZE:g} box, eta={RESISTIVITY:g}, nu={VISCOSITY:g}")
print(f"dt={DT:g}, t_end={T_END:g} ({int(round(T_END / DT))} steps, save every {SAVE_EVERY}); "
      f"devices: {jax.devices()}", flush=True)
start = time.perf_counter()
result = run_orszag_tang(
    shape=(NX, NY),
    resistivity=RESISTIVITY,
    viscosity=VISCOSITY,
    dt=DT,
    t_end=T_END,
    save_every=SAVE_EVERY,
)
run_seconds = time.perf_counter() - start  # includes JIT compilation
print(f"Run time (including compile): {run_seconds:.2f} s", flush=True)

# --- Results ---------------------------------------------------------------
total = result["total_energy"]
print(f"saved states: {result['time'].size}")
print(f"relative total-energy change: {(total[-1] - total[0]) / total[0]:+.4f}")
print(f"final magnetic divergence (Linf): {result['final_divergence']:.3e}")

figure_path = plot_kinetic_comparison(
    result, OUTDIR / "kinetic_comparison.png", VENCELS_CSV, VENCELS_DOMAIN_SIZE
)
write_manifest(
    OUTDIR / "manifest.json",
    config={
        "schema": "mhx.example.orszag_tang_vencels_comparison.v1",
        "shape": [NX, NY],
        "domain": [0.0, VENCELS_DOMAIN_SIZE, 0.0, VENCELS_DOMAIN_SIZE],
        "resistivity": RESISTIVITY,
        "viscosity": VISCOSITY,
        "dt": DT,
        "t_end": T_END,
        "final_magnetic_divergence_linf": result["final_divergence"],
        "vencels_csv": str(VENCELS_CSV),
    },
    outputs={"kinetic_comparison": figure_path.name},
    claim_level="validation",
    claim_scope="Fluid-vs-kinetic Orszag--Tang comparison against the Vencels reference.",
)
print(f"wrote {figure_path}")
print(f"wrote {OUTDIR / 'manifest.json'}")
