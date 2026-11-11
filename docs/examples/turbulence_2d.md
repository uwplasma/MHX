# 2D forced MHD turbulence and its spectrum

Run continually forced 2D reduced-MHD turbulence, keep every saved frame, and
render the current density beside the live magnetic spectrum.

## Set up

The run uses `eta = nu = 5e-5`, forcing at `k = 1` to `3`, and a 2/3 dealiasing
cutoff. Install the package, then open
`examples/turbulence/benchmark_turbulent_spectrum.py` and edit the constants in
"Input parameters". The ones that matter are `NX`, `NY`, `RESISTIVITY`,
`VISCOSITY`, `DT`, `T_END`, `SAVE_EVERY` and `OUTDIR`.

```bash
python -m pip install -e .
```

## Run

For a small run, set `NX = 64`, `NY = 64`, `T_END = 2` and `SAVE_EVERY = 100`,
then run from the repository root:

```bash
python examples/turbulence/benchmark_turbulent_spectrum.py
```

This writes `turbulent_spectrum.npz` (time, psi, omega, current density,
energies, and the 1D spectrum), `diagnostics.json`, `validation.json`, and a
summary figure under `figures/`, all in `OUTDIR`
(`outputs/examples/turbulence_2d` by default).

Render the animation from the saved frames. In
`examples/turbulence/render_turbulence_2d.py` the constants `NPZ`, `FRAMES`,
`KMIN`, `KMAX` and `KPLOT` choose the file, the number of GIF frames, the fit
band and the largest wavenumber shown (for the small run, try `FRAMES = 6`,
`KMIN = 3`, `KMAX = 20`, `KPLOT = 30`):

```bash
python examples/turbulence/render_turbulence_2d.py
```

The GIF, a final figure, and `slope_history.csv` land in
`outputs/examples/turbulence_2d/dynamic/`. The fit band `KMIN` to `KMAX` should
stay below the 2/3 dealiasing edge.

Other scripts in `examples/turbulence/` read the same file through their `NPZ`
constant: `analyze_spectrum.py` fits the spectrum over `K_MIN` to `K_MAX` and the
window `T_MIN` to `T_MAX`, and `animate_turbulence.py` animates `j_z`.

## Scale up

The defaults are `NX = NY = 256` to `T_END = 50`. For a poster-quality cascade
set `NX = NY = 1024` and a longer `T_END` on a GPU, and keep the fit band at
`KMIN = 4`, `KMAX = 40`. A `1024 x 1024` run stores every saved frame, so raise
`SAVE_EVERY` to keep the file size manageable.
