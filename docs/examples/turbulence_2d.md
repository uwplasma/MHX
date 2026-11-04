# 2D forced MHD turbulence and its spectrum

Run continually forced 2D reduced-MHD turbulence, keep every saved frame, and
render the current density beside the live magnetic spectrum.

## Set up

The run uses `eta = nu = 5e-5`, forcing at `k = 1` to `3`, and a 2/3 dealiasing
cutoff. Install the package, then check the options:

```bash
python -m pip install -e .
python examples/turbulence/benchmark_turbulent_spectrum.py --help
```

## Run

A small run on a 64 x 64 grid:

```bash
python examples/turbulence/benchmark_turbulent_spectrum.py \
  --nx 64 --ny 64 --t-end 2 --save-every 100 --outdir outputs/turbulence_2d
```

This writes `turbulent_spectrum.npz` (time, psi, omega, current density,
energies, and the 1D spectrum), `diagnostics.json`, `validation.json`, and a
summary figure under `figures/`.

Render the animation from the saved frames:

```bash
python examples/turbulence/render_turbulence_2d.py \
  --npz outputs/turbulence_2d/turbulent_spectrum.npz \
  --frames 6 --kmin 3 --kmax 20 --kplot 30
```

The GIF, a final figure, and `slope_history.csv` land in
`outputs/turbulence_2d/dynamic/`. The fit band `--kmin` to `--kmax` should stay
below the 2/3 dealiasing edge.

Other scripts in `examples/turbulence/` read the same file:
`analyze_spectrum.py` fits the spectrum, and `animate_turbulence.py` animates
`j_z`.

## Scale up

The defaults are `256 x 256` to `t = 50`. For a poster-quality cascade use
`--nx 1024 --ny 1024` and a longer `--t-end` on a GPU, and widen the fit band to
`--kmin 4 --kmax 40`. A `1024 x 1024` run stores every saved frame, so raise
`--save-every` to keep the file size manageable.
