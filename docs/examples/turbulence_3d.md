# 3D forced MHD turbulence

Run forced 3D incompressible MHD turbulence with the paper-matching setup: a
periodic box of side `2*pi`, Gaussian white-noise forcing of each Elsasser field in
the band `n_perp` in `[1, 2]` with `|n_z| <= 1` and amplitude `0.048`, and a
hyper-resistive `eta = nu = 6 / 21**4`.

## Set up

The script reads its settings from environment variables. Leave them unset for
the defaults:

| Variable | Default | Meaning |
| --- | --- | --- |
| `MHX_TURBULENCE_3D_N` | `64` | grid points per side |
| `MHX_TURBULENCE_3D_DT` | `2e-3` | time step |
| `MHX_TURBULENCE_3D_T_END` | `200` | end time |
| `MHX_TURBULENCE_3D_SAVE_EVERY` | `500` | steps between saved frames |
| `MHX_TURBULENCE_3D_OUTDIR` | `outputs/gallery/turbulence_3d` | output folder |

## Run

A quick check on a 16-cubed grid:

```bash
MHX_TURBULENCE_3D_N=16 MHX_TURBULENCE_3D_T_END=0.2 \
MHX_TURBULENCE_3D_SAVE_EVERY=10 \
python examples/gallery/11_turbulence_3d.py
```

This takes under a minute and prints an energy summary. It writes `trajectory.npz`, `spectrum.png`, and two GIFs (`slices.gif`,
`outer_faces_cube.gif`) to the output folder.

The spectrum is averaged over a late window (`t = 180` to `200` by default). A run
that ends before that window falls back to the final-state spectrum and prints
a warning, so report it as a final-state spectrum.

## Scale up

Use `MHX_TURBULENCE_3D_N=128` with `MHX_TURBULENCE_3D_T_END=200` on a GPU. The
dissipation coefficients are fixed in the script (`RESISTIVITY` and `VISCOSITY`), so
edit them to change the Reynolds number as the grid grows.
