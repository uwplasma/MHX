# 3D forced MHD turbulence

Run forced 3D incompressible MHD turbulence with the paper-matching setup: a
periodic box of side `2*pi`, Gaussian white-noise forcing of each Elsasser field in
the band `n_perp` in `[1, 2]` with `|n_z| <= 1` and amplitude `0.048`, and a
hyper-resistive `eta = nu = 6 / 21**4`.

## Set up

Edit the constants at the top of `examples/gallery/11_turbulence_3d.py`:

| Constant | Default | Meaning |
| --- | --- | --- |
| `N` | `64` | grid points per side |
| `DT` | `2.0e-3` | time step |
| `T_END` | `200.0` | end time |
| `SAVE_EVERY` | `500` | steps between saved frames |
| `OUTDIR` | `Path("outputs/gallery/turbulence_3d")` | output folder |

## Run

For a quick check on a 16-cubed grid, set `N = 16`, `T_END = 0.2` and
`SAVE_EVERY = 10`, then run from the repository root:

```bash
python examples/gallery/11_turbulence_3d.py
```

This takes under a minute, prints progress and an energy summary, and writes
`trajectory.npz`, `spectrum.png`, and two GIFs (`slices.gif`,
`outer_faces_cube.gif`) to the output folder.

The spectrum is averaged over a late window (`t = 180` to `200` by default). A run
that ends before that window falls back to the final-state spectrum and prints
a warning, so report it as a final-state spectrum.

## Scale up

Set `N = 128` (with `T_END = 200.0`) on a GPU. The dissipation coefficients are
fixed in the script (`RESISTIVITY` and `VISCOSITY`), so edit them to change the
Reynolds number as the grid grows.
