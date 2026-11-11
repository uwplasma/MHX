# 3D Orszag--Tang vortex

Run the 3D incompressible Orszag--Tang vortex through the `mhx.Simulation` API,
then track its energies, a spectrum snapshot and the current on the cube faces.

## Set up

```python
import mhx

simulation = mhx.Simulation(
    shape=(64, 64, 64),
    equations="mhd3d",
    equilibrium=mhx.OrszagTang3DEquilibrium(beta=0.8),
    viscosity=5.0e-3,
    resistivity=5.0e-3,
    dt=2.0e-3,
    t_end=2.0,
    save_every=100,
)

result = simulation.run()
result.print_summary()
result.plot("outputs/orszag_tang_3d/summary.png")
result.save("outputs/orszag_tang_3d")
```

`result.save` writes `trajectory.npz`. The same code ships as
`examples/gallery/09_orszag_tang_3d.py`.

## Run

```bash
python examples/gallery/09_orszag_tang_3d.py
```

This takes about two minutes on a laptop CPU and writes to
`outputs/gallery/orszag_tang_3d/`.

Track the run. Open `examples/orszag_tang/track_orszag_tang_3d.py`, set the
constants under "Input parameters" (for a quick pass `KMIN = 3`, `KMAX = 16` and
`GIF_FRAMES = 5`), and run it:

```bash
python examples/orszag_tang/track_orszag_tang_3d.py
```

`NPZ` points at `outputs/gallery/orszag_tang_3d/trajectory.npz` by default.
Results land in `outputs/gallery/orszag_tang_3d/track/` (set `OUTDIR` to change
it): `energy.png`, `spectrum.png` with a power-law fit, `cube_final.png`,
`cube.gif`, and the `history.csv` and `spectrum.csv` data. The script rebuilds
`t = 0` analytically, because the NPZ does not store it, so set `BETA` if you
changed `beta`. Set `FIELD = "B"` for `|B|` on the faces. The fit stops at
`k = N/3`.

## Scale up

Raise `shape` to `(128, 128, 128)` or higher on a GPU and lower `viscosity` and
`resistivity` with it. Keep `dt` small enough for the CFL and diffusion limits.
Time per step grows with the number of grid points, so expect about eight times
the work per doubling of `N`.
