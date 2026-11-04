# Sweet--Parker collapse and scaling

Test whether a large-`Delta'` double-Harris sheet collapses into a thin
Sweet--Parker current sheet, then measure how the sheet thickness and
reconnection rate scale with the Lundquist number.

## Set up

The setup is a periodic double-Harris sheet of half-width `a` in a
`Lx x Ly` box, seeded with a single `(0, 1)` mode. The sheets sit at `Lx/4` and
`3Lx/4`. The default box is `Lx = 8*pi`, so the islands clear each other for
longer. The script measures the left sheet at every saved time with
`mhx.diagnostics.measure_sweet_parker_sheet`.

```bash
python -m pip install -e .
python examples/sweet_parker/collapse_check.py --help
```

## Run

A short check on a small grid:

```bash
python examples/sweet_parker/collapse_check.py \
  --nx 256 --ny 64 --t-end 2 --outdir outputs/sweet_parker/small
```

The folder gets `summary.json`, `histories.npz`, `histories.png`,
`current_snapshots.png`, `snapshots.npz` and `domain.gif`. A run this short does
not collapse. It only checks that the setup runs.

Useful options:

- `--eta` and `--pm` set the resistivity and the magnetic Prandtl number.
- `--hold-equilibrium` stops the Harris sheet from diffusing away, so the sheet
  keeps a constant `E_0`.
- `--sap` stops shortly after the clean collapse phase ends.
- `--replot RUN_DIR` re-assesses and re-plots a finished run.
- `--devices N` splits the x axis across `N` GPUs. `--nx` must be divisible by `N`.

## Scaling scan

Run `collapse_check.py` at several resistivities, each into its own folder, then
plot them together:

```bash
python examples/sweet_parker/plot_scaling.py \
  outputs/sweet_parker/eta2e-3 outputs/sweet_parker/eta1e-3 --out sp_scaling.png
```

The plot shows `delta/L` and the reconnection rate against the sheet Lundquist
number `S_L`, plus the local slope between neighbouring runs, which should
approach `-1/2` for both. Each run needs a steady window, so use the full
defaults (`--nx 3072 --ny 384 --t-end 300`) for real data.

## Scale up

On Perlmutter, `examples/sweet_parker/perlmutter_collapse.sbatch` submits the
default-resolution run, and `modal_run_sweet_parker_collapse.py` runs it on a
Modal H100.
