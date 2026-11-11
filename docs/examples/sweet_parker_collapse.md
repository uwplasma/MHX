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

Install the package, then open `examples/sweet_parker/collapse_check.py` and edit
the constants in "Input parameters".

```bash
python -m pip install -e .
```

## Run

For a short check on a small grid, set `NX = 256`, `NY = 64`, `T_END = 2` and
`OUTDIR = Path("outputs/sweet_parker/small")`, then run from the repository root:

```bash
python examples/sweet_parker/collapse_check.py
```

The folder gets `summary.json`, `histories.npz`, `histories.png`,
`current_snapshots.png`, `snapshots.npz` and `domain.gif`. A run this short does
not collapse. It only checks that the setup runs.

Constants that matter:

- `ETA` and `PM` set the resistivity and the magnetic Prandtl number.
- `HOLD_EQUILIBRIUM = True` stops the Harris sheet from diffusing away, so the
  sheet keeps a constant `E_0`.
- `SAP = True` stops shortly after the clean collapse phase ends
  (`SAP_BUFFER` time units later).
- `REPLOT = Path("outputs/sweet_parker/<run>")` re-assesses and re-plots a
  finished run without simulating.
- `DEVICES = N` splits the x axis across `N` GPUs. `NX` must be divisible by `N`.

## Scaling scan

Run `collapse_check.py` at several values of `ETA`, each with its own `OUTDIR`,
then list those folders in `RUN_DIRS` in `examples/sweet_parker/plot_scaling.py`
(and set `OUTPUT` for the figure) and run it:

```bash
python examples/sweet_parker/plot_scaling.py
```

The plot shows `delta/L` and the reconnection rate against the sheet Lundquist
number `S_L`, plus the local slope between neighbouring runs, which should
approach `-1/2` for both. Each run needs a steady window, so use the full
defaults (`NX = 3072`, `NY = 384`, `T_END = 300`) for real data.

## Scale up

On Perlmutter, `examples/sweet_parker/perlmutter_collapse.sbatch` submits the
default-resolution run (it copies the script into the output folder and sets the
constants with `sed`; override them with environment variables such as `NX`, `ETA`
and `TEND`). To use a Modal H100 instead, run
`modal run modal_run.py --script examples/sweet_parker/collapse_check.py --overrides "NX=3072 ETA=2e-3"`.
