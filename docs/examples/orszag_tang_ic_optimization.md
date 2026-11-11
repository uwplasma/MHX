# Optimize Orszag--Tang initial conditions

Recover four initial-condition parameters `p0`--`p3` of a modified Orszag--Tang
vortex by differentiating the reduced-MHD evolution end to end. A "true" state
generated from known parameters is the target, and L-BFGS minimizes the
distance to it.

## Set up

The script needs `optax`:

```bash
python -m pip install -e ".[research]"
```

The true values default to `P0 = 1.5, P1 = 0.8, P2 = 1.2, P3 = 0.5`, and the
resistivity and viscosity (`ETA`, `NU`) to `1e-3`. Change them in the "Input
parameters" block of `examples/orszag_tang/optimize_initial_conditions.py`.

## Run

Edit the constants if you want, then run the script from the repository root.
For a quick run on a 32 x 32 grid set `RESOLUTION = 32` and `OPT_STEPS = 3`:

```bash
python examples/orszag_tang/optimize_initial_conditions.py
```

`RESOLUTION` sets `NX` and `NY` together (leave it `None` to use them
separately). The run prints the loss at each step and writes
`initial_condition_opt_loss.png`, the two GIFs and the optimization history to
`OUTDIR`.

To see the loss landscape, set `SCAN_ETA = True` and any of `SCAN_P1` to
`SCAN_P4` (these sweep `p0` to `p3`) to `True`, and `SCAN_POINTS = 11`. Each scan
writes `scan_<name>_vs_loss.png`, and `scan_results.npz` collects them.

## Scale up

Use `OPT_STEPS = 20` or more, `RESOLUTION = 128`, and `T_END` of 1 or longer.
Reverse mode stores the whole tape, so memory grows with the number of steps.
Use a GPU for resolutions above 128.
