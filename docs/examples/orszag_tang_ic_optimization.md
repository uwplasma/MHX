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

The true values default to `p0=1.5, p1=0.8, p2=1.2, p3=0.5`. Change them with
`--p0` to `--p3`. The resistivity and viscosity default to `1e-3`.

## Run

A quick run on a 32 x 32 grid:

```bash
python examples/orszag_tang/optimize_initial_conditions.py \
  --resolution 32 --opt-steps 3 --outdir outputs/orszag_tang_ic_opt
```

`--resolution` sets `nx` and `ny` together. The run prints the loss at each
step and writes `initial_condition_opt_loss.png` and the optimization history to
`--outdir`.

Add one-dimensional loss scans around the true values to see the loss
landscape. Each scan writes `scan_<name>_vs_loss.png`:

```bash
python examples/orszag_tang/optimize_initial_conditions.py \
  --resolution 32 --opt-steps 3 --scan-eta --scan-p1 --scan-points 11 \
  --outdir outputs/orszag_tang_ic_scan
```

## Scale up

Use `--opt-steps 20` or more, `--resolution 128`, and `--t-end` of 1 or longer.
Reverse mode stores the whole tape, so memory grows with the number of
steps. Use a GPU for resolutions above 128.
