# Long-run evidence

This page records the first real nonlinear runs executed under a 30-minute
single-run budget. The evidence is useful, but the interpretation is deliberately
skeptical: these runs validate long integration, checkpointing, media, and
nonlinear budget gates; they do **not** yet demonstrate Rutherford growth or
plasmoid onset.

## Reproducible command sequence

The completed duration run used the restartable production executor:

```bash
mhx campaign rutherford-plan-production \
  --outdir outputs/long_runs/rutherford_96_dt005_full_20260512 \
  --nx 96 --ny 96 --dt 0.05 --target-saved-frames 200 \
  --max-walltime-hours 0.5 \
  --seconds-per-step-estimate 0.04 \
  --checkpoint-interval-minutes 5 \
  --preemption-margin-minutes 2

mhx campaign rutherford-execute \
  outputs/long_runs/rutherford_96_dt005_full_20260512 \
  --max-steps 45802 --movies \
  --max-relative-energy-growth 1e-6 \
  --max-divergence-linf 1e-8
```

The active nonlinear-budget run used the multi-mode reduced-MHD state from
[`nonlinear.py`](https://github.com/uwplasma/MHX/blob/main/src/mhx/benchmarks/nonlinear.py):

```python
from mhx.benchmarks.nonlinear import write_nonlinear_energy_budget_validation

write_nonlinear_energy_budget_validation(
    "outputs/long_runs/nonlinear_budget_96_dt005_steps20000_20260512",
    shape=(96, 96),
    resistivity=2e-2,
    viscosity=2e-2,
    dt=5e-3,
    steps=20000,
    save_every=50,
    max_budget_residual=5e-4,
    max_relative_energy_growth=1e-8,
)
```

The current-sheet long replay uses the periodic double-Harris initializer from
[`equilibria.py`](https://github.com/uwplasma/MHX/blob/main/src/mhx/physics/equilibria.py)
and the scalable base-vs-seeded benchmark in
[`current_sheet.py`](https://github.com/uwplasma/MHX/blob/main/src/mhx/benchmarks/current_sheet.py):

```bash
mhx benchmark double-harris-long-run \
  --outdir outputs/long_runs/periodic_double_harris_seeded_128_t100_20260512 \
  --nx 128 --ny 128 --t-end 100 --save-every 200 --fit-stop 10 --no-movies
```

## Rutherford-duration executor run

The `96×96` Rutherford-duration executor run completed the configured duration
target:

| Quantity | Value |
| --- | ---: |
| RK4 steps | 45,802 |
| final time | 2290.1 |
| saved samples | 202 |
| policy e-folds | 30 |
| elapsed walltime | 888.6 s |
| executor gates | passed |
| final/initial reconnecting-flux proxy | `2.73e-6` |
| final/initial island-width proxy | `1.65e-3` |
| final/initial total energy | `1.03e-2` |
| max kinetic energy | `5.85e-9` |

![Rutherford duration histories](_static/validation/long_runs/rutherford_96_production_histories.png)

The fixed-scale movies show the same conclusion visually: the periodic cosine
field diffuses away rather than forming growing islands.

![Long-run flux preview](_static/validation/long_runs/rutherford_96_flux_preview.gif)

![Long-run current-density preview](_static/validation/long_runs/rutherford_96_current_preview.gif)

### Skeptical interpretation

This is strong evidence for the production-executor path:

- the full target step count is completed in one restartable bundle;
- checkpoint state, checkpoint metadata, resume plan, manifest hashes, fixed-scale
  movies, and history schema are written;
- finite-history, energy, divergence, checkpoint, and movie gates all pass.

It is **not** evidence for Rutherford growth. The reconnecting-flux proxy,
island-width proxy, current, and total energy all decay, and the kinetic energy
stays nearly zero. The current periodic cosine initial condition is therefore a
long dissipative integration test, not a tearing-growth experiment.

## Active nonlinear energy-budget run

The second run uses a genuinely nonlinear multi-mode initial condition with a
large ideal-to-full RHS ratio. It checks the periodic reduced-MHD budget

$$
\frac{dE}{dt} = -\eta \langle j^2 \rangle - \nu \langle \omega^2 \rangle,
\qquad
E = \frac{1}{2}\langle |\nabla\psi|^2 + |\nabla\phi|^2\rangle .
$$

| Quantity | Value |
| --- | ---: |
| grid | `96×96` |
| RK4 steps | 20,000 |
| final time | 100.0 |
| saved samples | 401 |
| nonlinear RHS ratio | 0.994 |
| relative energy drop | 0.985 |
| max relative budget residual | `3.65e-5` |
| gates | passed |

![Long nonlinear energy budget](_static/validation/long_runs/nonlinear_budget_96_long.png)

This is good evidence that nonlinear Poisson brackets, spectral current,
dissipation signs, and RK4 integration remain coherent over a substantially
longer run than the FAST CI defaults.

## Seeded double-Harris long replay

The new periodic double-Harris run is the first long nonlinear current-sheet
replay in this rebuild that shows an early instability-path response rather
than pure decay of a stable cosine equilibrium. It advances a base run and a
seeded run on the same grid and measures

$$
A_s(t)=\frac{\|q_s(t)-q_b(t)\|_2}{\epsilon}.
$$

| Quantity | Value |
| --- | ---: |
| grid | `128×128` |
| RK4 steps | 10,000 |
| final time | 100.0 |
| saved samples | 51 |
| early fitted growth rate | `0.141` |
| early amplification | `5.27×` |
| maximum amplification | `7.35×` |
| final/initial total energy | `0.351` |
| max kinetic energy | `6.13e-7` |
| elapsed walltime | `61.9 s` |
| gates | passed |

![Seeded double-Harris 128x128 long replay](_static/validation/long_runs/double_harris_seeded_128_t100.png)

### Skeptical interpretation

This result is a real improvement over the earlier Rutherford-duration cosine
run because the perturbation grows for several Alfvén times before saturating
and relaxing. It still does **not** close a paper-grade reconnection claim:
the kinetic energy remains very small, the current-sheet peak decays under the
chosen resistivity/viscosity, and the run has no resolution, time-step, seed,
or aspect-ratio sweep. The correct conclusion is that MHX now has a scalable
nonlinear current-sheet validation lane suitable for those sweeps.

## Current claim boundary

These runs support:

- long-run stability of the current reduced-MHD code path;
- production-executor artifact correctness under a completed duration target;
- nonlinear energy/dissipation-budget correctness for an active nonlinear state.
- early seeded-growth response for a periodic double-Harris current-sheet replay.

These runs do not yet support:

- Rutherford island-growth scaling;
- plasmoid onset statistics;
- Sweet-Parker reconnection-rate scaling;
- publication-grade reconnection claims.

The next required physics step is to turn the double-Harris replay into a
convergence campaign: sweep resolution, time step, seed amplitude/mode, sheet
width/aspect ratio, and Lundquist number, then promote only those figures whose
scalings survive the sweep.
