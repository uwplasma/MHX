# Make your first movie

Movies are the fastest way to check whether a run did what you expect. This
page shows both movie paths: the one-command CLI route and a Python loop you
can adapt.

## The CLI route

Every configured run can regenerate its figures and a flux movie:

```bash
mhx init outputs/movies/linear_tearing.toml
mhx run outputs/movies/linear_tearing.toml --outdir outputs/movies/smoke
mhx figures outputs/movies/smoke --gif
```

This writes `figures/flux_movie.gif` next to `flux_final.png` and
`mode_amplitude.png` inside the run directory. `mhx report` adds a Markdown
summary of the same run.

## The Python route

For full control, render frames with Matplotlib and assemble them with
`imageio`, which the base install includes. This example animates the growing
island from [your first run](first_run.md):

```python
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np

import mhx

output = Path("outputs/movies/island")
output.mkdir(parents=True, exist_ok=True)

simulation = mhx.Simulation(
    shape=(64, 64),
    equilibrium=mhx.PeriodicDoubleHarrisEquilibrium(
        width=0.4,
        perturbation_amplitude=4.0e-3,
        perturbation_mode=(2, 1),
    ),
    resistivity=5.0e-3,
    viscosity=5.0e-3,
    dt=2.0e-2,
    t_end=40.0,
    save_every=100,
)
result = simulation.run()

flux = np.asarray(result.trajectory.states.psi)
island = flux - flux.mean(axis=2, keepdims=True)
times = np.asarray(result.trajectory.times)
scale = np.abs(island).max()

frames = []
for index, time in enumerate(times):
    figure, axis = plt.subplots(figsize=(4, 4), dpi=120)
    axis.imshow(
        island[index].T,
        origin="lower",
        cmap="RdBu_r",
        vmin=-scale,
        vmax=scale,
    )
    axis.contour(flux[index].T, colors="black", linewidths=0.4, levels=12)
    axis.set_title(f"island flux, t = {time:.0f}")
    axis.set_xticks([])
    axis.set_yticks([])
    figure.tight_layout()
    frame_path = output / f"frame_{index:03d}.png"
    figure.savefig(frame_path)
    plt.close(figure)
    frames.append(imageio.imread(frame_path))

imageio.mimsave(output / "island_flux.gif", frames, duration=200, loop=0)
```

Three rules keep a movie honest:

1. **Fix the color scale across frames.** A per-frame scale hides growth and
   invents motion.
2. **Show the changing part.** Plot the island flux or the residual against a
   base run, and overlay total-flux contours for orientation.
3. **Label time.** A movie without a clock supports no claim about rates.

The [media inventory](../project/media_inventory.md) records these rules,
plus the source command and claim level, for every committed movie.
