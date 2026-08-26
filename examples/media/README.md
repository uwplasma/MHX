# Documentation media campaign

These scripts generate the simulation bundles and staged media used by both
the project README and the documentation gallery. README GIFs or posters and
documentation movies are encoded from the same plotted frame sequence. There
is no separate low-resolution README simulation.

Nothing in this directory overwrites committed documentation media. Simulation
bundles are written under `outputs/media-campaign/` and renders are written
under `outputs/media-preview/`.

## Commands

Inspect source availability:

```bash
python examples/media/run_all.py status --preset final
```

Run one case:

```bash
python examples/media/decaying_turbulence.py simulate --preset final
python examples/media/decaying_turbulence.py render --preset final
```

Run the complete final campaign and then render it:

```bash
python examples/media/run_all.py simulate --preset final --allow-expensive
python examples/media/run_all.py render --preset final
```

Use `--preset preview` while changing plotting code. A preview bundle is
recorded as such and is not suitable for promoting as final documentation
media.

## Cases

- `decaying_turbulence.py`: fixed-scale current density plus a live magnetic spectrum.
- `double_harris_reconnection.py`: broadband-seeded plasmoid-chain flux and current-sheet views through t=60.
- `forced_turbulence_2d.py`: continually forced 2-D turbulence with current, live spectrum, and per-frame compensated-spectrum constant fits.
- `orszag_tang_2d.py`: the curated labeled 2-D current-density GIF and MP4.
- `orszag_tang_3d.py`: 192-cubed midplane and maximum-projection current magnitude.
- `kelvin_helmholtz.py`: periodic passive-dye roll-up ported from the source notebook logic.
- `turbulence_3d.py`: 96-cubed midplane and maximum-projection current magnitude.

Each staged render writes a provenance record under
`outputs/media-preview/records/` and refreshes the aggregate
`outputs/media-preview/media_build.json`. Inspect the staged files before copying
selected outputs into `docs/_static/`, then update
`docs/figures/manifest.toml` and run:

```bash
python tools/render_all_media.py --check-only
```
