# Gallery

Every movie here comes from a recorded MHX run. The caption under each one
states what it shows, the command that made it, and the claim boundary. The
[media inventory](project/media_inventory.md) holds the full provenance,
settings, and quality checks for each source.

## Harris-sheet reconnection

```{video} _static/movies/double_harris_reconnection.mp4
:loop:
:muted:
:width: 100%
```

Residual reconnecting flux $\Delta\psi$ in a seeded double-Harris sheet, with
total-flux contours and X/O markers, from a 128 x 128 run to $t=160$.
Validation media anchored to the Harris and FKR tearing picture
{cite}`harris1962,furth1963`. The markers are diagnostic annotations, not
converged Rutherford or plasmoid evidence.

```{video} _static/movies/double_harris_current_sheet.mp4
:loop:
:muted:
:width: 100%
```

Full-domain view of the same run through the residual current $\Delta j_z$
across both sheets.

```{video} _static/movies/double_harris_island_64.mp4
:loop:
:muted:
:width: 60%
```

The island movie from [make your first movie](getting_started/first_movie.md)
at its exact documented settings: 64 x 64, $t_{\mathrm{end}}=40$, $S=200$.
Solver output without validation gates, shown so readers can check their own
result.

## Orszag--Tang vortex

```{video} _static/movies/orszag_tang_current.mp4
:loop:
:muted:
:width: 100%
```

Current-density filament formation in the reduced-MHD Orszag--Tang vortex
{cite}`orszag1979` at 96 x 96 to $t=10$. Generate it with:

```bash
mhx benchmark orszag-tang --outdir outputs/orszag_tang --nx 96 --ny 96 --t-end 10 --movies
```

```{video} _static/movies/orszag_tang_vorticity.mp4
:loop:
:muted:
:width: 100%
```

Vorticity roll-up from the same run.

```{video} _static/movies/orszag_tang_flux.mp4
:loop:
:muted:
:width: 100%
```

Flux-function deformation and dissipative mixing from the same run. All
three are nonlinear reduced-MHD validation media, not compressible
shock-capturing results.

## Turbulence

```{video} _static/movies/decaying_mhd_turbulence_current.mp4
:loop:
:muted:
:width: 100%
```

Current filaments in decaying reduced-MHD turbulence {cite}`biskamp2003` at
64 x 64 to $t=8$:

```bash
mhx benchmark decaying-turbulence --outdir outputs/turbulence --nx 64 --ny 64 --t-end 8 --movies
```

```{video} _static/movies/forced_turbulent_reconnection.mp4
:loop:
:muted:
:width: 100%
```

Forced turbulent reconnection proxy at 64 x 64 to $t=80$, with flux contours
and a reconnection-rate proxy. Pedagogical validation media, not a
three-dimensional fast-reconnection test.

## Linear theory

```{video} _static/movies/harris_layer_sweep.mp4
:loop:
:muted:
:width: 100%
```

The direct Harris-sheet eigenproblem swept over Lundquist number: growth
decreases with $S$ while the resonant layer narrows, the classical tearing
localization of {cite}`furth1963`. From `mhx benchmark linear-tearing-layer`.

## Reproduce or extend

Regenerate every movie on this page with:

```bash
python examples/make_docs_movies.py
```

The script transcodes the committed validated sources and renders the island
example. [Make your first movie](getting_started/first_movie.md) shows how to
build your own.
