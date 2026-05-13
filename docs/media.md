# Validation media

The README only keeps a compact visual teaser. Detailed media context belongs
here so each visual carries its claim level, source path, equations, and
literature anchor. These assets are either generated from MHX validation gates
or explicitly labeled theory schematics; none are production nonlinear plasmoid
claims.

## At-a-glance media table

| Asset | What it shows | Claim boundary and anchor |
| --- | --- | --- |
| ![Double-Harris reconnection replay](_static/readme/double_harris_reconnection.gif) | Seeded periodic double-Harris nonlinear replay compressed into a landing-page flux movie. | Solver-generated validation media from the seeded double-Harris benchmark; bounded evidence, not converged Rutherford/plasmoid production. |
| ![Double-Harris current sheet](_static/readme/double_harris_current_sheet.gif) | Same replay through fixed-scale out-of-plane current density, emphasizing sheet sharpening and current filaments. | Solver-generated validation media; useful for morphology QA before larger seed, duration, and resolution sweeps. |
| ![Harris tearing layer sweep](_static/readme/harris_layer_sweep.gif) | Direct Harris-sheet eigenproblem: growth decreases with $S$ while the resonant flow/current layer narrows. | Solver-generated validation media from `mhx benchmark linear-tearing-layer`; anchored to classical tearing localization from [FKR 1963](https://doi.org/10.1063/1.1706761) and the reduced-MHD Harris eigenproblem used by [MacTaggart 2019](https://eprints.gla.ac.uk/191898/1/191898.pdf). |
| ![Plasmoid scaling schematic](_static/readme/plasmoid_scaling_schematic.gif) | Schematic Sweet-Parker sheet fragmentation with $\gamma_{\max}\tau_A\propto S^{1/4}$ and $k_{\max}L\propto S^{3/8}$. | Theory schematic only; anchored to [Loureiro, Schekochihin & Cowley 2007](https://arxiv.org/abs/astro-ph/0703631), not a nonlinear MHX plasmoid result. |
| ![MHD turbulence cascade schematic](_static/readme/mhd_turbulence_cascade.gif) | Synthetic magnetic-flux eddies, current filaments, and an animated cascade guide. | Theory/pedagogy schematic only; not a nonlinear MHX turbulence simulation. |
| ![Seeded double-Harris flux](_static/validation/periodic_double_harris_seeded_long_run/figures/periodic_double_harris_flux.gif) | Seeded periodic double-Harris nonlinear replay at `64×64`, showing magnetic-flux evolution over `t_end=30`. | Validation bridge from Harris tearing to longer nonlinear campaigns; bounded evidence, not converged Rutherford/plasmoid production. |
| ![Seeded double-Harris current](_static/validation/periodic_double_harris_seeded_long_run/figures/periodic_double_harris_current.gif) | Same seeded run through fixed-scale out-of-plane current density. | Checks current-density visualization and dissipative nonlinear replay before aspect-ratio, seed, and resolution sweeps. |
| ![Seeded double-Harris convergence](_static/validation/periodic_double_harris_convergence/periodic_double_harris_convergence.png) | FAST resolution/time-step sweep for the seeded periodic double-Harris replay. | Convergence scaffold that gates spread in early growth/amplification before any production Rutherford/plasmoid claim. |
| ![Restartable Rutherford flux chunk](_static/validation/rutherford_production_execution/fixed_scale_flux_movie.gif) | Restartable Rutherford executor chunk with fixed-scale magnetic flux. | Execution-path validation for the chunked production runner; not completed nonlinear production evidence. |
| ![Restartable Rutherford current chunk](_static/validation/rutherford_production_execution/fixed_scale_current_density_movie.gif) | Same executor chunk through current density, using fixed color limits. | Checks the movie/artifact lane and the current-density visualization contract. |

Still validation figures live on the [physics validation](validation.md),
[long-run evidence](long_run_evidence.md), and
[publication checklist](publication_checklist.md) pages where they can be
interpreted with equations, tolerances, and source links.

## README reconnection and current-sheet previews

The README flux/current pair is compressed from the longer seeded periodic
double-Harris validation movies. The original validation command remains:

```bash
mhx benchmark double-harris-long-run \
  --outdir outputs/benchmarks/periodic_double_harris_seeded_long_run \
  --nx 64 --ny 64 --t-end 30 --save-every 100 --movies
```

The run advances a base periodic double-Harris sheet and a seeded copy, then
tracks normalized perturbation growth, total energy, kinetic energy, peak
current, and current-density frames. It remains `claim_level = "validation"`
until convergence, seed, aspect-ratio, and duration sweeps are attached.

![Double-Harris reconnection replay](_static/readme/double_harris_reconnection.gif)

![Double-Harris current sheet](_static/readme/double_harris_current_sheet.gif)

Source links:

- [Current-sheet validation implementation](https://github.com/uwplasma/MHX/blob/main/src/mhx/benchmarks/current_sheet.py)
- [Current-sheet tests](https://github.com/uwplasma/MHX/blob/main/tests/test_current_sheet_eigenvalue_validation.py)
- [README media generator](https://github.com/uwplasma/MHX/blob/main/examples/make_readme_media.py)

## Harris tearing layer sweep

This GIF is generated from `mhx benchmark linear-tearing-layer`. It uses the
direct finite-domain Harris-sheet eigenproblem and shows the FAST validation
trend: increasing Lundquist number reduces the growth rate and narrows the
localized flow/current response near the resonant surface. The anchor is the
classical tearing-mode picture from Furth, Killeen & Rosenbluth and the
reduced-MHD Harris eigenproblem used in the MacTaggart validation papers.

![Harris tearing layer sweep](_static/readme/harris_layer_sweep.gif)

Source links:

- [Layer benchmark implementation](https://github.com/uwplasma/MHX/blob/main/src/mhx/benchmarks/tearing_eigen.py)
- [Layer validation tests](https://github.com/uwplasma/MHX/blob/main/tests/test_linear_tearing_eigenvalue_validation.py)
- [README media generator](https://github.com/uwplasma/MHX/blob/main/examples/make_readme_media.py)

## Sweet-Parker plasmoid scaling schematic

This GIF is an explicitly labeled theory schematic, not a nonlinear MHX solver
result. It visualizes the Loureiro-Schekochihin-Cowley Sweet-Parker plasmoid
scalings

$$
\gamma_{\max}\tau_A\propto S^{1/4},\qquad k_{\max}L\propto S^{3/8}.
$$

The purpose is pedagogic: a reviewer should immediately see the literature
target that future nonlinear MHX plasmoid runs must recover.

![Plasmoid scaling schematic](_static/readme/plasmoid_scaling_schematic.gif)

Source links:

- [Schematic generator](https://github.com/uwplasma/MHX/blob/main/examples/make_readme_media.py)
- [Analytic scaling gate](https://github.com/uwplasma/MHX/blob/main/src/mhx/benchmarks/scaling.py)
- [Scaling validation tests](https://github.com/uwplasma/MHX/blob/main/tests/test_reconnection_scaling_validation.py)

## MHD turbulence cascade schematic

This GIF is explicitly schematic. It combines synthetic magnetic-flux eddies,
current filaments, and a $k^{-5/3}$ guide curve to communicate the kind of
turbulent MHD morphology future high-Re campaigns should target. The exponent
is used as a recognizable inertial-range guide, not as a validation result.
Classical reference points include
[Kraichnan 1958](https://doi.org/10.1103/PhysRev.109.1407), the
Iroshnikov--Kraichnan phenomenology summarized in
[Verma 2004](https://doi.org/10.1016/j.physrep.2004.07.007), and anisotropic
strong-MHD-turbulence ideas associated with Goldreich--Sridhar as discussed in
modern reviews such as
[Schekochihin 2009](https://arxiv.org/abs/0911.2581).

![MHD turbulence cascade schematic](_static/readme/mhd_turbulence_cascade.gif)

## Nonlinear validation movies

The periodic double-Harris long-run movie pair is generated by:

```bash
mhx benchmark double-harris-long-run \
  --outdir outputs/benchmarks/periodic_double_harris_seeded_long_run \
  --nx 64 --ny 64 --t-end 30 --save-every 100 --movies
```

It visualizes the validation bridge documented in
[physics validation](validation.md).
The run advances a base periodic double-Harris sheet and a seeded copy, then
tracks normalized perturbation growth, total energy, kinetic energy, peak
current, and current-density frames. It remains `claim_level = "validation"`
until convergence, seed, aspect-ratio, and duration sweeps are attached.

![Seeded double-Harris magnetic flux movie](_static/validation/periodic_double_harris_seeded_long_run/figures/periodic_double_harris_flux.gif)

![Seeded double-Harris current-density movie](_static/validation/periodic_double_harris_seeded_long_run/figures/periodic_double_harris_current.gif)

The restartable Rutherford executor movie pair is generated by:

```bash
mhx campaign rutherford-execute \
  outputs/campaigns/rutherford_production_plan \
  --max-steps 128 --movies
```

It validates fixed-scale flux/current movie writing, checkpoint metadata, and
manifested artifacts for a production executor chunk. A partial chunk is still
validation-level unless the planned duration is completed and production
convergence evidence is attached.

![Fixed-scale Rutherford flux movie](_static/validation/rutherford_production_execution/fixed_scale_flux_movie.gif)

![Fixed-scale Rutherford current-density movie](_static/validation/rutherford_production_execution/fixed_scale_current_density_movie.gif)

Source links:

- [Current-sheet validation implementation](https://github.com/uwplasma/MHX/blob/main/src/mhx/benchmarks/current_sheet.py)
- [Production executor implementation](https://github.com/uwplasma/MHX/blob/main/src/mhx/campaigns/production.py)
- [Current-sheet tests](https://github.com/uwplasma/MHX/blob/main/tests/test_current_sheet_eigenvalue_validation.py)
- [Production executor tests](https://github.com/uwplasma/MHX/blob/main/tests/test_production_campaign.py)

Regenerate the README teaser movies with:

```bash
python examples/make_readme_media.py
```

The generated files are intentionally compact:

- `docs/_static/readme/double_harris_reconnection.gif`
- `docs/_static/readme/double_harris_current_sheet.gif`
- `docs/_static/readme/harris_layer_sweep.gif`
- `docs/_static/readme/plasmoid_scaling_schematic.gif`
- `docs/_static/readme/mhd_turbulence_cascade.gif`
