# Validation movies

The README movies are deliberately small, deterministic GIFs that are safe to
ship in the repository. They are meant to communicate the validation roadmap
quickly without implying that MHX already has production nonlinear plasmoid
simulations.

## Harris tearing layer sweep

This GIF is generated from `mhx benchmark linear-tearing-layer`. It uses the
direct finite-domain Harris-sheet eigenproblem and shows the FAST validation
trend: increasing Lundquist number reduces the growth rate and narrows the
localized flow/current response near the resonant surface. The anchor is the
classical tearing-mode picture from Furth, Killeen & Rosenbluth and the
reduced-MHD Harris eigenproblem used in the MacTaggart validation papers.

![Harris tearing layer sweep](_static/readme/harris_layer_sweep.gif)

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

Regenerate both movies with:

```bash
python examples/make_readme_media.py
```

The generated files are intentionally compact:

- `docs/_static/readme/harris_layer_sweep.gif`
- `docs/_static/readme/plasmoid_scaling_schematic.gif`
