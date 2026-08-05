# MHX Documentation Refactor Plan

Generated: 2026-08-05

This file is the implementation plan for a from-scratch refactor of the MHX
documentation, and the persistent execution log for that work. It complements
`plan.md`, which governs the solver rebuild. Where the two disagree about
documentation, this file wins. Every documentation pass must append a log entry
at the bottom with what changed, what was checked, and what remains.

---

## 0. Mission

Rebuild the MHX documentation so that:

- a new graduate student runs a visible reconnection simulation in under five
  minutes and understands what they are looking at;
- a reviewer finds every equation, discretization choice, tolerance, and claim
  boundary without reading source code;
- the differentiable core, the actual reason MHX exists as a JAX code, is
  taught, demonstrated, and validated in the docs instead of only asserted;
- every page reads like a person wrote it, under the writing rules in
  section 6;
- movies and figures make the physics visible, with fixed color scales,
  recorded commands, and claim levels, so engagement never outruns evidence.

The refactor keeps what already works: warnings-as-errors builds, the figure
manifest with checksums, claim-level discipline, the prose checker, and the
docs CI workflow. It replaces structure, theme, style, media format, and most
page content.

---

## 1. Ground truth (verified 2026-08-05)

Facts measured on a fresh clone at commit `b84d429`, Python 3.12, macOS,
CPU-only JAX, before this plan was written. Documentation claims must stay
consistent with these until a log entry updates them.

### 1.1 What the code is

- Two-dimensional periodic reduced MHD: magnetic flux `psi` and vorticity
  `omega`, Fourier pseudo-spectral derivatives, two-thirds dealiasing.
- Time integration: RK4 through `jax.lax.scan`, and backward Euler through
  SOLVAX Newton--Krylov with a spectral diffusion preconditioner.
- Equilibria: `PeriodicDoubleHarrisEquilibrium`, `CosineTearingEquilibrium`,
  `ZeroEquilibrium`. Physics terms and diagnostics have plugin registries.
- Parallelism: field sharding across a one-dimensional device mesh, ensemble
  parallelism over independent cases, and multi-process runs through
  `mhx.initialize_distributed()`.
- Public API: `mhx.Simulation` / `SimulationResult` / `run_ensemble`, a TOML
  config path, and a Typer CLI with `run`, `figures`, `report`, `benchmark`
  (36 subcommands covering the validation gates), `validate`, `campaign`, and
  `neural-ode` command groups.
- Output: compressed NPZ trajectories plus JSON manifests with schema
  versions, settings, diagnostics, and checksums.

### 1.2 What runs, and how fast

- `python -m pip install -e ".[dev]"` works from a clean venv.
- The README first-run example (64 x 64, 100 RK4 steps) compiles in 0.28 s and
  runs in 0.065 s on one laptop CPU device. Total script wall time about 6 s.
- `pytest -m "not slow"`: 282 tests pass in 93 s locally. CI also enforces
  coverage at or above 95 percent.
- `sphinx-build -W -b html docs docs/_build/html` passes.
- The CLI tutorial path works: `mhx init`, `mhx run`, `mhx figures --gif`,
  `mhx report`, `mhx artifact-manifest`.
- A gradient flows through the functional core today: `jax.value_and_grad` of
  final magnetic energy with respect to resistivity, through `evolve_rk4` plus
  `reduced_mhd_rhs_spectral` at 32 x 32 for 20 steps, matches a central finite
  difference to about 10 percent under float32. The remaining gap is
  float32 finite-difference noise; the docs tutorial must repeat this check
  under `JAX_ENABLE_X64` and report the tighter agreement.

### 1.3 What the current docs are

- 32 flat Markdown pages, 7517 lines, one undivided toctree, alabaster theme,
  no dark mode, no landing-page design.
- Roughly half the toctree is internal or reviewer-facing bookkeeping:
  `audit`, `paper_plan`, `paper_pipeline`, `publication_checklist`,
  `reviewer_evidence`, `campaigns`, `campaign_runner`, `long_run_evidence`,
  `nonlinear_campaign_evidence`, `seed_robust_qi`, `time_windows`,
  `neural_ode_reproducibility`, `writing_style`, `migration`, `release`,
  `api_policy`, `audit`. New users meet this before they meet the physics.
- `validation.md` is a single 1337-line page with 31 sections.
- `api.md` dumps 22 `automodule` blocks on one page.
- There is no page about differentiation. The word "differentiable" opens the
  README and the package docstring, and no public example or doc page takes a
  gradient. `examples/gallery/` has seven scripts; none differentiate.
- Media: 17 MB under `docs/_static`, all GIF and PNG. The largest GIF is
  648 KB. Frames are small, color scales are washed out, and the README
  quickstart figure looks static because the equilibrium field dominates the
  perturbation at `t_end=2`.
- `readthedocs.yaml` requests a `docs` extra that `pyproject.toml` does not
  define. `.gitignore` ignores `*.mp4` globally.
- Existing hard gates that a refactor must move in lockstep:
  `tools/check_prose.py` (style, pinned file list),
  `tests/test_docs_links.py` (pinned toctree entries, pinned source links,
  pinned verbatim sentences), `tests/test_readme_media.py`,
  `tests/test_gallery.py`, and `.github/workflows/docs.yml`.
- Open PR #5 adds compressible MHD, Kelvin--Helmholtz benchmarks, and
  differentiable notebook examples. The new structure must give that work a
  home without blocking on it.

---

## 2. Diagnosis

The current docs are accurate and disciplined but organized for the people who
built the validation program, not for the people the code wants as users. The
specific failures:

1. **Audience inversion.** Evidence bookkeeping outnumbers teaching material.
   Dedalus and DESC keep theory, tutorials, and developer records in separate
   sections; MHX interleaves them alphabetically.
2. **The differentiability hole.** The one property that separates MHX from
   Athena++, PLUTO, OpenMHD, and Dedalus is undocumented and undemonstrated.
3. **No visual pathway.** The first figure a user produces shows a nearly
   static field. The good movies exist but sit in a media appendix.
4. **Monoliths.** One 1337-line validation page and one 24-module API page
   cannot be navigated or cited.
5. **Dated presentation.** Alabaster, no dark mode, no cards, no video, GIF
   compression artifacts.
6. **Style drift risk.** `writing_style.md` and `check_prose.py` encode good
   rules over a small pinned file list; most pages are unchecked.

---

## 3. Target information architecture

Diataxis-mapped tree. Names are final unless a log entry records a change.
Everything under `docs/project/` leaves the main navigation and lands in a
collapsed "Project records" section at the bottom of the sidebar.

```text
docs/
  index.md                      # landing page, section 5.1
  getting_started/
    install.md                  # from install.md; add GPU install, x64 note
    first_run.md                # from quickstart.md; new run settings, sec 7.2
    first_movie.md              # new: result -> movie in ten lines
    troubleshooting.md          # new: JAX platforms, x64, compile times
  tutorials/                    # executed notebooks, outputs committed
    01_tearing_mode.md          # linear tearing: growth rate vs FKR theory
    02_reconnection.md          # double-Harris: islands, X/O points, movies
    03_gradients.md             # flagship: d(energy)/d(eta), FD check, x64
    04_inverse_problem.md       # optimize perturbation toward target island
    05_ensembles_and_devices.md # vmap-style ensembles, sharding, GPUs
    06_implicit_stepping.md     # backward Euler + SOLVAX, convergence fields
  how_to/
    choose_settings.md          # grid, dt, dealiasing, integrator choice
    run_from_toml.md            # from parts of tutorial.md; CLI workflow
    run_on_gpus.md              # from README parallel section
    run_multi_process.md        # from 07_multi_process context
    save_load_and_reproduce.md  # user side of output_schema.md + manifests
    extend_physics.md           # from plugins.md
    add_diagnostics.md          # from diagnostics.md registry parts
  physics/
    reduced_mhd.md              # equations, normalization, assumptions,
                                # regime of validity; from reduced_mhd.md +
                                # model_assembly.md, rewritten as explanation
    spectral_method.md          # from architecture.md spectral/time paths
    time_integration.md         # RK4, backward Euler, CFL guidance
    differentiability.md        # new: what differentiates, what does not,
                                # x64 policy, checkpointing, nonsmooth
                                # diagnostics warning (plasmoid counts)
    solvax_boundary.md          # from architecture.md: MHX vs SOLVAX contract
  validation/
    index.md                    # claim levels, how to read a gate, promotion
    exact_limits.md             # resistive decay, diffusion eigenvalue,
                                # linearized RHS consistency
    linear_tearing.md           # FKR window, growth, Delta-prime, eigenvalue,
                                # dispersion, layer, time-domain replay
    nonlinear.md                # Orszag--Tang, turbulence, energy budget,
                                # duration audit
    reconnection_campaigns.md   # double-Harris long runs, convergence,
                                # seed robustness, Rutherford executor
    scaling_theory.md           # FKR / ideal-tearing / plasmoid scaling gates
  gallery.md                    # movie showcase; every entry: movie, command,
                                # claim level, link to validation page
  reference/
    api/
      index.md                  # map of the public API
      core.md                   # simulation, ensemble, config, grids
      physics.md                # equilibria, terms, equations
      numerics.md               # spectral, time integrators, parallel
      diagnostics.md            # diagnostics + plotting
      benchmarks.md             # benchmarks, campaigns, neural_ode
      io.md                     # io, versioning
    cli.md                      # command reference with examples
    config_schema.md            # TOML schema; from config parts of docs
    output_schema.md            # trimmed from output_schema.md
    performance.md              # from performance.md + scaling data
  develop/
    contributing.md             # new: dev install, checks, PR expectations
    style.md                    # from writing_style.md + section 6 rules
    release.md                  # from release.md + migration.md + api_policy.md
    architecture.md             # internal layering; from architecture.md
  project/                      # out of main nav; collapsed section
    paper_plan.md  paper_pipeline.md  publication_checklist.md
    reviewer_evidence.md  audit.md  campaigns.md  campaign_runner.md
    long_run_evidence.md  nonlinear_campaign_evidence.md
    seed_robust_qi.md  neural_ode_reproducibility.md  time_windows.md
    literature.md  media_inventory.md   # from media.md
```

Rules:

- Every physics and validation page states equations, parameters, assumptions,
  command, outputs, tolerances, and failure modes, per `plan.md` section 14.
- Every source-code discussion links to real files with repository-relative
  paths, and `tests/test_docs_links.py` keeps enforcing existence.
- PR #5 content lands as `tutorials/07_kelvin_helmholtz.md`,
  `physics/compressible_mhd.md`, and `validation/kelvin_helmholtz.md` when it
  merges. Nothing in this plan blocks on it.

---

## 4. Toolchain

Stay on Sphinx plus MyST. A move to mkdocs-material would force converting
every MyST directive, dollar-math block, and autodoc use for a look that
Sphinx themes now match.

Changes, all in one phase:

1. **Theme: `sphinx-book-theme`.** The JAX-ecosystem norm (JAX and Flax both
   use it), built on pydata-sphinx-theme, dark mode, MyST-native, plays well
   with sphinx-design. Fallback if it disappoints: `pydata-sphinx-theme`
   directly, which xarray and scikit-learn use.
2. **Extensions to add:** `sphinx-design` (landing cards, tabs),
   `sphinx-copybutton`, `myst-nb` (replaces plain `myst_parser`; renders
   committed notebook outputs, never executes on ReadTheDocs),
   `sphinxcontrib-video` (HTML5 mp4 embeds), `sphinxcontrib-bibtex`
   (`docs/references.bib`; replaces hand-written literature lists),
   `sphinx.ext.intersphinx` (JAX, NumPy, SOLVAX), `sphinxext-opengraph`
   (social preview cards).
3. **`pyproject.toml`:** define a real `docs` extra holding the full docs
   toolchain, keep `dev` for lint/test, and make `dev` include `docs`.
4. **`readthedocs.yaml`:** point at the `docs` extra it already requests,
   set `fail_on_warning: true`, and verify the next build log on ReadTheDocs
   after merge. Configure the default version and a redirect from
   `/en/latest/` deliberately.
5. **`.gitignore`:** replace the global `*.mp4` ignore with an allowlist so
   `docs/_static/**/*.mp4` can be committed.
6. **`docs/conf.py`:** rewrite for the above; keep `dollarmath` and `amsmath`;
   turn on MyST heading anchors for stable cross-references.

Executable examples policy: tutorials are jupytext-paired MyST notebooks with
outputs committed for anything slower than a few seconds. ReadTheDocs renders;
it never executes. A scheduled CI job (monthly, and on release branches)
re-executes all tutorials on a runner with a time budget and fails on drift,
so committed outputs cannot silently rot. sphinx-gallery is rejected: it
re-runs scripts at build time, which JAX compile times and ReadTheDocs' 15
minute budget cannot absorb.

---

## 5. Landing page and README

### 5.1 `docs/index.md`

Above the fold: one sentence stating what MHX is and is not, the hero
reconnection movie (autoplay, muted, loop), install command, and a ten-line
runnable example that produces visible dynamics. Below: four sphinx-design
cards (Get started, Tutorials, Physics and validation, API reference), then
citation and SOLVAX/uwplasma links. The xarray landing page is the model.

### 5.2 `README.md`

Keep the current honest tone and claim boundaries. Changes:

- Hero: one strong reconnection movie as a committed GIF at or under 5 MB,
  plus a stills row. GitHub renders committed GIFs up to 10 MB; PyPI renders
  neither video tags nor GitHub user-attachment videos, so the README hero
  stays a GIF with an absolute URL.
- Cut the README's second half (parallel guide, integrator guide, doc table)
  down to links into the new docs sections. The README teaches the first run
  and points elsewhere for everything else.
- First-run example switches to the settings from section 7.2 so the first
  figure a user ever makes shows reconnection happening.

---

## 6. Writing rules

The existing `docs/writing_style.md` rules stay: lead with the result, active
voice, one term per object, conditions before instructions, sentence and
paragraph caps, no contractions, no semicolons, no em dashes, American
spelling, facts with settings and hardware, limits beside results.

Add the no-ai-slop rules (github.com/petergyang/no-ai-slop) that the current
checker does not cover:

1. No binary-contrast framing ("This is not X. It is Y."). State Y.
2. No colon reveals, rhetorical setups, faux-insight openers, or
   fake-profound closing lines. End sections on a concrete point.
3. No trailing `-ing` analysis clauses ("...highlighting the flexibility of
   the solver"). State the consequence or cut.
4. No importance puffery ("plays a vital role", "stands as a testament").
5. No weasel attribution ("it is well known"). Cite the reference or cut.
6. No synonym cycling. The solver is "MHX" or "the solver" everywhere.
7. No formatting slop: no emoji in headings, no bold mid-sentence for
   emphasis, no bullet lists where two sentences of prose read better, no
   heading over a two-sentence section.
8. Concrete beats abstract: numbers with hardware, commands with expected
   output, file paths over descriptions of file paths.

Extend `tools/check_prose.py`:

- Cover `README.md` and every page under `docs/` except `docs/project/` and
  generated API stubs.
- Extend `BANNED_TERMS` with the remaining no-ai-slop list: `delve` variants
  already present plus `seamless`, `powerful`, `blazingly`, `effortless`,
  `intricate`, `meticulous`, `multifaceted`, `paramount`, `transformative`,
  `elevate`, `embark`, `supercharge`, `ever-evolving`, `state-of-the-art`,
  `beacon`, `pivotal`, `vital role`, `testament`, `in conclusion`,
  `it is worth noting`, `in order to`.
- Add pattern checks for trailing `-ing` analysis clauses after commas and
  for "not X. It is Y." contrast pairs. Keep both as warnings for one release
  while pages migrate, then promote to errors.

Accuracy rule, unchanged in spirit from the current docs: every quantitative
sentence traces to a command, a test, or a manifest. When a page shows a
number this plan did not verify, the page links the artifact that did.

---

## 7. Media plan

### 7.1 Format and embedding

- Docs pages embed mp4 (H.264, yuv420p) through `sphinxcontrib-video` with
  `autoplay`, `muted`, `loop`. Target 720p and 1 to 4 MB per movie. GIFs in
  docs pages are retired; mp4 at equal quality is 5 to 10 times smaller.
- The README keeps exactly one GIF (hero) under 5 MB.
- All media stay committed in-repo under `docs/_static/`. The repository is
  public, so committed media is the distribution channel; keep total media
  under about 60 MB so clones stay fast.
- `docs/figures/manifest.toml` remains the registry: path, sha256, generating
  command, claim level, sources, tests. Every new movie gets an entry.
- The movie QA check gains a motion gate: reject any README or gallery movie
  whose inter-frame pixel change is below a threshold, so visually static
  total-field movies cannot ship again. (`tests/test_readme_media.py`
  extension; the 2026-05-22 log in `plan.md` already asks for this.)

### 7.2 The engagement fix for the first run

The README and `first_run.md` example changes from `t_end=2` (nothing visible
happens) to a seeded double-Harris run sized to show island growth within
about 30 s of laptop CPU time, with the residual-flux view
`delta_psi = psi - psi_equilibrium` in the default four-panel figure. The
residual-flux media policy from `docs/media.md` becomes the default
presentation everywhere, not a reviewer-only convention: total-field views
carry the equilibrium and look frozen; residual views show the physics.
Exact settings get fixed during phase P3 by parameter scan and recorded in
the figure manifest.

### 7.3 Movie set

Regenerate everything at 256 x 256 or higher with perceptually uniform
colormaps (sequential for magnitudes, diverging centered at zero for signed
fields), fixed color scales across frames, and labeled time. Sources and
claim levels stay anchored to the validation gates that produce them.

| Movie | Source | Placement |
| --- | --- | --- |
| Double-Harris reconnection, residual flux + current, 256 x 256, t_end 160 | GPU campaign replay | hero: index, README |
| Orszag--Tang current density | `mhx benchmark orszag-tang --movies` | gallery, validation/nonlinear |
| Orszag--Tang vorticity | same run | gallery |
| Decaying turbulence current filaments | `mhx benchmark decaying-turbulence --movies` | gallery, validation/nonlinear |
| Forced turbulent reconnection | `mhx benchmark forced-turbulent-reconnection --movies` | gallery |
| Harris tearing layer sweep vs S | `mhx benchmark linear-tearing-layer` | validation/linear_tearing |
| Island growth with X/O tracking | tutorial 02 notebook | tutorials/02 |
| Gradient descent on perturbation amplitude | tutorial 04 notebook | tutorials/04, gallery |
| Ensemble sharding timeline | tutorial 05 | reference/performance |

CPU-scale movies (64 to 128 squared) regenerate locally. The 256 x 256 hero
and campaign replays run on the office GPUs (two RTX A4000, verified in the
`plan.md` 2026-05-22 log). Schematic-only GIFs (`plasmoid_scaling_schematic`,
`mhd_turbulence_cascade`) are dropped from README and gallery; they may stay
in `docs/project/media_inventory.md` labeled as theory schematics.

---

## 8. Page migration map

Every current page has exactly one destination. "Split" means content divides
across the listed targets; nothing is deleted without a destination or an
explicit drop note.

| Current page | Destination |
| --- | --- |
| index.md | rewritten landing page (5.1) |
| install.md | getting_started/install.md |
| quickstart.md | getting_started/first_run.md |
| tutorial.md | split: how_to/run_from_toml.md, getting_started/first_movie.md |
| architecture.md | split: physics/spectral_method.md, physics/solvax_boundary.md, develop/architecture.md |
| reduced_mhd.md | physics/reduced_mhd.md (rewritten as explanation, benchmark specifics move to validation pages) |
| model_assembly.md | physics/reduced_mhd.md |
| diagnostics.md | split: how_to/add_diagnostics.md, reference/api/diagnostics.md |
| output_schema.md | split: how_to/save_load_and_reproduce.md, reference/output_schema.md |
| validation.md | split into validation/{exact_limits,linear_tearing,nonlinear,reconnection_campaigns,scaling_theory}.md |
| benchmarks.md | split: reference/cli.md, validation/index.md |
| performance.md | reference/performance.md |
| media.md | split: gallery.md, project/media_inventory.md |
| plugins.md | how_to/extend_physics.md |
| literature.md | replaced by references.bib + inline citations; archive copy in project/ |
| writing_style.md | develop/style.md (extended, section 6) |
| release.md, migration.md, api_policy.md | develop/release.md |
| api.md | reference/api/*.md (six pages) |
| audit, paper_plan, paper_pipeline, publication_checklist, reviewer_evidence, campaigns, campaign_runner, long_run_evidence, nonlinear_campaign_evidence, seed_robust_qi, time_windows, neural_ode_reproducibility | docs/project/, content preserved |

### 8.1 Gate migration

`tests/test_docs_links.py` is rewritten in the same PR as each structural
move, never after. The rewrite keeps the three durable protections and drops
the brittle ones:

- keep: image links resolve; source-code links exist; required pages present
  in navigation (updated names).
- keep: claim-level vocabulary asserted on validation/index.md and
  project/reviewer_evidence.md.
- drop: pinned verbatim sentences ("duration-complete validation bundle" and
  similar). Replace with assertions that the relevant section headings exist,
  so wording can improve without test edits.

`tools/check_prose.py` file list grows per phase P2. `docs.yml` gains the
tutorial re-execution job (section 4). `tests/test_gallery.py` extends to any
new example scripts.

---

## 9. Examples and tutorials

- `examples/gallery/` stays the script-first entry point and gains
  `08_gradient.py`: value-and-grad of final magnetic energy with respect to
  resistivity, with an x64 finite-difference check printed at the end. This
  is the code from the verified 2026-08-05 ground-truth run, cleaned up.
- Tutorials (section 3) are notebooks; each begins with the exact runtime and
  hardware of the committed execution and ends with links to the validation
  pages that bound its claims.
- Tutorial 03 (gradients) and 04 (inverse problem) fulfill `plan.md` section
  13.1 demos 1 and 2 at reduced scope and must include the section 13.2
  gradient-validation checklist: finite-difference check, dtype check,
  nonsmoothness warning.
- `examples/README.md` and `examples/gallery/README.md` shrink to tables:
  script, one-line purpose, wall time, output.

---

## 10. Execution phases

Each phase is one PR, keeps `sphinx-build -W`, `check_prose`, `pytest -m "not
slow"`, and `ruff` green, and appends a log entry here. Order is fixed;
phases must not be combined.

- **P0. Toolchain.** Theme, extensions, `docs` extra, `readthedocs.yaml`,
  `.gitignore` mp4 allowlist, conf.py rewrite. Pages untouched. Verify the
  ReadTheDocs build log renders the new theme.
  Acceptance: site builds on RTD with sphinx-book-theme, dark mode works,
  all 32 existing pages render.
- **P1. Structure.** Create the section 3 tree, move pages with `git mv`,
  split the two monoliths mechanically (content unchanged), rewrite
  `index.md`, demote `docs/project/`, rewrite `test_docs_links.py` (8.1).
  Acceptance: navigation shows the Diataxis sections; no page lost (old name
  redirects or an explicit map in the PR body); docs tests pass.
- **P2. Content rewrite.** Page by page through getting_started, physics,
  how_to, validation, reference, develop, applying section 6 and checking
  every command and number against the code. Extend `check_prose.py`
  coverage and rules in the same PR as each batch.
  Acceptance: every rewritten page passes the extended checker; every
  command in rewritten pages executed at least once locally.
- **P3. Media.** Regenerate the movie set (7.3), convert docs embeds to mp4,
  build the gallery page, fix the first-run settings (7.2), add the motion
  QA gate, update the figure manifest and README hero.
  Acceptance: no GIFs referenced from docs pages except the README hero;
  every gallery entry has manifest, command, claim level; motion gate passes.
- **P4. Tutorials.** Write notebooks 01 through 06, commit executed outputs,
  add the scheduled re-execution job, add `08_gradient.py` to the gallery
  scripts.
  Acceptance: tutorials render on RTD with outputs; re-execution job passes
  on a manual dispatch; gradient tutorial shows an x64 finite-difference
  match.
- **P5. README.** Rewrite per 5.2 with the new hero.
  Acceptance: `check_prose.py` and `test_readme_media.py` pass; README shows
  install, first run, hero movie, links, citation, and nothing else.
- **P6. Polish and release.** RTD default version and redirects, opengraph
  cards, CHANGELOG entry, version tag per `develop/release.md`.
  Acceptance: fresh-eyes pass, RTD `/en/latest/` and `/en/stable/` resolve,
  a clean clone follows first_run.md start to finish.

Dependencies: P1 needs P0. P2 through P4 can interleave after P1 but each PR
stays single-purpose. P5 needs P3. P6 is last. The office-GPU renders in P3
can trail the rest of P3 by one PR if the machines are busy.

---

## 11. Out of scope

- Solver features, physics modules, and anything in `plan.md` sections 5
  through 13 beyond what documentation demonstrates.
- Comparison-with-other-codes docs (`plan.md` section 12): those pages come
  with the comparison harness, not before it.
- Notebook-hosting services, binder links, external video hosting.
- PR #5 review and merge (its docs land in the reserved slots afterward).

---

## 12. Log

### 2026-08-05 — Plan created

- Cloned fresh at `b84d429`, installed, ran the README example, the CLI
  tutorial path, and the fast test suite (282 passed, 93 s). Built docs with
  warnings-as-errors. Verified a gradient through the functional core with a
  float32 finite-difference check.
- Surveyed Diffrax, Equinox, JAX-Fluids, astronomix, Dedalus, Gkeyll,
  jax-cfd, simsopt, DESC, xarray, and scikit-learn docs; chose Sphinx +
  sphinx-book-theme + MyST-NB + sphinx-design + sphinxcontrib-video; mapped
  the tree to Diataxis; merged the no-ai-slop rule set into the prose gate
  plan.
- Next: execute P0.

### 2026-08-05 — P0 complete: toolchain

- Switched the theme to `sphinx-book-theme` and rewrote `docs/conf.py`:
  `myst_nb` replaces plain `myst_parser` with `nb_execution_mode = "off"`,
  plus `sphinx_design`, `sphinx_copybutton`, `sphinxcontrib.video`,
  `sphinxcontrib.bibtex`, `sphinx.ext.intersphinx` (JAX, NumPy, Python), and
  `sphinxext.opengraph`. Heading anchors on to depth 3.
- Added a real `docs` extra to `pyproject.toml`; `dev` now includes
  `mhx[docs]`, so `pip install -e ".[dev]"` keeps covering CI and
  ReadTheDocs keeps requesting `docs`. Set `fail_on_warning: true` in
  `readthedocs.yaml`.
- Seeded `docs/references.bib` with the citations the physics and validation
  pages will use (FKR, Rutherford, Strauss, Sweet, Parker, Loureiro,
  Pucci--Velli, Orszag--Tang, Biskamp, Orszag 2/3 rule, Canuto, JAX,
  Knoll--Keyes, Saad--Schultz GMRES, White).
- Fixed `.gitignore`: the docs allowlist now un-ignores
  `docs/_static/**/*.mp4`; the stray second global `*.mp4` line is gone.
- Updated `tests/test_packaging.py` to expect the three extras and to pin
  the dev-includes-docs contract.
- Checks: `sphinx-build -W` passes on the new toolchain with all 32 pages,
  `check_prose.py` passes, docs/link/media/packaging tests pass, `ruff`
  passes. Deviation from the plan: phases land as direct commits to `main`
  rather than PRs, recorded here once instead of per phase.
- Next: P1 restructure.

### 2026-08-05 — P1 complete: structure

- Moved every page into the section 3 tree with `git mv`: getting_started,
  how_to, physics, validation, reference (with `reference/api/`), develop,
  and project. Merged `release.md`, `migration.md`, and `api_policy.md` into
  `develop/release.md` with demoted headings.
- Split `validation.md` into `validation/{index,exact_limits,linear_tearing,
  nonlinear,reconnection_campaigns,scaling_theory}.md` by section, content
  unchanged. Split `api.md` into seven `reference/api/` pages with short
  layer introductions, and added `mhx.time_integrators` to the reference.
- Rewrote `docs/index.md` as the landing page: hero movie, install, runnable
  example, four sphinx-design cards, grouped hidden toctrees with captions.
  `docs/project/` renders as a collapsed "Project records" sidebar group.
- Rewrote link targets mechanically (script): page-to-page links through an
  old-to-new map with relative-path computation, `_static` and data-file
  links by depth, `{image}` directives and one `{doc}` role by hand.
- Gates moved in lockstep: `tests/test_docs_links.py` rewritten (toctree
  coverage, orphan check, area-level source links, image existence, claim
  levels, split-page heading preservation; verbatim sentence pins dropped),
  `tools/check_prose.py` path list updated and directive-option lines now
  stripped before checks, `tests/test_readme_media.py` media page path
  updated, `src/mhx/benchmarks/release_candidate.py` REQUIRED_DOCS updated,
  `tools/nonlinear_campaign_evidence.py` output defaults moved to
  `docs/project/`, `docs/figures/manifest.toml` source paths updated.
- Checks: `sphinx-build -W` passes, `check_prose.py` passes, full fast suite
  282 passed, `ruff` passes.
- Next: P2 content rewrite with equations, derivations, and citations.

### 2026-08-05 — P2 batch 1: physics, getting started, validation index

- Rewrote `physics/reduced_mhd.md` as the model centerpiece: the equations
  exactly as coded, the symbol table, the derivation from incompressible
  visco-resistive MHD through the bracket identities, Alfvén-unit
  normalization with S, Re, and Pm, conservation laws with the exact
  dissipation balance, all three equilibrium formulas, the physics-term
  table, and explicit assumptions and limits. Citations: Strauss 1976,
  Biskamp 2000/2003, FKR 1963, Harris 1962, Sweet, Parker, Loureiro 2007,
  Pucci--Velli 2014.
- Added `physics/spectral_method.md` (Fourier operators, zero-mode gauge,
  Orszag two-thirds rule, batched transforms), `physics/time_integration.md`
  (RK4 formulas, step-choice limits with the RK4 stability bound, backward
  Euler with Newton--Krylov and GMRES citations),
  `physics/differentiability.md` (worked value-and-grad example, gradient
  validation checklist, nondifferentiable diagnostics list, checkpointing
  note), and `physics/solvax_boundary.md` (ownership table and the residual
  seam).
- Rewrote `getting_started/install.md` (GPU wheel, x64 policy, verify path)
  and `getting_started/first_run.md` (per-setting physics meaning, real
  output excerpt, island-flux view). Added `first_movie.md` (CLI route and a
  tested imageio loop) and `troubleshooting.md` (six first-contact
  failures). The island visualization subtracts the y-averaged profile,
  which isolates the growing mode. Subtracting the initial state instead
  shows mostly resistive background diffusion. Both scripts were executed:
  the island amplitude grows from 0.006 to 0.013 over t_end=40 at 64x64.
- Rewrote `validation/index.md` around how to read a gate and the claim
  levels table. Added `how_to/run_on_gpus.md` and `reference/config_schema.md`
  (assembly TOML from the old model_assembly page). Removed
  `physics/model_assembly.md` after redistributing its content. Added
  `reference/bibliography.md` backed by `references.bib`, alpha labels.
- Extended `tools/check_prose.py`: strips display and inline math, covers 20
  documents, and bans the remaining no-ai-slop terms.
- Checks: `check_prose` 20 documents pass, `sphinx-build -W` passes with all
  citations resolving, 282 fast tests pass, `ruff` passes.
- Remaining in P2: light passes over how_to/run_from_toml, extend_physics,
  add_diagnostics, reference/cli intro, and a develop/architecture trim.

### 2026-08-05 — P2 batch 2: how-to titles and architecture trim

- Retitled and reframed the moved pages for their new roles:
  `how_to/run_from_toml.md` (was the tutorial page),
  `how_to/extend_physics.md`, `how_to/add_diagnostics.md`, and
  `reference/cli.md`.
- Trimmed `develop/architecture.md`: the spectral, time, and SOLVAX sections
  now point to the physics pages instead of duplicating them. Added the four
  JAX design rules. Kept the layering table and device path.
- Checks: `check_prose` passes, `sphinx-build -W` passes, docs and release
  gates pass. P2 is complete except for content that P3 and P4 own, movie
  embeds and tutorials.
- Next: verify the ReadTheDocs build of the new theme, then P3 media.

### 2026-08-05 — P3 CPU slice: mp4 embeds, gallery, motion gate

- Verified the ReadTheDocs deployment renders sphinx-book-theme with the
  landing cards and hero before starting P3.
- Added `examples/make_docs_movies.py`: transcodes the committed validated
  GIF movies to H.264 mp4 under `docs/_static/movies/` (frames and claim
  levels unchanged, sizes drop about 4x) and renders the island movie at the
  exact `first_movie.md` settings. Ten movies, 780 KiB total. Added
  `imageio-ffmpeg` to the dev extra so CI carries its own ffmpeg.
- Docs pages now embed mp4 through `sphinxcontrib-video`: the landing hero
  autoplays muted in a loop, the gallery page (`docs/gallery.md`) shows all
  ten movies with commands, claim boundaries, and citations, and
  `first_movie.md` shows the expected output of its own script.
- Added `tests/test_docs_movies.py`: video targets resolve, movies stay
  under 6 MB, docs pages outside `project/` embed no GIFs, every committed
  movie has a figure-manifest entry, and a motion gate rejects visually
  static movies. The gate immediately rejected the historical seeded
  double-Harris total-flux movie (mean inter-frame motion 0.44 against the
  0.5 floor), which is the media policy working. That movie stays in the
  project records; the validation page embeds the current-density view with
  a note. Wired the test into the docs workflow.
- Manifest: eleven new figure entries with sha256, command, claim level,
  sources, and tests. The island render is labeled `smoke` because it is a
  real solver run without validation gates.
- Checks: `check_prose` 21 documents, `sphinx-build -W`, `ruff`, and the
  fast suite (296 passed) are green.
- Remaining in P3, office-GPU pass: re-render the double-Harris hero and the
  Orszag--Tang set at 256 x 256, refresh `strong_scaling.png` styling, and
  regenerate the README GIF hero from the new render.
- Next: P4 tutorials.

### 2026-08-05 — P4 slice: gradient script and gradients tutorial

- Added `examples/gallery/08_gradient.py`: value-and-grad of a final
  magnetic-energy proxy with respect to resistivity through a 20-step RK4
  solve, under forced float64, with central-difference checks at two step
  sizes. Measured agreement: relative error 3.7e-11 at eps=1e-5. The float32
  run of the same check earlier in this log disagreed at the percent level,
  which confirms the x64 policy. `tests/test_gallery.py` now admits the
  functional-core script shape and requires its self-check.
- Added `docs/tutorials/03_gradients.ipynb`, executed locally with committed
  outputs: the loss over a solve, the gradient, the finite-difference table,
  and a resistivity scan with the tangent line drawn from `value_and_grad`.
  MyST-NB renders it without executing. Tutorials group added to the
  navigation.
- Added `tools/execute_tutorials.py` and a monthly plus on-demand
  `Tutorials` workflow that re-executes every notebook and fails on
  execution errors. Output drift shows in the job log as a diffstat. Exact
  output comparison is deliberately not enforced yet: timings and float
  last digits differ across machines. Revisit after the first scheduled
  runs.
- Checks: `sphinx-build -W` renders the notebook with outputs, prose and
  gallery and docs gates pass, fast suite 296 passed, `ruff` clean.
- Remaining in P4: tutorials 01, 02, 04, 05, 06 following the same recipe.
- Next: P5 README rewrite.

### 2026-08-05 — P5: README rewrite and the first-run engagement fix

- Closed plan item 7.2. The default `SimulationResult.plot` second panel now
  shows the island flux, the deviation of the final flux from its y average,
  with total-flux contours overlaid. Before this change the panel showed the
  total final flux, which is visually identical to the initial condition
  even at `t_end=40`. No test pinned the panel titles.
- The README, landing page, and `first_run.md` example moved from
  `t_end=2.0, save_every=10` to `t_end=40.0, save_every=100`: 2000 RK4
  steps, 1.3 s run time on a laptop CPU, and a clearly grown island in the
  produced figure. Documented output numbers were re-measured (final energy
  2.389953e-01). `first_run.md` section 4 now quantifies the island growth
  instead of working around a static figure.
- Rewrote the README per plan 5.2: badges, mission, hero movie, first run,
  gallery table with `08_gradient.py`, the three-movie validation row,
  short physics and parallel sections, a trimmed documentation table, and
  development commands. The integrator guide and the long parallel guide
  are now links into the docs. All release-candidate README markers and
  the media-test constraints kept passing without gate edits.
- Checks: `check_prose` 21 documents, `sphinx-build -W`, fast suite 296
  passed, `ruff` clean.
- Remaining: P4 tutorials 01, 02, 04, 05, 06. P6 release polish. Office-GPU
  media pass.
