# Reproducible paper pipeline

The paper pipeline is the single-command bundle for reviewer-facing validation
artifacts. It runs the deterministic FAST validation suite, writes the
release-vs-publication readiness report, and then records a top-level manifest
and recursive SHA-256 artifact manifest.

```bash
mhx validate paper-pipeline --outdir outputs/paper_pipeline
python examples/tools/verify_paper_artifacts.py \
  --artifact-root docs/_static/validation \
  --artifact-root outputs/paper_pipeline
```

The default command executes every active validation-suite case. For quick CI
or local command-shape checks, use an explicit subset and keep the subset
boundary in the manifest:

```bash
mhx validate paper-pipeline \
  --outdir outputs/paper_pipeline_smoke \
  --cases resistive_decay,fkr_window \
  --no-require-release-ready
```

## Expected files

The command writes:

| File | Purpose |
| --- | --- |
| `paper_pipeline.json` | Machine-readable pipeline metadata with schema `mhx.paper_pipeline.v1`. |
| `paper_pipeline.md` | Human-readable gate summary and included validation cases. |
| `validation.json` | Gate result with schema `mhx.paper_pipeline.gates.v1`. |
| `validation_suite/validation_suite.json` | Per-case validation-suite summary. |
| `validation_suite/artifact_manifest.json` | Recursive checksums for validation-suite artifacts. |
| `readiness/readiness.json` | Public-release and publication-claim readiness assessment. |
| `readiness/figures/readiness_matrix.png` | Visual release-vs-paper gate matrix. |
| `artifact_manifest.json` | Recursive checksums for the whole paper-pipeline bundle. |
| `manifest.json` | Top-level claim metadata and stable output paths. |

Every manifest is claim-scoped as `validation`. This is intentional: the
pipeline can prove that the repository is public-release ready and that FAST
validation figures are reproducible, but it cannot promote nonlinear
Rutherford, plasmoid, or turbulent reconnection claims without the separate
production promotion gates described in the publication checklist.

The companion verifier checks the static paper figure manifest and any
recursive artifact manifests. It is a hash and traceability gate, not a physics
promotion gate: a validation figure with matching checksums remains
`validation` until the production evidence in the claim boundary below exists.

For a release handoff, run the static repository gate against the generated
readiness report:

```bash
mhx validate release-candidate \
  --outdir outputs/release_candidate \
  --readiness outputs/paper_pipeline/readiness \
  --require-readiness
```

That command checks packaging, CI, ReadTheDocs, citation/version metadata,
examples, docs, README claim boundaries, active legacy-import hygiene, and
documentation figure hashes.

## Claim boundary

The validation suite includes linear theory gates, manufactured linearized-RHS
checks, nonlinear energy-budget checks, double-Harris validation runs,
seed-robust quality indicators, neural-ODE reproducibility bundles, and
campaign executor smoke tests. These support method and workflow claims. A
paper-level nonlinear reconnection claim additionally requires:

1. duration-complete nonlinear histories;
2. resolution, time-step, parameter, and seed sweeps;
3. fixed-scale movies and X/O critical-point histories;
4. positive reconnecting-flux and island-width response gates;
5. a passing promotion-readiness report; and
6. an artifact manifest with hashes for every plotted file.

The source implementation is
[`paper_pipeline.py`](https://github.com/uwplasma/MHX/blob/main/src/mhx/benchmarks/paper_pipeline.py),
and CLI wiring lives in
[`main.py`](https://github.com/uwplasma/MHX/blob/main/src/mhx/cli/main.py).
The documentation and artifact-manifest verifier lives in
[`verify_paper_artifacts.py`](https://github.com/uwplasma/MHX/blob/main/examples/tools/verify_paper_artifacts.py).
