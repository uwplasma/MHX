# Nonlinear campaign evidence claims

This generated table records the local nonlinear campaign evidence inspected for the
double-Harris, Rutherford, and forced turbulent-reconnection lanes. It is deliberately
conservative: every row remains below production physics claim level.

Regenerate with:

```bash
python tools/nonlinear_campaign_evidence.py \
  --output-json docs/nonlinear_campaign_evidence.json \
  --output-md docs/nonlinear_campaign_evidence.md
```

Summarize one candidate campaign bundle with:

```bash
python tools/nonlinear_campaign_evidence.py \
  --campaign-dir outputs/campaigns/rutherford_production \
  --output-json outputs/campaigns/rutherford_production/promotion_gate_summary.json \
  --output-md outputs/campaigns/rutherford_production/promotion_gate_summary.md \
  --require-production-ready
```

The per-campaign summary fails closed: `gate_ready` is true only if the duration,
convergence, seed-QI, X/O critical-point, reconnecting-flux, island-width,
fixed-scale movie, and manifest gates all pass. `production_claim_ready` also
requires the upstream promotion report to allow `claim_level_if_passed = "production"`.
The X/O gate requires both detected X points and detected O points rather than only
the presence of count arrays.

Seeded double-Harris long-run bundles are supported as validation-promotion inputs:
the summarizer reads `periodic_double_harris_seeded_long_run.npz`,
`figures/periodic_double_harris_flux.gif`, and
`figures/periodic_double_harris_current.gif` when those files are present. These
bundles can be `gate_ready` for their declared validation media claim while still
leaving `production_claim_ready = false`.

## Latest bounded GPU validation gate

The latest inspected GPU bundle is
`outputs/campaigns/gpu_nonlinear_20260522_085049/double_harris_long_n128_t160`.
Its local gate summary reports:

- `gate_ready = true`;
- `production_claim_ready = false`;
- `claim_level_if_gates_pass = "validation"`;
- `reconnected_flux_amplification = 8.35649`;
- `island_width_amplification = 2.89076`;
- `max_x_point_count = 4` and `max_o_point_count = 2`.

The ready gates are duration, convergence, X/O critical points, reconnecting
flux, island width, fixed-scale movies, and manifests. This is the strongest
current double-Harris validation-media bundle, but it is not a production
Rutherford, Sweet-Parker, or plasmoid-chain result because the attached
promotion report declares `claim_level_if_passed = "validation"`.

Recreate that gate summary with:

```bash
python tools/nonlinear_campaign_evidence.py \
  --campaign-dir outputs/campaigns/gpu_nonlinear_20260522_085049/double_harris_long_n128_t160 \
  --output-json outputs/campaigns/gpu_nonlinear_20260522_085049/double_harris_long_n128_t160/promotion_gate_summary.json \
  --output-md outputs/campaigns/gpu_nonlinear_20260522_085049/double_harris_long_n128_t160/promotion_gate_summary.md
```

The full input bundle was reproduced with:

```bash
RUN_DIR=outputs/campaigns/gpu_nonlinear_20260522_085049

python3 -m mhx.cli.main benchmark double-harris-long-run \
  --outdir "$RUN_DIR/double_harris_long_n128_t160" \
  --nx 128 --ny 128 --width 0.36 --eta 0.0045 --nu 0.0045 \
  --perturbation-amplitude 0.004 --mode-x 2 --mode-y 1 \
  --dt 0.02 --t-end 160 --save-every 100 \
  --fit-start 0 --fit-stop 16 \
  --min-early-growth-rate 1e-9 \
  --min-max-growth-factor 1.000000001 \
  --min-reconnected-flux-amplification 1.000000001 \
  --min-island-width-amplification 1.000000001 \
  --movies

python3 -m mhx.cli.main benchmark double-harris-convergence \
  --outdir "$RUN_DIR/double_harris_convergence_n64_96_128" \
  --resolutions 64,96,128 --dt-values 0.02,0.01 \
  --reference-resolution 96 --reference-dt 0.02 \
  --width 0.36 --eta 0.0045 --nu 0.0045 \
  --perturbation-amplitude 0.004 --mode-x 2 --mode-y 1 \
  --t-end 32 --save-interval 2 --fit-start 0 --fit-stop 12 \
  --min-early-growth-rate 1e-9 \
  --min-max-growth-factor 1.000000001 \
  --max-relative-growth-rate-spread 10 \
  --max-relative-max-growth-spread 20 \
  --max-relative-flux-amplification-spread 20 \
  --max-relative-width-amplification-spread 20

python3 -m mhx.cli.main benchmark double-harris-parameter-sweep \
  --outdir "$RUN_DIR/double_harris_width_sweep" \
  --sweep-axis width --widths 0.30,0.36,0.42 \
  --nx 96 --ny 96 --eta 0.0045 --nu 0.0045 \
  --perturbation-amplitude 0.004 --mode-x 2 --mode-y 1 \
  --dt 0.02 --t-end 32 --save-interval 2 \
  --fit-start 0 --fit-stop 12 \
  --min-early-growth-rate 1e-9 \
  --min-max-growth-factor 1.000000001 \
  --max-relative-growth-rate-spread 20 \
  --max-relative-max-growth-spread 40

python3 -m mhx.cli.main benchmark double-harris-parameter-sweep \
  --outdir "$RUN_DIR/double_harris_eta_sweep" \
  --sweep-axis resistivity --etas 0.0035,0.0045,0.0060 \
  --viscosities 0.0035,0.0045,0.0060 \
  --nx 96 --ny 96 --width 0.36 \
  --perturbation-amplitude 0.004 --mode-x 2 --mode-y 1 \
  --dt 0.02 --t-end 32 --save-interval 2 \
  --fit-start 0 --fit-stop 12 \
  --min-early-growth-rate 1e-9 \
  --min-max-growth-factor 1.000000001 \
  --max-relative-growth-rate-spread 20 \
  --max-relative-max-growth-spread 40

python3 -m mhx.cli.main benchmark seed-robust-qi \
  --outdir "$RUN_DIR/seed_robust_qi_n32_t0p12" \
  --seeds 0,1,2,3,4,5 --nx 32 --ny 32 \
  --t-end 0.12 --dt 0.01 --eta 0.0045 --nu 0.0045 \
  --noise-amplitude 1e-6

python3 -m mhx.cli.main benchmark double-harris-promotion-check \
  "$RUN_DIR/double_harris_long_n128_t160" \
  --outdir "$RUN_DIR/double_harris_long_n128_t160/promotion" \
  --convergence-dir "$RUN_DIR/double_harris_convergence_n64_96_128" \
  --require-movies --min-history-samples 50 \
  --min-convergence-dirs 1 --min-t-end 120 \
  --min-reconnected-flux-amplification 1.05 \
  --min-island-width-amplification 1.05 \
  --max-relative-energy-increase 1e-8

python3 -m mhx.cli.main artifact-manifest "$RUN_DIR"
```

## Claim table

| Lane | Artifact | Validation | Readiness gate | Key metrics | Production blocker |
| --- | --- | --- | --- | --- | --- |
| periodic double-Harris | `outputs/nonlinear_campaign_evidence_20260522/double_harris_convergence_n16_24_t8` | pass | not production-ready | `case_count`=4, `t_end`=8.0, `resolution_growth_rate_spread`=0.0897579, `timestep_growth_rate_spread`=1.84821e-10, `resolution_max_growth_spread`=0.500001, `timestep_max_growth_spread`=3.93251e-10 | Small 16/24 grid and short t=8 validation sweep; production needs larger resolution, duration, seed, width/aspect, and Lundquist sweeps. |
| periodic double-Harris | `outputs/nonlinear_campaign_evidence_20260522/double_harris_long_n48_t24` | pass | not production-ready | `shape`=[48, 48], `t_end`=24.0, `samples`=13, `fitted_early_growth_rate`=0.221796, `max_growth_factor`=7.35302, `reconnected_flux_amplification`=5.89309, `island_width_amplification`=2.42757, `relative_energy_increase`=0.0 | Positive response is validation evidence only; no production-duration convergence, seed-QI, aspect-ratio, or Lundquist sweep is attached. |
| Rutherford executor | `outputs/nonlinear_campaign_evidence_20260522/rutherford_fast_n24_t1` | pass | not production-ready | `shape`=[24, 24], `t_end`=1.0, `steps`=100, `samples`=21, `max_relative_energy_growth`=-9.98053e-05, `max_magnetic_divergence_linf`=0.0 | FAST schema/diagnostic run is far shorter than Rutherford-duration requirements and cannot be promoted to nonlinear physics. |
| forced turbulent reconnection | `outputs/nonlinear_campaign_evidence_20260522/forced_turbulent_reconnection_n24_t4` | pass | validation-ready | `shape`=[24, 24], `t_end`=4.0, `samples`=21, `reconnection_proxy_change`=1.91639, `max_abs_reconnection_rate_proxy`=3.56186, `current_linf_growth`=0.273459, `max_relative_energy_growth`=0.0, `max_magnetic_divergence_linf`=0.0, `promotion_ready`=True, `history_sample_count`=21 | 2-D reduced-MHD proxy and single deterministic seed; no turbulent ensemble, 3-D physics, inertial range, or LV99 scaling evidence. |
| periodic double-Harris | `outputs/campaigns/double_harris_convergence_gpu_n32_48_64_t16_20260519_173637` | pass | not production-ready | `case_count`=5, `t_end`=16.0, `resolution_growth_rate_spread`=0.00411786, `timestep_growth_rate_spread`=1.96263e-10, `resolution_max_growth_spread`=0.00162281, `timestep_max_growth_spread`=2.05568e-10 | Medium validation sweep is not a production-scale duration, seed, aspect-ratio, or Lundquist campaign. |
| periodic double-Harris | `outputs/campaigns/gpu_nonlinear_20260522_085049/double_harris_long_n128_t160` | pass | gate-ready; production false | `shape`=[128, 128], `width`=0.36, `resistivity`=0.0045, `viscosity`=0.0045, `t_end`=160.0, `samples`=81, `fitted_early_growth_rate`=0.0716284, `max_growth_factor`=8.62337, `reconnected_flux_amplification`=8.35649, `island_width_amplification`=2.89076, `max_x_point_count`=4, `max_o_point_count`=2, `relative_energy_increase`=0.0, `promotion_ready`=True, `history_sample_count`=81 | Convergence-backed validation media passes duration, X/O, flux, width, movie, and manifest gates, but the attached promotion report declares claim_level_if_passed=validation; it is not a production Rutherford, Sweet-Parker, or plasmoid-chain claim. |
| periodic double-Harris | `outputs/campaigns/growing_double_harris_gpu_96_t120_20260518_044120` | pass | validation-ready | `shape`=[96, 96], `t_end`=120.0, `samples`=41, `fitted_early_growth_rate`=0.186895, `max_growth_factor`=7.35301, `reconnected_flux_amplification`=6.47122, `island_width_amplification`=2.54386, `max_x_point_count`=4, `max_o_point_count`=2, `relative_energy_increase`=0.0, `promotion_ready`=True, `history_sample_count`=41 | Convergence-backed validation media only; production claims still need larger seed, width/aspect, Lundquist, and duration sweeps. |
| Rutherford executor | `outputs/campaigns/rutherford_current_schema_96_dt005_20260517_161235` | pass | not production-ready | `shape`=[96, 96], `end_step`=45802, `target_step`=45802, `history_samples`=202, `max_relative_energy_growth`=-2.05077e-06, `final_magnetic_divergence_linf`=0.0, `promotion_ready`=False, `reconnected_flux_amplification`=1.0, `island_width_amplification`=1.0, `max_x_point_count`=2, `max_o_point_count`=1, `terminal_step`=45802, `history_sample_count`=202 | Duration target completed, but promotion failed because reconnecting-flux and island-width amplification remained 1.00. |
| forced turbulent reconnection | `outputs/readme_media/forced_turbulent_reconnection_64_t80_wide` | pass | not production-ready | `shape`=[64, 64], `t_end`=80.0, `samples`=41, `reconnection_proxy_change`=1.38249, `max_abs_reconnection_rate_proxy`=0.343463, `current_linf_growth`=13.829, `max_relative_energy_growth`=0.0, `max_magnetic_divergence_linf`=0.0 | Validation media only: single deterministic 2-D proxy run, no ensemble or 3-D turbulent-reconnection scaling. |

## Claim boundary

- Passing double-Harris rows support validation-level nonlinear response and convergence scaffolding, not Rutherford/plasmoid production physics.
- Passing Rutherford rows support executor/schema/duration mechanics unless the promotion report passes with positive response, convergence, seed-QI, geometry, and media gates.
- Passing forced turbulent-reconnection rows support 2-D reduced-MHD proxy-media readiness only, not 3-D turbulent-reconnection or LV99 scaling claims.
- Large binary outputs remain under `outputs/`, which is git-ignored; this page and the JSON summary are the small review artifacts.

## Remaining production blockers

- Latest GPU double-Harris gate: production remains blocked by
  `claim_level_if_passed = "validation"`, no production Rutherford or plasmoid
  target, and missing production-scale seed/aspect/Lundquist/duration sweeps.
- Rutherford duration-complete executor: production remains blocked by
  reconnecting-flux and island-width amplification both staying at `1.00`, even
  though duration, convergence, seed-QI, movies, and manifests are present.
- Forced turbulent reconnection: production remains blocked by the 2-D
  reduced-MHD proxy setup, single deterministic seed, no ensemble, no 3-D
  turbulent-reconnection physics, and no LV99 scaling evidence.
