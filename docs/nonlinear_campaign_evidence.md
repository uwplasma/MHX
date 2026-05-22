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

## Claim table

| Lane | Artifact | Validation | Readiness gate | Key metrics | Production blocker |
| --- | --- | --- | --- | --- | --- |
| periodic double-Harris | `outputs/nonlinear_campaign_evidence_20260522/double_harris_convergence_n16_24_t8` | pass | not production-ready | `case_count`=4, `t_end`=8.0, `resolution_growth_rate_spread`=0.0897579, `timestep_growth_rate_spread`=1.84821e-10, `resolution_max_growth_spread`=0.500001, `timestep_max_growth_spread`=3.93251e-10 | Small 16/24 grid and short t=8 validation sweep; production needs larger resolution, duration, seed, width/aspect, and Lundquist sweeps. |
| periodic double-Harris | `outputs/nonlinear_campaign_evidence_20260522/double_harris_long_n48_t24` | pass | not production-ready | `shape`=[48, 48], `t_end`=24.0, `samples`=13, `fitted_early_growth_rate`=0.221796, `max_growth_factor`=7.35302, `reconnected_flux_amplification`=5.89309, `island_width_amplification`=2.42757, `relative_energy_increase`=0.0 | Positive response is validation evidence only; no production-duration convergence, seed-QI, aspect-ratio, or Lundquist sweep is attached. |
| Rutherford executor | `outputs/nonlinear_campaign_evidence_20260522/rutherford_fast_n24_t1` | pass | not production-ready | `shape`=[24, 24], `t_end`=1.0, `steps`=100, `samples`=21, `max_relative_energy_growth`=-9.98053e-05, `max_magnetic_divergence_linf`=0.0 | FAST schema/diagnostic run is far shorter than Rutherford-duration requirements and cannot be promoted to nonlinear physics. |
| forced turbulent reconnection | `outputs/nonlinear_campaign_evidence_20260522/forced_turbulent_reconnection_n24_t4` | pass | validation-ready | `shape`=[24, 24], `t_end`=4.0, `samples`=21, `reconnection_proxy_change`=1.91639, `max_abs_reconnection_rate_proxy`=3.56186, `current_linf_growth`=0.273459, `max_relative_energy_growth`=0.0, `max_magnetic_divergence_linf`=0.0, `promotion_ready`=True, `history_sample_count`=21 | 2-D reduced-MHD proxy and single deterministic seed; no turbulent ensemble, 3-D physics, inertial range, or LV99 scaling evidence. |
| periodic double-Harris | `outputs/campaigns/double_harris_convergence_gpu_n32_48_64_t16_20260519_173637` | pass | not production-ready | `case_count`=5, `t_end`=16.0, `resolution_growth_rate_spread`=0.00411786, `timestep_growth_rate_spread`=1.96263e-10, `resolution_max_growth_spread`=0.00162281, `timestep_max_growth_spread`=2.05568e-10 | Medium validation sweep is not a production-scale duration, seed, aspect-ratio, or Lundquist campaign. |
| periodic double-Harris | `outputs/campaigns/growing_double_harris_gpu_96_t120_20260518_044120` | pass | validation-ready | `shape`=[96, 96], `t_end`=120.0, `samples`=41, `fitted_early_growth_rate`=0.186895, `max_growth_factor`=7.35301, `reconnected_flux_amplification`=6.47122, `island_width_amplification`=2.54386, `max_x_point_count`=4, `max_o_point_count`=2, `relative_energy_increase`=0.0, `promotion_ready`=True, `history_sample_count`=41 | Convergence-backed validation media only; production claims still need larger seed, width/aspect, Lundquist, and duration sweeps. |
| Rutherford executor | `outputs/campaigns/rutherford_current_schema_96_dt005_20260517_161235` | pass | not production-ready | `shape`=[96, 96], `end_step`=45802, `target_step`=45802, `history_samples`=202, `max_relative_energy_growth`=-2.05077e-06, `final_magnetic_divergence_linf`=0.0, `promotion_ready`=False, `reconnected_flux_amplification`=1.0, `island_width_amplification`=1.0, `max_x_point_count`=2, `max_o_point_count`=1, `terminal_step`=45802, `history_sample_count`=202 | Duration target completed, but promotion failed because reconnecting-flux and island-width amplification remained 1.00. |
| forced turbulent reconnection | `outputs/readme_media/forced_turbulent_reconnection_64_t80_wide` | pass | not production-ready | `shape`=[64, 64], `t_end`=80.0, `samples`=41, `reconnection_proxy_change`=1.38249, `max_abs_reconnection_rate_proxy`=0.343463, `current_linf_growth`=13.829, `max_relative_energy_growth`=0.0, `max_magnetic_divergence_linf`=0.0 | Validation media only: single deterministic 2-D proxy run, no ensemble or 3-D turbulent-reconnection scaling. |

## Claim boundary

- Passing double-Harris rows support validation-level nonlinear response and convergence scaffolding, not Rutherford/plasmoid production physics.
- Passing Rutherford rows support executor/schema/duration mechanics unless the promotion report passes with positive response, convergence, seed-QI, geometry, and media gates.
- Passing forced turbulent-reconnection rows support 2-D reduced-MHD proxy-media readiness only, not 3-D turbulent-reconnection or LV99 scaling claims.
- Large binary outputs remain under `outputs/`, which is git-ignored; this page and the JSON summary are the small review artifacts.
