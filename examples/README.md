# Examples

Each example is deterministic and writes outputs outside the repository source
tree by default.

| Example | Command |
| --- | --- |
| Linear tearing TOML smoke run | `mhx run examples/linear_tearing.toml --outdir outputs/examples/linear_tearing` |
| Hyper-resistivity toy term | `mhx run examples/linear_tearing_hyper.toml --outdir outputs/examples/linear_tearing_hyper` |
| Two-fluid Ohm's-law toy term | `mhx run examples/linear_tearing_twofluid_toy.toml --outdir outputs/examples/twofluid_toy` |
| Local plugin demo | `mhx run examples/linear_tearing_plugin_demo.toml --outdir outputs/examples/plugin_demo` |
| Python wrapper for linear tearing | `python examples/run_linear_tearing.py --outdir outputs/examples/linear_tearing_script` |
| Orszag--Tang nonlinear media | `python examples/run_orszag_tang.py --outdir outputs/examples/orszag_tang --nx 64 --ny 64 --t-end 6` |
| Clean Kelvin--Helmholtz incompressible notebook | `jupyter notebook examples/run_kelvin_helmholtz_incompressible.ipynb` |
| Smooth Kelvin--Helmholtz compressible-MHD notebook | `jupyter notebook examples/run_kelvin_helmholtz_compressible_mhd.ipynb` |
| Kelvin--Helmholtz backpropagation notebook | `jupyter notebook examples/run_kelvin_helmholtz_backpropagation.ipynb` |
| Validation media refresh | `python examples/make_validation_media.py` |
| README media refresh | `python examples/make_readme_media.py` |
| Neural-ODE reproducibility figures | `python examples/make_neural_ode_reproducibility.py` |
| Latent-ODE FAST training | `python examples/train_latent_ode_fast.py --outdir outputs/examples/latent_ode_fast` |
| Rutherford production plan | `python examples/make_rutherford_production_plan.py --outdir outputs/examples/rutherford_plan` |
| Rutherford executor chunk | `python examples/run_rutherford_production_chunk.py --outdir outputs/examples/rutherford_chunk --movies` |
| Paper artifact verifier | `python examples/tools/verify_paper_artifacts.py --artifact-root docs/_static/validation` |

The package skeleton in `examples/plugin_template/` is the recommended layout
for third-party physics and diagnostics plugins.

## Publication examples

The `publication_*.py` scripts are standalone, FAST, publication-style
workflows. Edit the input parameters at the top of each file for larger runs.
By default they write under `outputs/examples/publication`; set
`MHX_EXAMPLE_OUTDIR_ROOT=/path/to/root` to redirect all outputs during tests.

| Problem | Command | Expected primary outputs |
| --- | --- | --- |
| Linear Harris tearing | `python examples/publication_linear_harris_tearing.py` | `$MHX_EXAMPLE_OUTDIR_ROOT/linear_harris_tearing/manifest.json`, `linear_tearing_timedomain.npz`, `figures/linear_tearing_timedomain.png`, `figures/publication_linear_harris_tearing_summary.png` |
| Double-Harris nonlinear reconnection | `python examples/publication_double_harris_reconnection.py` | `$MHX_EXAMPLE_OUTDIR_ROOT/double_harris_reconnection/manifest.json`, `periodic_double_harris_seeded_long_run.npz`, `figures/periodic_double_harris_flux.gif`, `figures/periodic_double_harris_current.gif`, `figures/publication_double_harris_delta_flux.gif`, `figures/publication_double_harris_reconnection_summary.png` |
| Orszag--Tang plus turbulence | `python examples/publication_orszag_tang_turbulence.py` | `$MHX_EXAMPLE_OUTDIR_ROOT/orszag_tang_turbulence/orszag_tang/manifest.json`, `$MHX_EXAMPLE_OUTDIR_ROOT/orszag_tang_turbulence/forced_turbulent_reconnection/manifest.json`, `figures/publication_orszag_tang_turbulence_summary.png`, generated current/flux GIFs |
| Rutherford production path | `python examples/publication_rutherford_production.py` | `$MHX_EXAMPLE_OUTDIR_ROOT/rutherford_production_path/campaign_plan.json`, `resume_plan.json`, `production_history.npz`, `checkpoints/state_step_000000000006.npz`, `figures/fixed_scale_flux_movie.gif`, `figures/publication_rutherford_production_summary.png` |
| Neural ODE | `python examples/publication_neural_ode.py` | `$MHX_EXAMPLE_OUTDIR_ROOT/neural_ode/manifest.json`, `dataset.npz`, `latent_ode_predictions.npz`, `latent_ode_metrics.json`, `figures/latent_ode_predictions.png`, `figures/publication_neural_ode_summary.png` |

For a clean rerun, remove the corresponding output directory first. The
Rutherford script intentionally demonstrates the restart/resume path, so an
existing checkpoint directory may cause it to resume instead of starting fresh.
