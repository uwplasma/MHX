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
| Validation media refresh | `python examples/make_validation_media.py` |
| README media refresh | `python examples/make_readme_media.py` |
| Neural-ODE reproducibility figures | `python examples/make_neural_ode_reproducibility.py` |
| Latent-ODE FAST training | `python examples/train_latent_ode_fast.py --outdir outputs/examples/latent_ode_fast` |
| Rutherford production plan | `python examples/make_rutherford_production_plan.py --outdir outputs/examples/rutherford_plan` |
| Rutherford executor chunk | `python examples/run_rutherford_production_chunk.py --outdir outputs/examples/rutherford_chunk --movies` |

The package skeleton in `examples/plugin_template/` is the recommended layout
for third-party physics and diagnostics plugins.
