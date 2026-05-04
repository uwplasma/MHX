"""Minimal Python driver for the first rebuilt MHX config."""

from __future__ import annotations

from mhx.benchmarks import run_linear_tearing_smoke
from mhx.config import load_config


def main() -> None:
    cfg = load_config("examples/linear_tearing.toml")
    _, diagnostics = run_linear_tearing_smoke(cfg)
    print(diagnostics)


if __name__ == "__main__":
    main()
