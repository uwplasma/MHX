"""Create a deterministic FAST neural-ODE reproducibility bundle.

This example intentionally avoids expensive neural training.  It writes the
dataset/split/baseline/calibration artifacts that a later neural-ODE experiment
must use and beat.
"""

from __future__ import annotations

import json
from pathlib import Path

from mhx.neural_ode import write_neural_ode_reproducibility_bundle


def main() -> None:
    output_dir = Path("outputs/examples/neural_ode_seed_qi_fast")
    manifest_path, validation = write_neural_ode_reproducibility_bundle(
        output_dir,
        seeds=(0, 1, 2, 3, 4, 5),
        shape=(16, 16),
        steps=24,
        dt=1.0e-2,
        observation_count=2,
    )
    print(f"wrote {manifest_path}")
    print(f"validation passed: {validation['passed']}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name, relative_path in sorted(manifest["outputs"].items()):
        print(f"{name}: {output_dir / relative_path}")


if __name__ == "__main__":
    main()
