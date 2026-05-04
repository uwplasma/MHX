from __future__ import annotations

from pathlib import Path
import dataclasses
import numpy as np

from mhx.config import load_config_yaml
from mhx.inverse_design.train import InverseDesignConfig, run_inverse_design
from mhx.io.paths import create_run_dir


def test_inverse_design_io_schema(tmp_path: Path) -> None:
    cfg = InverseDesignConfig.fast("original")
    cfg = dataclasses.replace(cfg, n_train_steps=1)

    run_paths = create_run_dir(root=tmp_path, tag="inverse", timestamp="0000", exist_ok=True)
    run_paths, history, _, _, _ = run_inverse_design(cfg, run_paths=run_paths)

    assert run_paths.config_yaml.exists()
    assert run_paths.history_npz.exists()
    assert run_paths.solution_initial_npz.exists()
    assert run_paths.solution_mid_npz.exists()
    assert run_paths.solution_final_npz.exists()

    cfg_payload = load_config_yaml(run_paths.config_yaml)
    assert "sim" in cfg_payload
    assert "objective" in cfg_payload
    assert "training" in cfg_payload

    h = np.load(run_paths.history_npz, allow_pickle=True)
    for key in ["loss", "f_kin", "complexity", "target_f_kin", "target_complexity", "lambda_complexity"]:
        assert key in h.files
    assert "schema_version" in h.files
    assert "mhx_version" in h.files
