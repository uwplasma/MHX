from __future__ import annotations

from mhx.config import ModelConfig
from mhx.solver.plugins import build_terms


def test_model_config_build_terms():
    cfg = ModelConfig(
        equilibrium_mode="original",
        rhs_terms=["linear_drag", "hyper_resistivity"],
        term_params={"linear_drag": {"mu": 0.05}, "hyper_resistivity": {"eta4": 1e-3}},
    )
    terms = build_terms(cfg.rhs_terms, cfg.term_params)
    assert len(terms) == 2
