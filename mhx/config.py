from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Optional

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    yaml = None


EquilibriumMode = Literal["original", "forcefree"]


@dataclass(frozen=True)
class Objective:
    """Objective for inverse design / comparisons.

    Loss definition:
      (f_kin - target_f_kin)^2 + lambda_complexity * (complexity - target_complexity)^2
    """

    target_f_kin: float
    target_complexity: float
    lambda_complexity: float

    def as_dict(self) -> Dict[str, float]:
        return dataclasses.asdict(self)


def objective_preset(eq_mode: EquilibriumMode) -> Objective:
    """Default objective presets by equilibrium branch.

    These are intentionally conservative and are meant to be *shared* across:
      - inverse design training
      - figure generation (comparisons)

    If a run saves an objective, that saved objective must take precedence.
    """
    # Single default until we have a documented reason to split presets.
    _ = eq_mode
    return Objective(target_f_kin=0.03, target_complexity=1e-5, lambda_complexity=1.0)


@dataclass(frozen=True)
class TearingSimConfig:
    # Grid and box
    Nx: int = 64
    Ny: int = 64
    Nz: int = 1
    Lx: float = 2.0 * math.pi
    Ly: float = 2.0 * math.pi
    Lz: float = 2.0 * math.pi

    # Physical parameters
    nu: float = 1e-3
    eta: float = 1e-3
    B0: float = 1.0
    a: float = 0.25
    B_g: float = 0.2
    eps_B: float = 1e-3

    # Time integration
    t0: float = 0.0
    t1: float = 60.0
    n_frames: int = 150
    dt0: float = 5e-4
    progress: bool = True
    jit: bool = False
    check_finite: bool = True

    equilibrium_mode: EquilibriumMode = "original"

    @classmethod
    def fast(cls, equilibrium_mode: EquilibriumMode = "original") -> "TearingSimConfig":
        return cls(
            Nx=16,
            Ny=16,
            Nz=1,
            t1=0.5,
            n_frames=6,
            dt0=5e-4,
            progress=False,
            jit=False,
            check_finite=True,
            equilibrium_mode=equilibrium_mode,
        )

    def as_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)




@dataclass(frozen=True)
class ModelConfig:
    equilibrium_mode: str | None = None
    rhs_terms: list[str] | None = None
    term_params: dict[str, dict[str, float]] | None = None
    diagnostics: list[str] | None = None

    def __post_init__(self):
        object.__setattr__(self, "rhs_terms", self.rhs_terms or [])
        object.__setattr__(self, "term_params", self.term_params or {})
        object.__setattr__(self, "diagnostics", self.diagnostics or [])

    def as_dict(self) -> Dict[str, Any]:
        return {
            "equilibrium_mode": self.equilibrium_mode,
            "rhs_terms": self.rhs_terms,
            "term_params": self.term_params,
            "diagnostics": self.diagnostics,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelConfig":
        return cls(
            equilibrium_mode=data.get("equilibrium_mode"),
            rhs_terms=list(data.get("rhs_terms", [])),
            term_params=dict(data.get("term_params", {})),
            diagnostics=list(data.get("diagnostics", [])),
        )


@dataclass(frozen=True)
class InverseDesignConfig:
    sim: TearingSimConfig
    objective: Objective
    model: ModelConfig = dataclasses.field(default_factory=ModelConfig)

    # Bounds for eta and nu (log10-space)
    log10_eta_min: float = -4.5
    log10_eta_max: float = -2.0
    log10_nu_min: float = -4.5
    log10_nu_max: float = -2.0

    # NN + training hyperparameters
    latent_dim: int = 1
    hidden_width: int = 32
    hidden_depth: int = 2
    learning_rate: float = 1e-3
    n_train_steps: int = 25
    print_every: int = 1
    z_train: float = 0.0
    seed: int = 1234

    @classmethod
    def default(cls, eq_mode: EquilibriumMode = "forcefree") -> "InverseDesignConfig":
        sim = TearingSimConfig(equilibrium_mode=eq_mode)
        obj = objective_preset(eq_mode)
        return cls(sim=sim, objective=obj)

    @classmethod
    def fast(cls, eq_mode: EquilibriumMode = "forcefree") -> "InverseDesignConfig":
        sim = TearingSimConfig.fast(equilibrium_mode=eq_mode)
        obj = objective_preset(eq_mode)
        return cls(sim=sim, objective=obj, n_train_steps=2, hidden_width=8, hidden_depth=1)

    def as_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def override_objective(
    objective: Objective,
    target_f_kin: Optional[float] = None,
    target_complexity: Optional[float] = None,
    lambda_complexity: Optional[float] = None,
) -> Objective:
    return Objective(
        target_f_kin=objective.target_f_kin if target_f_kin is None else float(target_f_kin),
        target_complexity=objective.target_complexity if target_complexity is None else float(target_complexity),
        lambda_complexity=objective.lambda_complexity if lambda_complexity is None else float(lambda_complexity),
    )


def dump_config_yaml(path, payload: Dict[str, Any]) -> None:
    """Write config payload to YAML; fall back to JSON if PyYAML unavailable."""
    if yaml is None:
        import json

        Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=True)


def load_config_yaml(path) -> Dict[str, Any]:
    if yaml is None:
        import json

        return json.loads(Path(path).read_text(encoding="utf-8"))
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_model_config(path) -> ModelConfig:
    data = load_config_yaml(path)
    if "model" in data:
        data = data["model"]
    return ModelConfig.from_dict(data)
