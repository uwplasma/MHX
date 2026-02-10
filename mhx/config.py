from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional


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
            equilibrium_mode=equilibrium_mode,
        )

    def as_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class InverseDesignConfig:
    sim: TearingSimConfig
    objective: Objective

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

