"""Configuration schema for rebuilt MHX runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

try:  # pragma: no cover - exercised only on Python 3.10.
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


def _tuple_from(value: Any, *, length: int, name: str, cast: type) -> tuple[Any, ...]:
    if value is None:
        raise ValueError(f"{name} must not be None")
    values = tuple(cast(item) for item in value)
    if len(values) != length:
        raise ValueError(f"{name} must have length {length}, got {len(values)}")
    return values


@dataclass(frozen=True)
class MeshConfig:
    """Uniform Cartesian mesh configuration."""

    shape: tuple[int, int] = (32, 32)
    lower: tuple[float, float] = (0.0, 0.0)
    upper: tuple[float, float] = (6.283185307179586, 6.283185307179586)
    periodic: tuple[bool, bool] = (True, True)

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any] | None) -> MeshConfig:
        mapping = mapping or {}
        return cls(
            shape=_tuple_from(
                mapping.get("shape", cls.shape),
                length=2,
                name="mesh.shape",
                cast=int,
            ),
            lower=_tuple_from(
                mapping.get("lower", cls.lower),
                length=2,
                name="mesh.lower",
                cast=float,
            ),
            upper=_tuple_from(
                mapping.get("upper", cls.upper),
                length=2,
                name="mesh.upper",
                cast=float,
            ),
            periodic=_tuple_from(
                mapping.get("periodic", cls.periodic),
                length=2,
                name="mesh.periodic",
                cast=bool,
            ),
        ).validated()

    def validated(self) -> MeshConfig:
        if any(points < 4 for points in self.shape):
            raise ValueError("mesh.shape entries must be >= 4")
        if any(hi <= lo for lo, hi in zip(self.lower, self.upper, strict=True)):
            raise ValueError("mesh.upper entries must exceed mesh.lower")
        return self


@dataclass(frozen=True)
class TimeConfig:
    """Time integration controls."""

    t0: float = 0.0
    t1: float = 1.0
    dt: float = 0.01
    save_every: int = 10

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any] | None) -> TimeConfig:
        mapping = mapping or {}
        return cls(
            t0=float(mapping.get("t0", cls.t0)),
            t1=float(mapping.get("t1", cls.t1)),
            dt=float(mapping.get("dt", cls.dt)),
            save_every=int(mapping.get("save_every", cls.save_every)),
        ).validated()

    def validated(self) -> TimeConfig:
        if self.t1 <= self.t0:
            raise ValueError("time.t1 must exceed time.t0")
        if self.dt <= 0.0:
            raise ValueError("time.dt must be positive")
        if self.save_every < 1:
            raise ValueError("time.save_every must be >= 1")
        return self


@dataclass(frozen=True)
class PhysicsConfig:
    """Physics model parameters for early rebuild workflows."""

    model: str = "reduced_mhd_linear_tearing"
    resistivity: float = 1.0e-3
    viscosity: float = 1.0e-3

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any] | None) -> PhysicsConfig:
        mapping = mapping or {}
        return cls(
            model=str(mapping.get("model", cls.model)),
            resistivity=float(mapping.get("resistivity", cls.resistivity)),
            viscosity=float(mapping.get("viscosity", cls.viscosity)),
        ).validated()

    def validated(self) -> PhysicsConfig:
        if self.resistivity < 0.0:
            raise ValueError("physics.resistivity must be non-negative")
        if self.viscosity < 0.0:
            raise ValueError("physics.viscosity must be non-negative")
        return self


@dataclass(frozen=True)
class NumericsConfig:
    """Numerical method switches."""

    method: str = "spectral"
    enable_x64: bool = True
    enable_jit: bool = True

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any] | None) -> NumericsConfig:
        mapping = mapping or {}
        return cls(
            method=str(mapping.get("method", cls.method)),
            enable_x64=bool(mapping.get("enable_x64", cls.enable_x64)),
            enable_jit=bool(mapping.get("enable_jit", cls.enable_jit)),
        )


@dataclass(frozen=True)
class DiagnosticsConfig:
    """Diagnostics requested for a run."""

    quantities: tuple[str, ...] = ("energy", "divergence_error")

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any] | None) -> DiagnosticsConfig:
        mapping = mapping or {}
        quantities = mapping.get("quantities", cls.quantities)
        return cls(quantities=tuple(str(item) for item in quantities))


@dataclass(frozen=True)
class RunConfig:
    """Top-level MHX run configuration."""

    name: str = "linear_tearing_smoke"
    output_dir: Path = Path("outputs/linear_tearing_smoke")
    mesh: MeshConfig = field(default_factory=MeshConfig)
    time: TimeConfig = field(default_factory=TimeConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    numerics: NumericsConfig = field(default_factory=NumericsConfig)
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> RunConfig:
        return cls(
            name=str(mapping.get("name", cls.name)),
            output_dir=Path(mapping.get("output_dir", cls.output_dir)),
            mesh=MeshConfig.from_mapping(mapping.get("mesh")),
            time=TimeConfig.from_mapping(mapping.get("time")),
            physics=PhysicsConfig.from_mapping(mapping.get("physics")),
            numerics=NumericsConfig.from_mapping(mapping.get("numerics")),
            diagnostics=DiagnosticsConfig.from_mapping(mapping.get("diagnostics")),
        )

    def with_output_dir(self, output_dir: str | Path) -> RunConfig:
        """Return a copy with a different output directory."""
        return replace(self, output_dir=Path(output_dir))

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible builtins."""
        data = asdict(self)
        data["output_dir"] = str(self.output_dir)
        for section in ("mesh", "time", "physics", "numerics", "diagnostics"):
            for key, value in data[section].items():
                if isinstance(value, tuple):
                    data[section][key] = list(value)
        return data

    def to_toml(self) -> str:
        """Serialize a stable starter TOML representation."""
        data = self.to_dict()
        return (
            f'name = "{data["name"]}"\n'
            f'output_dir = "{data["output_dir"]}"\n\n'
            "[mesh]\n"
            f"shape = {data['mesh']['shape']}\n"
            f"lower = {data['mesh']['lower']}\n"
            f"upper = {data['mesh']['upper']}\n"
            f"periodic = {str(data['mesh']['periodic']).lower()}\n\n"
            "[time]\n"
            f"t0 = {data['time']['t0']}\n"
            f"t1 = {data['time']['t1']}\n"
            f"dt = {data['time']['dt']}\n"
            f"save_every = {data['time']['save_every']}\n\n"
            "[physics]\n"
            f'model = "{data["physics"]["model"]}"\n'
            f"resistivity = {data['physics']['resistivity']}\n"
            f"viscosity = {data['physics']['viscosity']}\n\n"
            "[numerics]\n"
            f"method = \"{data['numerics']['method']}\"\n"
            f"enable_x64 = {str(data['numerics']['enable_x64']).lower()}\n"
            f"enable_jit = {str(data['numerics']['enable_jit']).lower()}\n\n"
            "[diagnostics]\n"
            f"quantities = {data['diagnostics']['quantities']}\n"
        )


def load_config(path: str | Path) -> RunConfig:
    """Load a TOML file into a :class:`RunConfig`."""
    config_path = Path(path)
    with config_path.open("rb") as handle:
        data = tomllib.load(handle)
    return RunConfig.from_mapping(data)
