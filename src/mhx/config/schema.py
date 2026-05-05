"""Configuration schema for rebuilt MHX runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from mhx.versioning import require_supported_api_version

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
    t1: float = 0.1
    dt: float = 0.01
    save_every: int = 1

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
    equilibrium: str = "cosine_tearing"
    equilibrium_parameters: dict[str, float] = field(default_factory=dict)
    resistivity: float = 1.0e-3
    viscosity: float = 1.0e-3
    plugin_modules: tuple[str, ...] = ()
    plugin_entry_point_groups: tuple[str, ...] = ()
    rhs_terms: tuple[str, ...] = ()
    term_parameters: dict[str, dict[str, float]] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any] | None) -> PhysicsConfig:
        mapping = mapping or {}
        return cls(
            model=str(mapping.get("model", cls.model)),
            equilibrium=str(mapping.get("equilibrium", cls.equilibrium)),
            equilibrium_parameters={
                str(key): float(value)
                for key, value in mapping.get("equilibrium_parameters", {}).items()
            },
            resistivity=float(mapping.get("resistivity", cls.resistivity)),
            viscosity=float(mapping.get("viscosity", cls.viscosity)),
            plugin_modules=tuple(
                str(item) for item in mapping.get("plugin_modules", cls.plugin_modules)
            ),
            plugin_entry_point_groups=tuple(
                str(item)
                for item in mapping.get(
                    "plugin_entry_point_groups",
                    cls.plugin_entry_point_groups,
                )
            ),
            rhs_terms=tuple(str(item) for item in mapping.get("rhs_terms", cls.rhs_terms)),
            term_parameters={
                str(name): {str(key): float(value) for key, value in parameters.items()}
                for name, parameters in mapping.get("term_parameters", {}).items()
            },
        ).validated()

    def validated(self) -> PhysicsConfig:
        if not self.equilibrium:
            raise ValueError("physics.equilibrium must be non-empty")
        if self.resistivity < 0.0:
            raise ValueError("physics.resistivity must be non-negative")
        if self.viscosity < 0.0:
            raise ValueError("physics.viscosity must be non-negative")
        if len(set(self.plugin_modules)) != len(self.plugin_modules):
            raise ValueError("physics.plugin_modules entries must be unique")
        if len(set(self.plugin_entry_point_groups)) != len(self.plugin_entry_point_groups):
            raise ValueError("physics.plugin_entry_point_groups entries must be unique")
        unknown_parameters = sorted(set(self.term_parameters) - set(self.rhs_terms))
        if unknown_parameters:
            raise ValueError(
                "physics.term_parameters contains entries not listed in physics.rhs_terms: "
                + ", ".join(unknown_parameters)
            )
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

    quantities: tuple[str, ...] = ("energy", "mode_growth", "divergence_error")
    plugin_modules: tuple[str, ...] = ()
    plugin_entry_point_groups: tuple[str, ...] = ()
    mode: tuple[int, int] = (1, 1)
    fit_time_window: tuple[float, float] | None = None

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any] | None) -> DiagnosticsConfig:
        mapping = mapping or {}
        quantities = mapping.get("quantities", cls.quantities)
        fit_time_window = mapping.get("fit_time_window", cls.fit_time_window)
        return cls(
            quantities=tuple(str(item) for item in quantities),
            plugin_modules=tuple(
                str(item) for item in mapping.get("plugin_modules", cls.plugin_modules)
            ),
            plugin_entry_point_groups=tuple(
                str(item)
                for item in mapping.get(
                    "plugin_entry_point_groups",
                    cls.plugin_entry_point_groups,
                )
            ),
            mode=_tuple_from(
                mapping.get("mode", cls.mode),
                length=2,
                name="diagnostics.mode",
                cast=int,
            ),
            fit_time_window=(
                None
                if fit_time_window is None
                else _tuple_from(
                    fit_time_window,
                    length=2,
                    name="diagnostics.fit_time_window",
                    cast=float,
                )
            ),
        ).validated()

    def validated(self) -> DiagnosticsConfig:
        if not self.quantities:
            raise ValueError("diagnostics.quantities must not be empty")
        if len(set(self.quantities)) != len(self.quantities):
            raise ValueError("diagnostics.quantities entries must be unique")
        if len(set(self.plugin_modules)) != len(self.plugin_modules):
            raise ValueError("diagnostics.plugin_modules entries must be unique")
        if len(set(self.plugin_entry_point_groups)) != len(self.plugin_entry_point_groups):
            raise ValueError("diagnostics.plugin_entry_point_groups entries must be unique")
        if self.fit_time_window is not None and self.fit_time_window[1] <= self.fit_time_window[0]:
            raise ValueError("diagnostics.fit_time_window upper bound must exceed lower bound")
        return self


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
        require_supported_api_version(context="RunConfig serializer")
        data = asdict(self)
        data["output_dir"] = str(self.output_dir)
        data["api_version"] = require_supported_api_version(context="RunConfig serializer")
        for section in ("mesh", "time", "physics", "numerics", "diagnostics"):
            for key, value in data[section].items():
                if isinstance(value, tuple):
                    data[section][key] = list(value)
        return data

    def to_toml(self) -> str:
        """Serialize a stable starter TOML representation."""
        data = self.to_dict()
        fit_time_window_line = (
            ""
            if data["diagnostics"]["fit_time_window"] is None
            else f"fit_time_window = {data['diagnostics']['fit_time_window']}\n"
        )
        equilibrium_parameter_lines = ""
        if data["physics"]["equilibrium_parameters"]:
            equilibrium_parameter_lines += "\n[physics.equilibrium_parameters]\n"
            for key, value in sorted(data["physics"]["equilibrium_parameters"].items()):
                equilibrium_parameter_lines += f"{key} = {value}\n"
        term_parameter_lines = ""
        for term_name, parameters in sorted(data["physics"]["term_parameters"].items()):
            term_parameter_lines += f"\n[physics.term_parameters.{term_name}]\n"
            for key, value in sorted(parameters.items()):
                term_parameter_lines += f"{key} = {value}\n"
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
            f'equilibrium = "{data["physics"]["equilibrium"]}"\n'
            f"resistivity = {data['physics']['resistivity']}\n"
            f"viscosity = {data['physics']['viscosity']}\n"
            f"plugin_modules = {data['physics']['plugin_modules']}\n"
            "plugin_entry_point_groups = "
            f"{data['physics']['plugin_entry_point_groups']}\n"
            f"rhs_terms = {data['physics']['rhs_terms']}\n\n"
            f"{equilibrium_parameter_lines}"
            f"{term_parameter_lines}"
            "[numerics]\n"
            f"method = \"{data['numerics']['method']}\"\n"
            f"enable_x64 = {str(data['numerics']['enable_x64']).lower()}\n"
            f"enable_jit = {str(data['numerics']['enable_jit']).lower()}\n\n"
            "[diagnostics]\n"
            f"quantities = {data['diagnostics']['quantities']}\n"
            f"plugin_modules = {data['diagnostics']['plugin_modules']}\n"
            "plugin_entry_point_groups = "
            f"{data['diagnostics']['plugin_entry_point_groups']}\n"
            f"mode = {data['diagnostics']['mode']}\n"
            f"{fit_time_window_line}"
        )


def load_config(path: str | Path) -> RunConfig:
    """Load a TOML file into a :class:`RunConfig`."""
    config_path = Path(path)
    with config_path.open("rb") as handle:
        data = tomllib.load(handle)
    return RunConfig.from_mapping(data)
