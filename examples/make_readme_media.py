"""Generate compact README engagement movies from validated/literature workflows."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np

from mhx.benchmarks import run_linear_tearing_layer_validation
from mhx.diagnostics import FluxCriticalPoint, detect_flux_critical_points

README_GIF_DURATION_MS = 90
DOUBLE_HARRIS_MAX_FRAMES = 18
DOUBLE_HARRIS_LENGTHS = (2.0 * np.pi, 2.0 * np.pi)
DOUBLE_HARRIS_SHEET_HALF_WIDTH = 1.2
ORSZAG_TANG_MAX_FRAMES = 36
TURBULENCE_MAX_FRAMES = 20


def main() -> None:
    """Write small GIFs used by the README and docs landing pages."""
    output_dir = Path("docs/_static/readme")
    output_dir.mkdir(parents=True, exist_ok=True)
    media_entries, double_harris_qa = _write_double_harris_readme_movies(output_dir)
    orszag_entries, orszag_qa = _write_orszag_tang_readme_movies(output_dir)
    turbulence_entries, turbulence_qa = _write_turbulence_readme_movies(output_dir)
    media_entries.extend(orszag_entries)
    media_entries.extend(turbulence_entries)
    _write_harris_layer_sweep(output_dir / "harris_layer_sweep.gif")
    media_entries.append(
        _gif_manifest_entry(
            output_dir / "harris_layer_sweep.gif",
            source="run_linear_tearing_layer_validation(grid_points=128)",
            t_end=None,
            time_span=None,
            notes="Validated linear Harris tearing layer/eigenfunction sweep over S.",
        )
    )
    _write_plasmoid_scaling_schematic(output_dir / "plasmoid_scaling_schematic.gif")
    media_entries.append(
        _gif_manifest_entry(
            output_dir / "plasmoid_scaling_schematic.gif",
            source="Loureiro-Schekochihin-Cowley analytic scaling schematic",
            t_end=None,
            time_span=None,
            notes="Theory schematic; not solver output.",
        )
    )
    _write_mhd_turbulence_schematic(output_dir / "mhd_turbulence_cascade.gif")
    media_entries.append(
        _gif_manifest_entry(
            output_dir / "mhd_turbulence_cascade.gif",
            source="Deterministic synthetic 2-D MHD-like Fourier cascade schematic",
            t_end=None,
            time_span=None,
            notes="Engagement schematic; not solver output.",
        )
    )
    _write_visual_qa_manifest(
        output_dir,
        media_entries,
        {
            "double_harris": double_harris_qa,
            "orszag_tang": orszag_qa,
            "turbulence": turbulence_qa,
        },
    )


def _write_double_harris_readme_movies(
    output_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Write README solver movies from the longest available nonlinear replay."""
    source = _select_double_harris_history(output_dir)
    history = _load_double_harris_history(source["history_path"])
    frame_indices = _sample_frame_indices(len(history["time"]), DOUBLE_HARRIS_MAX_FRAMES)
    source_label = (
        f"{source['source_kind']} double-Harris validation "
        f"{history['shape'][0]}×{history['shape'][1]}, t≤{history['t_end']:.0f}"
    )
    flux_path = output_dir / "double_harris_reconnection.gif"
    current_path = output_dir / "double_harris_current_sheet.gif"
    _write_harris_sheet_contour_movie(
        history,
        frame_indices,
        path=flux_path,
        source_label=source_label,
    )
    _write_harris_full_domain_contour_movie(
        history,
        frame_indices,
        path=current_path,
        source_label=source_label,
    )
    snapshots = _write_double_harris_snapshot_contact_sheets(output_dir, history, source_label)
    visual_qa = _double_harris_visual_qa(history, source, snapshots)
    common_source = {
        "source": str(source["history_path"]),
        "source_kind": source["source_kind"],
        "source_samples": int(len(history["time"])),
        "source_shape": list(history["shape"]),
        "t_end": history["t_end"],
        "time_span": [float(history["time"][0]), float(history["time"][-1])],
        "validation_passed": source.get("validation_passed"),
    }
    entries = [
        _gif_manifest_entry(
            flux_path,
            source=common_source,
            t_end=history["t_end"],
            time_span=common_source["time_span"],
            notes=(
                "Solver-generated Harris-sheet reconnection movie: out-of-plane "
                "current density with magnetic-flux/Az contours and detected X/O markers."
            ),
        ),
        _gif_manifest_entry(
            current_path,
            source=common_source,
            t_end=history["t_end"],
            time_span=common_source["time_span"],
            notes=(
                "Solver-generated full-domain periodic double-Harris movie: total "
                "current density with magnetic-flux/Az contours across both sheets."
            ),
        ),
    ]
    return entries, visual_qa


def _select_double_harris_history(output_dir: Path) -> dict[str, Any]:
    """Prefer precomputed longer nonlinear histories; generate a labeled fallback if absent."""
    candidates = sorted(
        {
            *Path("outputs/readme_media").glob("**/periodic_double_harris_seeded_long_run.npz"),
            *Path("outputs/long_runs").glob("**/periodic_double_harris_seeded_long_run.npz"),
            *Path("outputs/docs_validation").glob("**/periodic_double_harris_seeded_long_run.npz"),
            *Path("outputs/ci").glob("**/periodic_double_harris_seeded_long_run.npz"),
            *(output_dir / "generated_double_harris_validation_t60").glob(
                "periodic_double_harris_seeded_long_run.npz"
            ),
        }
    )
    ranked: list[tuple[tuple[float, int, int], dict[str, Any]]] = []
    for path in candidates:
        try:
            history = _load_double_harris_history(path, fields_only=True)
        except (KeyError, ValueError, OSError):
            continue
        validation_path = path.parent / "validation.json"
        validation_passed = None
        if validation_path.exists():
            validation_passed = bool(json.loads(validation_path.read_text()).get("passed"))
        source_kind = "precomputed long-run artifact"
        if output_dir in path.parents:
            source_kind = "generated README fallback artifact"
        rank = (
            float(history["t_end"]),
            int(history["shape"][0] * history["shape"][1]),
            int(validation_passed is not False),
        )
        ranked.append(
            (
                rank,
                {
                    "history_path": path,
                    "source_kind": source_kind,
                    "validation_passed": validation_passed,
                },
            )
        )
    if ranked:
        return max(ranked, key=lambda item: item[0])[1]
    return _generate_double_harris_fallback(output_dir)


def _generate_double_harris_fallback(output_dir: Path) -> dict[str, Any]:
    """Create a clearly labeled README-local longer validation replay when needed."""
    from mhx.benchmarks import write_periodic_double_harris_seeded_long_run_validation
    from mhx.runtime import configure_jax

    configure_jax(enable_x64=True)
    fallback_dir = output_dir / "generated_double_harris_validation_t60"
    history_path = fallback_dir / "periodic_double_harris_seeded_long_run.npz"
    if not history_path.exists():
        write_periodic_double_harris_seeded_long_run_validation(
            fallback_dir,
            shape=(64, 64),
            width=0.32,
            resistivity=1.5e-3,
            viscosity=1.5e-3,
            perturbation_amplitude=2.0e-2,
            perturbation_mode=(0, 1),
            dt=2.0e-2,
            t_end=60.0,
            save_every=50,
            fit_window=(0.0, 12.0),
            min_early_growth_rate=1.0e-9,
            min_early_growth_factor=1.000000001,
            min_max_growth_factor=1.000000001,
            movies=False,
        )
    validation_path = fallback_dir / "validation.json"
    validation_passed = None
    if validation_path.exists():
        validation_passed = bool(json.loads(validation_path.read_text()).get("passed"))
    return {
        "history_path": history_path,
        "source_kind": "generated README fallback artifact",
        "validation_passed": validation_passed,
    }


def _write_orszag_tang_readme_movies(
    output_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Write README movies from the longest available Orszag--Tang nonlinear replay."""
    source = _select_orszag_tang_history(output_dir)
    history = _load_orszag_tang_history(source["history_path"])
    frame_indices = _sample_frame_indices(len(history["time"]), ORSZAG_TANG_MAX_FRAMES)
    source_label = (
        f"{source['source_kind']} Orszag-Tang vortex "
        f"{history['shape'][0]}×{history['shape'][1]}, t≤{history['t_end']:.1f}"
    )
    current_path = output_dir / "orszag_tang_current.gif"
    vorticity_path = output_dir / "orszag_tang_vorticity.gif"
    flux_path = output_dir / "orszag_tang_flux.gif"
    _write_field_movie(
        history["current_density"][frame_indices],
        history["time"][frame_indices],
        path=current_path,
        cmap="RdBu_r",
        title_prefix="Orszag-Tang current density",
        source_label=source_label,
        symmetric=True,
    )
    _write_field_movie(
        history["omega"][frame_indices],
        history["time"][frame_indices],
        path=vorticity_path,
        cmap="RdBu_r",
        title_prefix="Orszag-Tang vorticity",
        source_label=source_label,
        symmetric=True,
    )
    _write_field_movie(
        history["psi"][frame_indices],
        history["time"][frame_indices],
        path=flux_path,
        cmap="viridis",
        title_prefix="Orszag-Tang flux",
        source_label=source_label,
        symmetric=False,
    )
    snapshots = _write_orszag_tang_snapshot_contact_sheets(output_dir, history, source_label)
    qa = _orszag_tang_visual_qa(history, source, snapshots)
    common_source = {
        "source": str(source["history_path"]),
        "source_kind": source["source_kind"],
        "source_samples": int(len(history["time"])),
        "source_shape": list(history["shape"]),
        "t_end": history["t_end"],
        "time_span": [float(history["time"][0]), float(history["time"][-1])],
        "validation_passed": source.get("validation_passed"),
    }
    return (
        [
            _gif_manifest_entry(
                current_path,
                source=common_source,
                t_end=history["t_end"],
                time_span=common_source["time_span"],
                notes=(
                    "Solver-generated reduced-MHD Orszag-Tang current-density movie "
                    "showing nonlinear filament formation."
                ),
            ),
            _gif_manifest_entry(
                vorticity_path,
                source=common_source,
                t_end=history["t_end"],
                time_span=common_source["time_span"],
                notes=(
                    "Solver-generated reduced-MHD Orszag-Tang vorticity movie "
                    "showing nonlinear vortex roll-up."
                ),
            ),
            _gif_manifest_entry(
                flux_path,
                source=common_source,
                t_end=history["t_end"],
                time_span=common_source["time_span"],
                notes="Solver-generated reduced-MHD Orszag-Tang flux-function movie.",
            ),
        ],
        qa,
    )


def _select_orszag_tang_history(output_dir: Path) -> dict[str, Any]:
    candidates = sorted(
        {
            *Path("outputs/readme_media").glob("**/orszag_tang_vortex.npz"),
            *Path("outputs/long_runs").glob("**/orszag_tang_vortex.npz"),
            *Path("outputs/docs_validation").glob("**/orszag_tang_vortex.npz"),
            *(output_dir / "generated_orszag_tang_validation_t4").glob(
                "orszag_tang_vortex.npz"
            ),
        }
    )
    ranked: list[tuple[tuple[int, float, int, float], dict[str, Any]]] = []
    for path in candidates:
        try:
            history = _load_orszag_tang_history(path, fields_only=True)
        except (KeyError, ValueError, OSError):
            continue
        validation_path = path.parent / "validation.json"
        validation_passed = None
        if validation_path.exists():
            validation_passed = bool(json.loads(validation_path.read_text()).get("passed"))
        source_kind = "precomputed nonlinear validation artifact"
        if output_dir in path.parents:
            source_kind = "generated README fallback artifact"
        diagnostics_path = path.parent / "diagnostics.json"
        activity_score = 0.0
        if diagnostics_path.exists():
            diagnostics = json.loads(diagnostics_path.read_text())
            activity_score = max(
                float(diagnostics.get("reconnection_proxy_change", 0.0) or 0.0),
                float(diagnostics.get("current_linf_growth", 0.0) or 0.0),
            )
        validation_rank = (
            2 if validation_passed is True else 1 if validation_passed is None else 0
        )
        rank = (
            validation_rank,
            float(history["t_end"]),
            int(history["shape"][0] * history["shape"][1]),
            activity_score,
        )
        ranked.append(
            (
                rank,
                {
                    "history_path": path,
                    "source_kind": source_kind,
                    "validation_passed": validation_passed,
                },
            )
        )
    if ranked:
        return max(ranked, key=lambda item: item[0])[1]
    return _generate_orszag_tang_fallback(output_dir)


def _generate_orszag_tang_fallback(output_dir: Path) -> dict[str, Any]:
    from mhx.benchmarks import write_orszag_tang_vortex_validation
    from mhx.runtime import configure_jax

    configure_jax(enable_x64=True)
    fallback_dir = output_dir / "generated_orszag_tang_validation_t4"
    history_path = fallback_dir / "orszag_tang_vortex.npz"
    if not history_path.exists():
        write_orszag_tang_vortex_validation(
            fallback_dir,
            shape=(48, 48),
            t_end=4.0,
            save_every=40,
            movies=False,
        )
    validation_path = fallback_dir / "validation.json"
    validation_passed = None
    if validation_path.exists():
        validation_passed = bool(json.loads(validation_path.read_text()).get("passed"))
    return {
        "history_path": history_path,
        "source_kind": "generated README fallback artifact",
        "validation_passed": validation_passed,
    }


def _load_orszag_tang_history(path: Path, *, fields_only: bool = False) -> dict[str, Any]:
    with np.load(path) as data:
        time = np.asarray(data["time"], dtype=float)
        psi = np.asarray(data["psi"], dtype=float)
        omega = np.asarray(data["omega"], dtype=float)
        current = np.asarray(data["current_density"], dtype=float)
        current_high_k = np.asarray(data["current_high_k_fraction"], dtype=float)
        vorticity_high_k = np.asarray(data["vorticity_high_k_fraction"], dtype=float)
        total_energy = np.asarray(data["total_energy"], dtype=float)
    if time.ndim != 1 or psi.ndim != 3:
        raise ValueError("invalid Orszag-Tang history arrays")
    result: dict[str, Any] = {
        "time": time,
        "psi": psi,
        "omega": omega,
        "current_density": current,
        "current_high_k_fraction": current_high_k,
        "vorticity_high_k_fraction": vorticity_high_k,
        "total_energy": total_energy,
        "shape": tuple(int(value) for value in psi.shape[1:]),
        "t_end": float(time[-1]),
    }
    if fields_only:
        return result
    return result


def _write_turbulence_readme_movies(
    output_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    decaying_source = _select_turbulence_history(
        output_dir,
        history_name="decaying_mhd_turbulence.npz",
        fallback_writer=_generate_decaying_turbulence_fallback,
    )
    forced_source = _select_turbulence_history(
        output_dir,
        history_name="forced_turbulent_reconnection.npz",
        fallback_writer=_generate_forced_turbulent_reconnection_fallback,
    )
    decaying = _load_turbulence_history(decaying_source["history_path"])
    forced = _load_turbulence_history(forced_source["history_path"])
    decaying_path = output_dir / "decaying_mhd_turbulence_current.gif"
    forced_path = output_dir / "forced_turbulent_reconnection.gif"
    _write_turbulence_current_contour_movie(
        decaying,
        path=decaying_path,
        title_prefix="Decaying MHD turbulence",
        max_frames=TURBULENCE_MAX_FRAMES,
    )
    _write_turbulence_current_contour_movie(
        forced,
        path=forced_path,
        title_prefix="Forced turbulent reconnection",
        max_frames=TURBULENCE_MAX_FRAMES,
    )
    snapshots = _write_turbulence_snapshot_contact_sheets(output_dir, decaying, forced)
    decaying_common = _history_source_manifest(decaying, decaying_source)
    forced_common = _history_source_manifest(forced, forced_source)
    qa = {
        "decaying_mhd_turbulence": _turbulence_visual_qa(
            decaying,
            decaying_source,
            snapshots["decaying_mhd_turbulence"],
        ),
        "forced_turbulent_reconnection": _turbulence_visual_qa(
            forced,
            forced_source,
            snapshots["forced_turbulent_reconnection"],
        ),
    }
    return (
        [
            _gif_manifest_entry(
                decaying_path,
                source=decaying_common,
                t_end=decaying["t_end"],
                time_span=decaying_common["time_span"],
                notes=(
                    "Solver-generated decaying reduced-MHD turbulence movie: "
                    "current density with magnetic-flux contours."
                ),
            ),
            _gif_manifest_entry(
                forced_path,
                source=forced_common,
                t_end=forced["t_end"],
                time_span=forced_common["time_span"],
                notes=(
                    "Solver-generated forced turbulent current-sheet movie with "
                    "reconnection-rate proxy diagnostics."
                ),
            ),
        ],
        qa,
    )


def _select_turbulence_history(
    output_dir: Path,
    *,
    history_name: str,
    fallback_writer: Any,
) -> dict[str, Any]:
    candidates = sorted(
        {
            *Path("outputs/readme_media").glob(f"**/{history_name}"),
            *Path("outputs/long_runs").glob(f"**/{history_name}"),
            *Path("outputs/docs_validation").glob(f"**/{history_name}"),
            *Path("outputs/ci").glob(f"**/{history_name}"),
            *output_dir.glob(f"generated_*_validation/**/{history_name}"),
        }
    )
    ranked: list[tuple[tuple[int, float, int, float], dict[str, Any]]] = []
    for path in candidates:
        try:
            history = _load_turbulence_history(path, fields_only=True)
        except (KeyError, ValueError, OSError):
            continue
        validation_path = path.parent / "validation.json"
        validation_passed = None
        if validation_path.exists():
            validation_passed = bool(json.loads(validation_path.read_text()).get("passed"))
        source_kind = "precomputed nonlinear validation artifact"
        if output_dir in path.parents:
            source_kind = "generated README fallback artifact"
        diagnostics_path = path.parent / "diagnostics.json"
        activity_score = 0.0
        if diagnostics_path.exists():
            diagnostics = json.loads(diagnostics_path.read_text())
            activity_score = max(
                float(diagnostics.get("reconnection_proxy_change", 0.0) or 0.0),
                float(diagnostics.get("current_linf_growth", 0.0) or 0.0),
            )
        validation_rank = (
            2 if validation_passed is True else 1 if validation_passed is None else 0
        )
        rank = (
            validation_rank,
            float(history["t_end"]),
            int(history["shape"][0] * history["shape"][1]),
            activity_score,
        )
        ranked.append(
            (
                rank,
                {
                    "history_path": path,
                    "source_kind": source_kind,
                    "validation_passed": validation_passed,
                },
            )
        )
    if ranked:
        return max(ranked, key=lambda item: item[0])[1]
    return fallback_writer(output_dir)


def _generate_decaying_turbulence_fallback(output_dir: Path) -> dict[str, Any]:
    from mhx.benchmarks import write_decaying_mhd_turbulence_validation
    from mhx.runtime import configure_jax

    configure_jax(enable_x64=True)
    fallback_dir = output_dir / "generated_decaying_turbulence_validation"
    history_path = fallback_dir / "decaying_mhd_turbulence.npz"
    if not history_path.exists():
        write_decaying_mhd_turbulence_validation(
            fallback_dir,
            shape=(32, 32),
            t_end=4.0,
            save_every=20,
            movies=False,
        )
    return _fallback_source(history_path)


def _generate_forced_turbulent_reconnection_fallback(output_dir: Path) -> dict[str, Any]:
    from mhx.benchmarks import write_forced_turbulent_reconnection_validation
    from mhx.runtime import configure_jax

    configure_jax(enable_x64=True)
    fallback_dir = output_dir / "generated_forced_turbulent_reconnection_validation"
    history_path = fallback_dir / "forced_turbulent_reconnection.npz"
    if not history_path.exists():
        write_forced_turbulent_reconnection_validation(
            fallback_dir,
            shape=(32, 32),
            t_end=20.0,
            save_every=50,
            movies=False,
        )
    return _fallback_source(history_path)


def _fallback_source(history_path: Path) -> dict[str, Any]:
    validation_path = history_path.parent / "validation.json"
    validation_passed = None
    if validation_path.exists():
        validation_passed = bool(json.loads(validation_path.read_text()).get("passed"))
    return {
        "history_path": history_path,
        "source_kind": "generated README fallback artifact",
        "validation_passed": validation_passed,
    }


def _load_turbulence_history(path: Path, *, fields_only: bool = False) -> dict[str, Any]:
    with np.load(path) as data:
        time = np.asarray(data["time"], dtype=float)
        psi = np.asarray(data["psi"], dtype=float)
        current = np.asarray(data["current_density"], dtype=float)
        total_energy = np.asarray(data["total_energy"], dtype=float)
        current_high_k = np.asarray(data["current_high_k_fraction"], dtype=float)
        reconnection_proxy = (
            np.asarray(data["reconnection_proxy"], dtype=float)
            if "reconnection_proxy" in data.files
            else None
        )
        reconnection_rate = (
            np.asarray(data["reconnection_rate_proxy"], dtype=float)
            if "reconnection_rate_proxy" in data.files
            else None
        )
    if time.ndim != 1 or psi.ndim != 3:
        raise ValueError("invalid turbulence history arrays")
    result: dict[str, Any] = {
        "time": time,
        "psi": psi,
        "current_density": current,
        "total_energy": total_energy,
        "current_high_k_fraction": current_high_k,
        "reconnection_proxy": reconnection_proxy,
        "reconnection_rate_proxy": reconnection_rate,
        "shape": tuple(int(value) for value in psi.shape[1:]),
        "t_end": float(time[-1]),
    }
    if fields_only:
        return result
    return result


def _history_source_manifest(history: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": str(source["history_path"]),
        "source_kind": source["source_kind"],
        "source_samples": int(len(history["time"])),
        "source_shape": list(history["shape"]),
        "t_end": history["t_end"],
        "time_span": [float(history["time"][0]), float(history["time"][-1])],
        "validation_passed": source.get("validation_passed"),
    }


def _load_double_harris_history(path: Path, *, fields_only: bool = False) -> dict[str, Any]:
    """Load double-Harris replay arrays and derived current-density frames."""
    with np.load(path) as data:
        perturbed_time = np.asarray(data["perturbed_time"], dtype=float)
        saved_time = np.concatenate(([0.0], perturbed_time))
        psi = np.concatenate(
            (
                np.asarray(data["perturbed_initial_psi"], dtype=float)[None, ...],
                np.asarray(data["perturbed_psi"], dtype=float),
            ),
            axis=0,
        )
        base_psi = np.concatenate(
            (
                np.asarray(data["base_initial_psi"], dtype=float)[None, ...],
                np.asarray(data["base_psi"], dtype=float),
            ),
            axis=0,
        )
    shape = tuple(int(value) for value in psi.shape[1:])
    result: dict[str, Any] = {
        "time": saved_time,
        "psi": psi,
        "base_psi": base_psi,
        "delta_psi": psi - base_psi,
        "shape": shape,
        "t_end": float(saved_time[-1]),
    }
    if fields_only:
        return result
    result["current"] = _current_density_frames(psi)
    result["delta_current"] = _current_density_frames(result["delta_psi"])
    return result


def _current_density_frames(psi_frames: np.ndarray) -> np.ndarray:
    from mhx.equations.reduced_mhd import current_density

    return np.asarray(
        [
            np.asarray(current_density(psi, lengths=DOUBLE_HARRIS_LENGTHS), dtype=float)
            for psi in psi_frames
        ]
    )


def _sample_frame_indices(frame_count: int, max_frames: int) -> np.ndarray:
    if frame_count <= max_frames:
        return np.arange(frame_count)
    return np.unique(np.linspace(0, frame_count - 1, max_frames, dtype=int))


def _write_field_movie(
    fields: np.ndarray,
    times: np.ndarray,
    *,
    path: Path,
    cmap: str,
    title_prefix: str,
    source_label: str,
    symmetric: bool = False,
) -> None:
    from matplotlib import colormaps

    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.asarray(fields)
    if symmetric:
        vmax = max(float(np.max(np.abs(values))), np.finfo(float).eps)
        vmin = -vmax
    else:
        vmin = float(np.min(values))
        vmax = float(np.max(values))
    colormap = colormaps[cmap]
    frames = []
    for field, time_value in zip(values, times, strict=True):
        _ = (time_value, title_prefix, source_label)
        normalized = np.clip((np.asarray(field).T - vmin) / (vmax - vmin), 0.0, 1.0)
        frames.append((255.0 * colormap(normalized)[..., :3]).astype(np.uint8))
    imageio.mimsave(
        path,
        frames,
        duration=README_GIF_DURATION_MS,
        loop=0,
        palettesize=48,
    )


def _write_turbulence_current_contour_movie(
    history: dict[str, Any],
    *,
    path: Path,
    title_prefix: str,
    max_frames: int,
) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    frame_indices = _sample_frame_indices(len(history["time"]), max_frames)
    current = np.asarray(history["current_density"])
    psi = np.asarray(history["psi"])
    x = np.linspace(0.0, DOUBLE_HARRIS_LENGTHS[0], history["shape"][0], endpoint=False)
    y = np.linspace(0.0, DOUBLE_HARRIS_LENGTHS[1], history["shape"][1], endpoint=False)
    x_mesh, y_mesh = np.meshgrid(x, y, indexing="ij")
    vmax = max(float(np.percentile(np.abs(current[frame_indices]), 99.2)), np.finfo(float).eps)
    frames = []
    for index in frame_indices:
        fig, ax = plt.subplots(figsize=(3.25, 3.0), dpi=72, constrained_layout=True)
        ax.imshow(
            current[index].T,
            origin="lower",
            extent=(0.0, DOUBLE_HARRIS_LENGTHS[0], 0.0, DOUBLE_HARRIS_LENGTHS[1]),
            aspect="equal",
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
        )
        levels = np.linspace(
            float(np.percentile(psi[index], 5.0)),
            float(np.percentile(psi[index], 95.0)),
            18,
        )
        ax.contour(x_mesh, y_mesh, psi[index], levels=levels, colors="black", linewidths=0.35)
        ax.set_title(f"{title_prefix}, t={history['time'][index]:.1f}", fontsize=9)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        frames.append(_figure_to_frame(fig))
        plt.close(fig)
    imageio.mimsave(path, frames, duration=README_GIF_DURATION_MS, loop=0, palettesize=48)


def _write_harris_sheet_contour_movie(
    history: dict[str, Any],
    frame_indices: np.ndarray,
    *,
    path: Path,
    source_label: str,
) -> None:
    """Write a single-sheet Harris reconnection movie with Az/psi contours."""
    import matplotlib.pyplot as plt

    frames = []
    y, normal, current_crop, psi_crop = _left_sheet_crop(history)
    vmax = max(float(np.percentile(np.abs(current_crop), 99.5)), np.finfo(float).eps)
    psi_levels = np.linspace(
        float(np.percentile(psi_crop, 3.0)),
        float(np.percentile(psi_crop, 97.0)),
        16,
    )
    for index in frame_indices:
        fig, ax = plt.subplots(figsize=(3.8, 2.75), dpi=72, constrained_layout=True)
        image = ax.imshow(
            current_crop[index],
            origin="lower",
            extent=(0.0, DOUBLE_HARRIS_LENGTHS[1], normal[0], normal[-1]),
            aspect="auto",
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
        )
        ax.contour(
            y,
            normal,
            psi_crop[index],
            levels=psi_levels,
            colors="black",
            linewidths=0.45,
            alpha=0.78,
        )
        _mark_sheet_critical_points(
            ax,
            _detect_critical_points_for_frame(history, index),
        )
        ax.set_title(f"Harris reconnection: Az contours, t={history['time'][index]:.0f}")
        ax.set_xlabel("sheet coordinate")
        ax.set_ylabel("normal to sheet")
        ax.set_xlim(0.0, DOUBLE_HARRIS_LENGTHS[1])
        ax.set_ylim(normal[0], normal[-1])
        _ = image, source_label
        frames.append(_figure_to_frame(fig))
        plt.close(fig)
    imageio.mimsave(path, frames, duration=README_GIF_DURATION_MS, loop=0, palettesize=32)


def _write_harris_full_domain_contour_movie(
    history: dict[str, Any],
    frame_indices: np.ndarray,
    *,
    path: Path,
    source_label: str,
) -> None:
    """Write a full periodic double-Harris current-sheet movie with Az contours."""
    import matplotlib.pyplot as plt

    frames = []
    _, _, x_mesh, y_mesh = _history_mesh(history)
    current = history["current"]
    psi = history["psi"]
    vmax = max(float(np.percentile(np.abs(current), 99.5)), np.finfo(float).eps)
    psi_levels = np.linspace(float(np.percentile(psi, 2.0)), float(np.percentile(psi, 98.0)), 20)
    for index in frame_indices:
        fig, ax = plt.subplots(figsize=(3.8, 2.75), dpi=72, constrained_layout=True)
        image = ax.imshow(
            current[index].T,
            origin="lower",
            extent=(0.0, DOUBLE_HARRIS_LENGTHS[0], 0.0, DOUBLE_HARRIS_LENGTHS[1]),
            aspect="auto",
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
        )
        ax.contour(
            x_mesh,
            y_mesh,
            psi[index],
            levels=psi_levels,
            colors="black",
            linewidths=0.4,
            alpha=0.76,
        )
        _mark_domain_critical_points(
            ax,
            _detect_critical_points_for_frame(history, index),
        )
        ax.set_title(f"Periodic double-Harris sheets, t={history['time'][index]:.0f}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        _ = image, source_label
        frames.append(_figure_to_frame(fig))
        plt.close(fig)
    imageio.mimsave(path, frames, duration=README_GIF_DURATION_MS, loop=0, palettesize=32)


def _history_mesh(
    history: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    shape = history["shape"]
    x = np.linspace(0.0, DOUBLE_HARRIS_LENGTHS[0], shape[0], endpoint=False)
    y = np.linspace(0.0, DOUBLE_HARRIS_LENGTHS[1], shape[1], endpoint=False)
    x_mesh, y_mesh = np.meshgrid(x, y, indexing="ij")
    return x, y, x_mesh, y_mesh


def _left_sheet_crop(
    history: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x, y, _, _ = _history_mesh(history)
    sheet = 0.25 * DOUBLE_HARRIS_LENGTHS[0]
    normal = ((x - sheet + 0.5 * DOUBLE_HARRIS_LENGTHS[0]) % DOUBLE_HARRIS_LENGTHS[0]) - (
        0.5 * DOUBLE_HARRIS_LENGTHS[0]
    )
    mask = np.abs(normal) <= DOUBLE_HARRIS_SHEET_HALF_WIDTH
    order = np.argsort(normal[mask])
    return (
        y,
        normal[mask][order],
        history["current"][:, mask, :][:, order, :],
        history["psi"][:, mask, :][:, order, :],
    )


def _detect_critical_points_for_frame(
    history: dict[str, Any],
    index: int,
) -> tuple[FluxCriticalPoint, ...]:
    return detect_flux_critical_points(
        np.asarray(history["psi"][index]),
        lengths=DOUBLE_HARRIS_LENGTHS,
        periodic=(True, True),
        max_points=12,
        min_separation=0.35,
    )


def _mark_sheet_critical_points(
    ax,
    points: tuple[FluxCriticalPoint, ...],
) -> None:
    sheet = 0.25 * DOUBLE_HARRIS_LENGTHS[0]
    for point in points:
        x_position, y_position = point.position
        normal = (
            (x_position - sheet + 0.5 * DOUBLE_HARRIS_LENGTHS[0])
            % DOUBLE_HARRIS_LENGTHS[0]
        ) - 0.5 * DOUBLE_HARRIS_LENGTHS[0]
        if abs(normal) > DOUBLE_HARRIS_SHEET_HALF_WIDTH:
            continue
        _draw_critical_point_marker(
            ax,
            x=y_position,
            y=normal,
            kind=point.kind,
            label_offset=(0.10, 0.10 if point.kind == "O" else -0.24),
        )


def _mark_domain_critical_points(
    ax,
    points: tuple[FluxCriticalPoint, ...],
) -> None:
    for point in points:
        _draw_critical_point_marker(
            ax,
            x=point.position[0],
            y=point.position[1],
            kind=point.kind,
            label_offset=(0.08, 0.08),
            label=False,
        )


def _draw_critical_point_marker(
    ax,
    *,
    x: float,
    y: float,
    kind: str,
    label_offset: tuple[float, float],
    label: bool = True,
) -> None:
    if kind == "O":
        ax.scatter([x], [y], s=34, marker="o", facecolor="white", edgecolor="black", zorder=5)
        text_color = "black"
    elif kind == "X":
        ax.scatter([x], [y], s=42, marker="x", color="white", linewidths=1.5, zorder=5)
        text_color = "white"
    else:
        return
    if label:
        ax.text(
            x + label_offset[0],
            y + label_offset[1],
            kind,
            fontsize=8,
            weight="bold",
            color=text_color,
            zorder=6,
        )


def _write_double_harris_snapshot_contact_sheets(
    output_dir: Path,
    history: dict[str, Any],
    source_label: str,
) -> dict[str, str]:
    flux_path = output_dir / "double_harris_flux_snapshots.png"
    current_path = output_dir / "double_harris_current_snapshots.png"
    current_sheet_path = output_dir / "double_harris_current_sheet_snapshots.png"
    _write_harris_sheet_snapshot_contact_sheet(
        history,
        path=flux_path,
        source_label=source_label,
    )
    _write_harris_full_snapshot_contact_sheet(
        history,
        path=current_path,
        source_label=source_label,
    )
    _write_snapshot_contact_sheet(
        history["current"],
        history["time"],
        path=current_sheet_path,
        cmap="RdBu_r",
        title="Fixed-scale total double-Harris current-density snapshots",
        source_label=source_label,
        symmetric=True,
    )
    return {
        "single_sheet_contours": str(flux_path),
        "full_domain_contours": str(current_path),
        "current_sheet": str(current_sheet_path),
    }


def _write_harris_sheet_snapshot_contact_sheet(
    history: dict[str, Any],
    *,
    path: Path,
    source_label: str,
) -> None:
    import matplotlib.pyplot as plt

    indices = [
        0,
        len(history["time"]) // 3,
        2 * len(history["time"]) // 3,
        len(history["time"]) - 1,
    ]
    y, normal, current_crop, psi_crop = _left_sheet_crop(history)
    vmax = max(float(np.percentile(np.abs(current_crop), 99.5)), np.finfo(float).eps)
    psi_levels = np.linspace(
        float(np.percentile(psi_crop, 3.0)),
        float(np.percentile(psi_crop, 97.0)),
        22,
    )
    fig, axes = plt.subplots(1, 4, figsize=(12.0, 3.2), constrained_layout=True)
    for ax, index in zip(axes, indices, strict=True):
        image = ax.imshow(
            current_crop[index],
            origin="lower",
            extent=(0.0, DOUBLE_HARRIS_LENGTHS[1], normal[0], normal[-1]),
            aspect="auto",
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
        )
        ax.contour(
            y,
            normal,
            psi_crop[index],
            levels=psi_levels,
            colors="black",
            linewidths=0.45,
            alpha=0.78,
        )
        _mark_sheet_critical_points(
            ax,
            _detect_critical_points_for_frame(history, index),
        )
        ax.set_title(f"t={history['time'][index]:.0f}")
        ax.set_xlabel("sheet coordinate")
        ax.set_ylabel("normal")
    fig.colorbar(image, ax=axes, shrink=0.76, label=r"$j_z$")
    fig.suptitle(
        f"Harris-sheet reconnection snapshots: current density + Az contours\n{source_label}"
    )
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _write_harris_full_snapshot_contact_sheet(
    history: dict[str, Any],
    *,
    path: Path,
    source_label: str,
) -> None:
    import matplotlib.pyplot as plt

    indices = [
        0,
        len(history["time"]) // 3,
        2 * len(history["time"]) // 3,
        len(history["time"]) - 1,
    ]
    x, y, x_mesh, y_mesh = _history_mesh(history)
    del x, y
    current = history["current"]
    psi = history["psi"]
    vmax = max(float(np.percentile(np.abs(current), 99.5)), np.finfo(float).eps)
    psi_levels = np.linspace(float(np.percentile(psi, 2.0)), float(np.percentile(psi, 98.0)), 28)
    fig, axes = plt.subplots(1, 4, figsize=(12.0, 3.2), constrained_layout=True)
    for ax, index in zip(axes, indices, strict=True):
        image = ax.imshow(
            current[index].T,
            origin="lower",
            extent=(0.0, DOUBLE_HARRIS_LENGTHS[0], 0.0, DOUBLE_HARRIS_LENGTHS[1]),
            aspect="auto",
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
        )
        ax.contour(
            x_mesh,
            y_mesh,
            psi[index],
            levels=psi_levels,
            colors="black",
            linewidths=0.4,
            alpha=0.76,
        )
        _mark_domain_critical_points(
            ax,
            _detect_critical_points_for_frame(history, index),
        )
        ax.set_title(f"t={history['time'][index]:.0f}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    fig.colorbar(image, ax=axes, shrink=0.76, label=r"$j_z$")
    fig.suptitle(
        f"Periodic double-Harris snapshots: current density + Az contours\n{source_label}"
    )
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _write_snapshot_contact_sheet(
    fields: np.ndarray,
    times: np.ndarray,
    *,
    path: Path,
    cmap: str,
    title: str,
    source_label: str,
    symmetric: bool = False,
) -> None:
    import matplotlib.pyplot as plt

    indices = [0, len(times) // 2, len(times) - 1]
    values = np.asarray(fields)
    if symmetric:
        vmax = max(float(np.max(np.abs(values))), np.finfo(float).eps)
        vmin = -vmax
    else:
        vmin = float(np.min(values))
        vmax = float(np.max(values))
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.2), constrained_layout=True)
    for ax, index in zip(axes, indices, strict=True):
        image = ax.imshow(
            values[index].T,
            origin="lower",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            extent=(0.0, DOUBLE_HARRIS_LENGTHS[0], 0.0, DOUBLE_HARRIS_LENGTHS[1]),
        )
        ax.set_title(f"t={times[index]:.1f}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    fig.colorbar(image, ax=axes, shrink=0.75)
    fig.suptitle(f"{title}\n{source_label}", fontsize=10)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _double_harris_visual_qa(
    history: dict[str, Any],
    source: dict[str, Any],
    snapshots: dict[str, str],
) -> dict[str, Any]:
    time = history["time"]
    mid_index = len(time) // 2
    current_peaks = np.max(np.abs(history["current"]), axis=(1, 2))
    delta_current_peaks = np.max(np.abs(history["delta_current"]), axis=(1, 2))
    delta_linf = np.max(np.abs(history["delta_psi"]), axis=(1, 2))
    peak_index = int(np.argmax(current_peaks))
    delta_peak_index = int(np.argmax(delta_current_peaks))
    growth_factor = float(np.max(delta_linf) / max(delta_linf[0], np.finfo(float).tiny))
    return {
        "source": str(source["history_path"]),
        "snapshot_contact_sheets": snapshots,
        "inspected_frames": [
            {"label": "first", "index": 0, "time": float(time[0])},
            {"label": "mid", "index": int(mid_index), "time": float(time[mid_index])},
            {"label": "final", "index": int(len(time) - 1), "time": float(time[-1])},
        ],
        "metrics": {
            "current_linf_first": float(current_peaks[0]),
            "current_linf_mid": float(current_peaks[mid_index]),
            "current_linf_final": float(current_peaks[-1]),
            "current_linf_peak": float(current_peaks[peak_index]),
            "current_linf_peak_time": float(time[peak_index]),
            "delta_current_linf_peak": float(delta_current_peaks[delta_peak_index]),
            "delta_current_linf_peak_time": float(time[delta_peak_index]),
            "flux_delta_linf_first": float(delta_linf[0]),
            "flux_delta_linf_mid": float(delta_linf[mid_index]),
            "flux_delta_linf_final": float(delta_linf[-1]),
            "flux_delta_linf_peak": float(np.max(delta_linf)),
            "flux_delta_growth_factor": growth_factor,
        },
        "observations": [
            (
                "The README Harris movie now shows the total out-of-plane current "
                "with magnetic-flux/Az contours, cropped around one current sheet "
                "so the automatically detected X/O topology is visible."
            ),
            (
                "X/O markers are selected by local |grad(Az)| minima and Hessian "
                "classification; locations are grid-localized rather than "
                "sub-cell Newton-refined."
            ),
            (
                "Flux contours evolve from a nearly straight Harris sheet into a "
                "single-island topology around the sheet, while total energy remains "
                "dissipative in the validation manifest."
            ),
            (
                "Perturbed-minus-base flux remains tracked quantitatively: it grows from "
                f"{delta_linf[0]:.3e} at t={time[0]:.1f} to "
                f"{delta_linf[-1]:.3e} at t={time[-1]:.1f}, with peak growth factor "
                f"{growth_factor:.2f}, consistent with a "
                "bounded nonlinear validation replay; this is not yet a "
                "production plasmoid/Rutherford claim."
            ),
        ],
    }


def _write_orszag_tang_snapshot_contact_sheets(
    output_dir: Path,
    history: dict[str, Any],
    source_label: str,
) -> dict[str, str]:
    current_path = output_dir / "orszag_tang_current_snapshots.png"
    vorticity_path = output_dir / "orszag_tang_vorticity_snapshots.png"
    flux_path = output_dir / "orszag_tang_flux_snapshots.png"
    _write_snapshot_contact_sheet(
        history["current_density"],
        history["time"],
        path=current_path,
        cmap="RdBu_r",
        title="Reduced-MHD Orszag-Tang current-density snapshots",
        source_label=source_label,
        symmetric=True,
    )
    _write_snapshot_contact_sheet(
        history["omega"],
        history["time"],
        path=vorticity_path,
        cmap="RdBu_r",
        title="Reduced-MHD Orszag-Tang vorticity snapshots",
        source_label=source_label,
        symmetric=True,
    )
    _write_snapshot_contact_sheet(
        history["psi"],
        history["time"],
        path=flux_path,
        cmap="viridis",
        title="Reduced-MHD Orszag-Tang flux snapshots",
        source_label=source_label,
        symmetric=False,
    )
    return {
        "current": str(current_path),
        "vorticity": str(vorticity_path),
        "flux": str(flux_path),
    }


def _write_turbulence_snapshot_contact_sheets(
    output_dir: Path,
    decaying: dict[str, Any],
    forced: dict[str, Any],
) -> dict[str, str]:
    decaying_path = output_dir / "decaying_mhd_turbulence_snapshots.png"
    forced_path = output_dir / "forced_turbulent_reconnection_snapshots.png"
    _write_snapshot_contact_sheet(
        decaying["current_density"],
        decaying["time"],
        path=decaying_path,
        cmap="RdBu_r",
        title="Decaying reduced-MHD turbulence current-density snapshots",
        source_label=f"{decaying['shape'][0]}×{decaying['shape'][1]}, t≤{decaying['t_end']:.0f}",
        symmetric=True,
    )
    _write_snapshot_contact_sheet(
        forced["current_density"],
        forced["time"],
        path=forced_path,
        cmap="RdBu_r",
        title="Forced turbulent reconnection current-density snapshots",
        source_label=f"{forced['shape'][0]}×{forced['shape'][1]}, t≤{forced['t_end']:.0f}",
        symmetric=True,
    )
    return {
        "decaying_mhd_turbulence": str(decaying_path),
        "forced_turbulent_reconnection": str(forced_path),
    }


def _orszag_tang_visual_qa(
    history: dict[str, Any],
    source: dict[str, Any],
    snapshots: dict[str, str],
) -> dict[str, Any]:
    time = history["time"]
    mid_index = len(time) // 2
    current_peaks = np.max(np.abs(history["current_density"]), axis=(1, 2))
    vorticity_peaks = np.max(np.abs(history["omega"]), axis=(1, 2))
    current_high_k = history["current_high_k_fraction"]
    vorticity_high_k = history["vorticity_high_k_fraction"]
    return {
        "source": str(source["history_path"]),
        "snapshot_contact_sheets": snapshots,
        "inspected_frames": [
            {"label": "first", "index": 0, "time": float(time[0])},
            {"label": "mid", "index": int(mid_index), "time": float(time[mid_index])},
            {"label": "final", "index": int(len(time) - 1), "time": float(time[-1])},
        ],
        "metrics": {
            "current_linf_first": float(current_peaks[0]),
            "current_linf_mid": float(current_peaks[mid_index]),
            "current_linf_final": float(current_peaks[-1]),
            "current_high_k_first": float(current_high_k[0]),
            "current_high_k_peak": float(np.max(current_high_k)),
            "vorticity_linf_first": float(vorticity_peaks[0]),
            "vorticity_linf_mid": float(vorticity_peaks[mid_index]),
            "vorticity_linf_final": float(vorticity_peaks[-1]),
            "vorticity_high_k_first": float(vorticity_high_k[0]),
            "vorticity_high_k_peak": float(np.max(vorticity_high_k)),
            "relative_energy_drop": float(
                (history["total_energy"][0] - history["total_energy"][-1])
                / max(abs(history["total_energy"][0]), np.finfo(float).tiny)
            ),
        },
        "observations": [
            (
                "The Orszag-Tang README movies are solver output from the "
                "incompressible reduced-MHD initial condition, not a full "
                "compressible shock-capturing MHD calculation."
            ),
            (
                "Current-density and vorticity snapshots show nonlinear transfer "
                "from large-scale modes into smaller filaments, as tracked by the "
                "high-wavenumber fraction diagnostics."
            ),
            (
                "The fixed-scale movie emphasizes morphology; quantitative claims "
                "are limited to the validation manifest gates and are not turbulence "
                "statistics."
            ),
        ],
    }


def _turbulence_visual_qa(
    history: dict[str, Any],
    source: dict[str, Any],
    snapshot_contact_sheet: str,
) -> dict[str, Any]:
    time = history["time"]
    mid_index = len(time) // 2
    current_peaks = np.max(np.abs(history["current_density"]), axis=(1, 2))
    energy = history["total_energy"]
    metrics = {
        "current_linf_first": float(current_peaks[0]),
        "current_linf_mid": float(current_peaks[mid_index]),
        "current_linf_final": float(current_peaks[-1]),
        "current_linf_peak": float(np.max(current_peaks)),
        "current_high_k_first": float(history["current_high_k_fraction"][0]),
        "current_high_k_peak": float(np.max(history["current_high_k_fraction"])),
        "relative_energy_drop": float(
            (energy[0] - energy[-1]) / max(abs(energy[0]), np.finfo(float).tiny)
        ),
    }
    if history["reconnection_proxy"] is not None:
        metrics["reconnection_proxy_change"] = float(
            np.max(history["reconnection_proxy"]) - np.min(history["reconnection_proxy"])
        )
    if history["reconnection_rate_proxy"] is not None:
        metrics["max_abs_reconnection_rate_proxy"] = float(
            np.max(np.abs(history["reconnection_rate_proxy"]))
        )
    return {
        "source": str(source["history_path"]),
        "snapshot_contact_sheet": snapshot_contact_sheet,
        "inspected_frames": [
            {"label": "first", "index": 0, "time": float(time[0])},
            {"label": "mid", "index": int(mid_index), "time": float(time[mid_index])},
            {"label": "final", "index": int(len(time) - 1), "time": float(time[-1])},
        ],
        "metrics": metrics,
        "observations": [
            (
                "The turbulence README movies are solver output from deterministic "
                "2-D reduced-MHD validation examples, not synthetic schematics."
            ),
            (
                "The forced current-sheet case includes a reconnection proxy and "
                "is literature-anchored to turbulent reconnection, but remains a "
                "2-D pedagogical validation rather than a 3-D LV99 production claim."
            ),
        ],
    }


def _write_harris_layer_sweep(path: Path) -> None:
    """Animate the validated Harris eigenfunction-layer sweep over S."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, NullFormatter

    result = run_linear_tearing_layer_validation(grid_points=128)
    x = result.selected_coordinate
    frame_paths = []
    for lundquist in result.lundquist:
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2), constrained_layout=True)
        marker = int(np.argmin(np.abs(result.lundquist - lundquist)))
        axes[0].loglog(
            result.lundquist,
            result.growth_rate,
            "o-",
            color="#3266a8",
            label=r"$\gamma$",
        )
        axes[0].loglog(
            result.lundquist,
            result.stream_half_width,
            "s-",
            color="#b54a4a",
            label="flow width",
        )
        axes[0].scatter(
            [result.lundquist[marker]],
            [result.growth_rate[marker]],
            s=90,
            facecolors="none",
            edgecolors="black",
            linewidths=1.4,
        )
        axes[0].set_xlabel("Lundquist number S")
        axes[0].set_ylabel("growth / width")
        axes[0].set_xlim(220.0, 2300.0)
        axes[0].xaxis.set_major_locator(FixedLocator([250.0, 1000.0, 2000.0]))
        axes[0].set_xticklabels(["250", "1000", "2000"])
        axes[0].xaxis.set_minor_formatter(NullFormatter())
        axes[0].set_title("Harris tearing layer gate")
        axes[0].legend(frameon=False, fontsize=7)
        axes[1].plot(x, result.selected_flux_eigenfunction, label=r"$\psi_1$", lw=1.6)
        axes[1].plot(x, result.selected_streamfunction_imag, label=r"Im $\phi_1$", lw=1.6)
        axes[1].plot(x, result.selected_current_density, label=r"$j_1$", lw=1.2)
        axes[1].axvline(0.0, color="0.65", lw=0.8)
        axes[1].set_xlim(-4.0, 4.0)
        axes[1].set_xlabel(r"$x/a$")
        axes[1].set_title(f"reference profiles; frame S={lundquist:.0f}")
        axes[1].legend(frameon=False, fontsize=7)
        fig.suptitle("Literature-anchored Harris tearing eigenfunction localization")
        frame_paths.append(_figure_to_frame(fig))
        plt.close(fig)
    imageio.mimsave(path, frame_paths, duration=850, loop=0, palettesize=64)


def _write_plasmoid_scaling_schematic(path: Path) -> None:
    """Animate the Loureiro-Schekochihin-Cowley Sweet-Parker plasmoid scalings."""
    import matplotlib.pyplot as plt

    x = np.linspace(0.0, 2.0 * np.pi, 240)
    y = np.linspace(-1.0, 1.0, 100)
    x_mesh, y_mesh = np.meshgrid(x, y, indexing="xy")
    lundquist_values = np.geomspace(1.0e4, 1.0e6, 8)
    frames = []
    for lundquist in lundquist_values:
        normalized = lundquist / lundquist_values[0]
        mode_scaling = normalized ** (3.0 / 8.0)
        island_count = max(2, int(np.ceil(mode_scaling)))
        growth = normalized**0.25
        sheet_width = 0.22 * normalized**-0.5
        perturbation = 0.18 * growth / (1.0 + growth)
        flux = np.tanh(y_mesh / sheet_width) + perturbation * np.cos(
            island_count * x_mesh
        ) * np.exp(-((y_mesh / (2.2 * sheet_width)) ** 2))
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2), constrained_layout=True)
        axes[0].contour(x_mesh, y_mesh, flux, levels=20, linewidths=0.7, cmap="viridis")
        axes[0].set_xticks([])
        axes[0].set_yticks([])
        axes[0].set_title(f"schematic chain: N≈{island_count}")
        axes[0].set_xlabel("current-sheet direction")
        axes[0].set_ylabel("inflow direction")
        axes[1].loglog(
            lundquist_values,
            (lundquist_values / lundquist_values[0]) ** 0.25,
            "-",
            color="#3266a8",
            label=r"$\gamma_{\max}\tau_A\propto S^{1/4}$",
        )
        axes[1].loglog(
            lundquist_values,
            (lundquist_values / lundquist_values[0]) ** (3.0 / 8.0),
            "-",
            color="#b54a4a",
            label=r"$k_{\max}L\propto S^{3/8}$",
        )
        axes[1].scatter([lundquist], [growth], color="#3266a8", s=45)
        axes[1].scatter([lundquist], [mode_scaling], color="#b54a4a", s=45)
        axes[1].set_xlabel("global Lundquist number S")
        axes[1].set_ylabel("relative scaling")
        axes[1].set_title("Sweet-Parker plasmoid theory")
        axes[1].legend(frameon=False, fontsize=7)
        fig.suptitle("Loureiro-Schekochihin-Cowley plasmoid scaling schematic")
        frames.append(_figure_to_frame(fig))
        plt.close(fig)
    imageio.mimsave(path, frames, duration=650, loop=0, palettesize=64)


def _write_mhd_turbulence_schematic(path: Path) -> None:
    """Animate a compact schematic of 2-D MHD-like eddies and spectral transfer."""
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(20260513)
    n = 96
    x = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    y = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    x_mesh, y_mesh = np.meshgrid(x, y, indexing="ij")
    modes = []
    for kx in range(1, 7):
        for ky in range(1, 7):
            wavenumber = np.hypot(kx, ky)
            amplitude = wavenumber ** (-5.0 / 3.0)
            phase = rng.uniform(0.0, 2.0 * np.pi)
            drift = rng.normal(0.0, 0.35)
            modes.append((kx, ky, amplitude, phase, drift, wavenumber))
    frames = []
    cascade_k = np.arange(1, 26)
    for frame_index, phase_shift in enumerate(np.linspace(0.0, 2.0 * np.pi, 8)):
        flux = np.zeros_like(x_mesh)
        current = np.zeros_like(x_mesh)
        for kx, ky, amplitude, phase, drift, wavenumber in modes:
            angle = kx * x_mesh + ky * y_mesh + phase + drift * phase_shift
            flux += amplitude * np.sin(angle)
            current += (wavenumber**2) * amplitude * np.sin(angle)
        flux /= max(float(np.std(flux)), np.finfo(float).eps)
        current /= max(float(np.std(current)), np.finfo(float).eps)
        spectrum = cascade_k ** (-5.0 / 3.0)
        cutoff = 1.0 / (1.0 + np.exp(-(cascade_k - (5 + frame_index)) / 1.8))

        fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.35), constrained_layout=True)
        axes[0].imshow(flux.T, origin="lower", cmap="viridis")
        axes[0].set_title("magnetic-flux eddies")
        axes[0].set_xticks([])
        axes[0].set_yticks([])
        axes[1].imshow(current.T, origin="lower", cmap="magma")
        axes[1].set_title("current filaments")
        axes[1].set_xticks([])
        axes[1].set_yticks([])
        axes[2].loglog(cascade_k, spectrum, color="#3266a8", label=r"$k^{-5/3}$ guide")
        axes[2].loglog(cascade_k, spectrum * cutoff, color="#b54a4a", label="animated cascade")
        axes[2].set_xlabel("wavenumber")
        axes[2].set_ylabel("relative power")
        axes[2].set_title("turbulent transfer")
        axes[2].legend(frameon=False, fontsize=7)
        fig.suptitle("MHD turbulence schematic")
        frames.append(_figure_to_frame(fig))
        plt.close(fig)
    imageio.mimsave(path, frames, duration=200, loop=0, palettesize=48)


def _figure_to_frame(fig) -> np.ndarray:
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()


def _gif_manifest_entry(
    path: Path,
    *,
    source: str | dict[str, Any],
    t_end: float | None,
    time_span: list[float] | None,
    notes: str,
) -> dict[str, Any]:
    frames = imageio.mimread(path)
    frame_shape = list(np.asarray(frames[0]).shape)
    return {
        "path": str(path),
        "frame_count": len(frames),
        "frame_shape": frame_shape,
        "duration_ms": _gif_duration_ms(path),
        "source": source,
        "t_end": t_end,
        "time_span": time_span,
        "notes": notes,
    }


def _gif_duration_ms(path: Path) -> int | None:
    durations: list[int] = []
    reader = imageio.get_reader(path)
    try:
        for index in range(len(imageio.mimread(path))):
            duration = reader.get_meta_data(index=index).get("duration")
            if duration is not None:
                durations.append(int(duration))
    finally:
        reader.close()
    unique = sorted(set(durations))
    return unique[0] if len(unique) == 1 else None


def _write_visual_qa_manifest(
    output_dir: Path,
    media_entries: list[dict[str, Any]],
    visual_qa: dict[str, Any],
) -> None:
    manifest = {
        "schema": "mhx.readme_media_visual_qa.v1",
        "generated_utc": datetime.now(UTC).isoformat(),
        "source_policy": (
            "README solver movies use the longest available matching nonlinear "
            "NPZ under outputs/readme_media, outputs/long_runs, outputs/docs_validation, "
            "or outputs/ci. "
            "Double-Harris fallback is a labeled 64×64, t_end=60 replay; Orszag-Tang "
            "fallback is a labeled 48×48, t_end=4 replay; turbulence fallbacks are "
            "labeled 32×32 validation replays. Committed release media "
            "should be regenerated from longer precomputed artifacts before publication."
        ),
        "media": media_entries,
        "visual_qa": visual_qa,
    }
    (output_dir / "readme_media_visual_qa.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
