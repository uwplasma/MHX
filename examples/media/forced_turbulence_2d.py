"""Run and render continually forced two-dimensional MHD turbulence."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from .common import (
        POSTER_PREVIEW,
        ROOT,
        encode_frames,
        figure_frame,
        load_source_metadata,
        render_stem,
        require_source,
        sample_indices,
        source_dir,
        validate_array_shape,
        write_render_record,
        write_source_metadata,
    )
except ImportError:  # Direct script execution.
    from common import (
        POSTER_PREVIEW,
        ROOT,
        encode_frames,
        figure_frame,
        load_source_metadata,
        render_stem,
        require_source,
        sample_indices,
        source_dir,
        validate_array_shape,
        write_render_record,
        write_source_metadata,
    )

CASE = "forced_turbulence_2d"
PRESETS = {
    "preview": {"shape": (64, 64), "dt": 2.0e-3, "t_end": 2.0, "save_every": 100},
    "final": {"shape": (256, 256), "dt": 1.0e-3, "t_end": 50.0, "save_every": 1000},
}


def simulate(*, preset: str, outdir: Path) -> Path:
    """Run the forced turbulent-spectrum example and retain its full trajectory."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from examples.turbulence.benchmark_turbulent_spectrum import (
        write_turbulent_spectrum_validation,
    )

    config = PRESETS[preset]
    write_turbulent_spectrum_validation(
        outdir,
        shape=config["shape"],
        dt=config["dt"],
        t_end=config["t_end"],
        save_every=config["save_every"],
    )
    source = outdir / "turbulent_spectrum.npz"
    write_source_metadata(
        outdir,
        {"case": CASE, "preset": preset, **config, "trajectory": source.name},
    )
    _validate_stored_shape(source, config["shape"])
    return source


def _validate_stored_shape(source: Path, expected: tuple[int, int]) -> None:
    import numpy as np

    with np.load(source, allow_pickle=False) as data:
        validate_array_shape(data["current_density"].shape[1:], expected, label=CASE)


def _magnetic_spectrum(psi_field: object) -> tuple[object, object]:
    """Match the validation benchmark's isotropic magnetic-spectrum normalization."""
    import numpy as np

    values = np.asarray(psi_field, dtype=float)
    spectrum = np.fft.fftn(values)
    n = values.shape[0]
    kx = np.fft.fftfreq(n) * n
    ky = np.fft.fftfreq(n) * n
    kx_grid, ky_grid = np.meshgrid(kx, ky, indexing="ij")
    k_squared = kx_grid**2 + ky_grid**2
    power = k_squared * np.abs(spectrum) ** 2
    power *= ((2.0 * np.pi / (2.0 * np.pi)) ** 2) * ((2.0 * np.pi / n) ** 2 / n**2)
    radius = np.round(np.sqrt(k_squared)).astype(int)
    wavenumber = np.arange(1, n // 2)
    shell_power = np.asarray([np.sum(power[radius == value]) for value in wavenumber])
    return wavenumber, shell_power


def _constant_compensated_fit(
    wavenumber: object,
    compensated: object,
    *,
    cutoff: float,
) -> tuple[float, float, object]:
    """Fit a constant over the benchmark inertial band and return its NRMSE."""
    import numpy as np

    k = np.asarray(wavenumber, dtype=float)
    values = np.asarray(compensated, dtype=float)
    mask = (k >= 5.0) & (k <= cutoff) & np.isfinite(values) & (values > 0.0)
    if np.count_nonzero(mask) < 2:
        return float("nan"), float("nan"), mask
    constant = float(np.mean(values[mask]))
    nrmse = float(np.sqrt(np.mean((values[mask] - constant) ** 2)) / abs(constant))
    return constant, nrmse, mask


def _write_final_spectrum(stem: str, wavenumber: object, power: object) -> Path:
    import matplotlib.pyplot as plt
    import numpy as np

    POSTER_PREVIEW.mkdir(parents=True, exist_ok=True)
    figure, (axis, compensated_axis) = plt.subplots(
        1,
        2,
        figsize=(11.4, 5.0),
        dpi=120,
        constrained_layout=True,
    )
    wavenumber = np.asarray(wavenumber)
    power = np.asarray(power)
    mask = power > 0.0
    compensated = power * wavenumber ** (5.0 / 3.0)
    fit_constant, fit_nrmse, fit_mask = _constant_compensated_fit(
        wavenumber,
        compensated,
        cutoff=2.0 * (float(wavenumber[-1]) + 1.0) / 3.0,
    )
    axis.loglog(
        wavenumber[mask],
        power[mask],
        color="#00a6a6",
        linewidth=2,
    )
    axis.set_xlabel(r"wavenumber $k$")
    axis.set_ylabel(r"magnetic spectrum $E_B(k)$")
    axis.set_title("Final magnetic spectrum")
    axis.grid(True, which="both", alpha=0.2)
    compensated_axis.loglog(
        wavenumber[mask],
        compensated[mask],
        color="#d95f02",
        linewidth=2,
    )
    if np.isfinite(fit_constant):
        compensated_axis.hlines(
            fit_constant,
            float(wavenumber[fit_mask].min()),
            float(wavenumber[fit_mask].max()),
            colors="black",
            linestyles="--",
            linewidth=1.5,
            label=rf"constant fit; NRMSE={fit_nrmse:.1%}",
        )
    compensated_axis.set_xlabel(r"wavenumber $k$")
    compensated_axis.set_ylabel(r"$k^{5/3}E_B(k)$")
    compensated_axis.set_title(r"Compensated spectrum (flat $\sim k^{-5/3}$)")
    compensated_axis.grid(True, which="both", alpha=0.2)
    compensated_axis.legend(frameon=False, loc="best")
    figure.suptitle("Forced 2-D turbulence")
    path = POSTER_PREVIEW / f"{stem}_final_spectrum.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def _render_spectrum_frames(
    psi: object,
    time: object,
    indices: list[int],
) -> tuple[list[object], list[float]]:
    import matplotlib.pyplot as plt
    import numpy as np

    spectra = [_magnetic_spectrum(psi[index]) for index in indices]
    compensated_spectra = [power * wavenumber ** (5.0 / 3.0) for wavenumber, power in spectra]
    positive = [power[power > 0.0] for _, power in spectra]
    positive_compensated = [values[values > 0.0] for values in compensated_spectra]
    minimum = min(float(values.min()) for values in positive if values.size)
    maximum = max(float(values.max()) for values in positive if values.size)
    compensated_minimum = min(float(values.min()) for values in positive_compensated if values.size)
    compensated_maximum = max(float(values.max()) for values in positive_compensated if values.size)
    k_max = max(int(wavenumber[-1]) for wavenumber, _ in spectra)
    frames = []
    fit_errors = []
    for index, (wavenumber, power), compensated in zip(
        indices, spectra, compensated_spectra, strict=True
    ):
        figure, (axis, compensated_axis) = plt.subplots(
            1,
            2,
            figsize=(11.4, 5.4),
            dpi=100,
            constrained_layout=True,
        )
        mask = power > 0.0
        fit_constant, fit_nrmse, fit_mask = _constant_compensated_fit(
            wavenumber,
            compensated,
            cutoff=psi.shape[1] / 3.0,
        )
        fit_errors.append(fit_nrmse)
        axis.loglog(
            wavenumber[mask],
            power[mask],
            color="#00a6a6",
            linewidth=2.2,
        )
        axis.axvline(
            psi.shape[1] / 3.0,
            color="0.55",
            linestyle=":",
            linewidth=1.2,
            label="2/3 cutoff",
        )
        axis.set_xlim(1.0, k_max)
        axis.set_ylim(0.7 * minimum, 1.4 * maximum)
        axis.set_xlabel(r"wavenumber $k$")
        axis.set_ylabel(r"magnetic spectrum $E_B(k)$")
        axis.set_title(f"Magnetic spectrum, t = {time[index]:.1f}")
        axis.grid(True, which="both", alpha=0.2)
        axis.legend(frameon=False, loc="lower left")
        compensated_axis.loglog(
            wavenumber[mask],
            compensated[mask],
            color="#d95f02",
            linewidth=2.2,
        )
        if np.isfinite(fit_constant):
            compensated_axis.hlines(
                fit_constant,
                float(wavenumber[fit_mask].min()),
                float(wavenumber[fit_mask].max()),
                colors="black",
                linestyles="--",
                linewidth=1.5,
                label=rf"constant fit; NRMSE={fit_nrmse:.1%}",
            )
        compensated_axis.axvline(
            psi.shape[1] / 3.0,
            color="0.55",
            linestyle=":",
            linewidth=1.2,
            label="2/3 cutoff",
        )
        compensated_axis.set_xlim(1.0, k_max)
        compensated_axis.set_ylim(
            0.7 * compensated_minimum,
            1.4 * compensated_maximum,
        )
        compensated_axis.set_xlabel(r"wavenumber $k$")
        compensated_axis.set_ylabel(r"$k^{5/3}E_B(k)$")
        compensated_axis.set_title(r"Compensated spectrum (flat $\sim k^{-5/3}$)")
        compensated_axis.grid(True, which="both", alpha=0.2)
        compensated_axis.legend(frameon=False, loc="lower left")
        frames.append(figure_frame(figure))
        plt.close(figure)
    return frames, fit_errors


def render(*, source: Path, maximum_frames: int = 51) -> dict[str, Path]:
    """Render clear current-density and time-dependent spectrum movies."""
    import matplotlib.pyplot as plt
    import numpy as np

    source = require_source(source, CASE)
    metadata = load_source_metadata(source)
    with np.load(source, allow_pickle=False) as data:
        time = np.asarray(data["time"], dtype=float)
        psi = np.asarray(data["psi"], dtype=float)
        current = np.asarray(data["current_density"], dtype=float)
        wavenumber = np.asarray(data["k_1d"], dtype=float)
        power = np.asarray(data["power_1d"], dtype=float)
    if metadata.get("shape"):
        validate_array_shape(current.shape[1:], metadata["shape"], label=CASE)

    indices = sample_indices(len(time), maximum_frames)
    limit = max(
        float(np.percentile(np.abs(current[indices]), 99.5)),
        np.finfo(float).eps,
    )
    frames = []
    for index in indices:
        figure, axis = plt.subplots(figsize=(6.2, 5.4), dpi=100, constrained_layout=True)
        image = axis.imshow(
            current[index].T,
            origin="lower",
            extent=(0.0, 2.0 * np.pi, 0.0, 2.0 * np.pi),
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
            aspect="equal",
        )
        axis.set_xlabel(r"$x$")
        axis.set_ylabel(r"$y$")
        axis.set_title(f"Continually forced 2-D MHD turbulence, t = {time[index]:.1f}")
        figure.colorbar(image, ax=axis, label=r"$j_z$ (code units)", shrink=0.82)
        frames.append(figure_frame(figure))
        plt.close(figure)

    spectrum_frames, fit_errors = _render_spectrum_frames(psi, time, indices)
    combined_frames = [
        np.concatenate((field_frame, spectrum_frame), axis=1)
        for field_frame, spectrum_frame in zip(frames, spectrum_frames, strict=True)
    ]
    finite_fit_indices = [
        position for position, error in enumerate(fit_errors) if np.isfinite(error)
    ]
    poster_index = (
        min(finite_fit_indices, key=fit_errors.__getitem__)
        if finite_fit_indices
        else len(combined_frames) // 2
    )
    poster_metadata = {
        **metadata,
        "poster_selection": {
            "criterion": "minimum compensated-spectrum constant-fit NRMSE",
            "frame_position": int(poster_index),
            "source_index": int(indices[poster_index]),
            "time": float(time[indices[poster_index]]),
            "nrmse": float(fit_errors[poster_index]),
        },
    }
    outputs = encode_frames(
        combined_frames,
        stem="forced_2d_turbulence",
        source=source,
        source_metadata=poster_metadata,
        times=time[indices],
        write_gif=True,
        write_mp4=True,
        fps=10,
        poster_index=poster_index,
    )
    stem = render_stem("forced_2d_turbulence", metadata)
    outputs["final_spectrum"] = _write_final_spectrum(stem, wavenumber, power)
    write_render_record(
        stem=stem,
        source=source,
        source_metadata=poster_metadata,
        times=time[indices],
        outputs=outputs,
    )
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    simulate_parser = subparsers.add_parser("simulate")
    simulate_parser.add_argument("--preset", choices=PRESETS, default="preview")
    simulate_parser.add_argument("--outdir", type=Path)
    render_parser = subparsers.add_parser("render")
    render_parser.add_argument("--source", type=Path)
    render_parser.add_argument("--preset", choices=PRESETS, default="preview")
    render_parser.add_argument("--max-frames", type=int, default=51)
    args = parser.parse_args()
    if args.command == "simulate":
        outdir = args.outdir or source_dir(CASE, args.preset)
        print(simulate(preset=args.preset, outdir=outdir))
    else:
        source = args.source or source_dir(CASE, args.preset) / "turbulent_spectrum.npz"
        for path in render(source=source, maximum_frames=args.max_frames).values():
            print(path)


if __name__ == "__main__":
    main()
