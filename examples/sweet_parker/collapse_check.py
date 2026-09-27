#!/usr/bin/env python3
"""Pre-test: does the large-Delta' double-Harris tearing mode collapse into a Sweet-Parker sheet?

Setup (Waelbroeck 1993; Loureiro et al. 2005): a periodic double-Harris sheet of
half-width ``a`` in a ``Lx x Ly`` box seeded with the single mode ``(0, 1)``.
With ``k a = 2*pi*a/Ly`` small, ``Delta' a = 2(1/(ka) - ka)`` is large, so once the
island passes ``w ~ 1/Delta'`` the X-point collapses into a thin current sheet
whose reconnection rate plateaus. The sheets sit at ``Lx/4`` and ``3Lx/4``, so
``Lx = 8*pi`` keeps them ``4*pi`` apart and the islands clear of each other for
longer. ``--hold-equilibrium`` replaces ``eta*lap(psi)`` by
``eta*lap(psi - psi_eq)`` (a constant ``E_0 = eta*j_eq``) so the Harris sheet
does not diffuse away.

The run is advanced in chunks and the left sheet (``x ~ Lx/4``) is measured at
every saved time with :func:`mhx.diagnostics.measure_sweet_parker_sheet`, so no
full trajectory is kept in memory. At the end the script reports:

* X-point collapse: the sheet thins (``delta_min/delta_ref <= 0.5``), the X-point
  current grows (``max|j_X|/|j_X|_ref >= 2``) and the sheet is elongated
  (``L_j/delta >= 5`` at minimum thickness);
* the clean collapse phase: from ``|j_X| >= 1.5 |j_X|_ref`` until the first
  secondary X-point on the sheet, the island full width exceeding half the sheet
  separation, or ``|j_X|`` dropping back;
* a steady window inside that phase where ``eta|j_X|`` varies by less than 10 %
  over at least three Alfven transit times ``L/v_A``, and whether, in that
  window, ``delta/dx >= 8`` (resolved), ``v_out/v_A,up >= 0.5`` (Alfvenic
  outflow), ``L/delta >= 10`` (elongated sheet) and ``S_L >= 500``;
* informational Sweet-Parker consistency ratios in the window.

Outputs (in ``--outdir``): ``histories.npz``, ``snapshots.npz``,
``histories.png``, ``current_snapshots.png``, ``domain.gif`` (full-domain ``j_z``
with flux contours every ``--gif-interval``) and ``summary.json``. Snapshots and
GIF frames are rolled in ``y`` so the X-point sits at ``y = Ly/2``.

Usage::

    python examples/sweet_parker/collapse_check.py --hold-equilibrium

Re-assess and re-plot a finished run from its saved outputs without simulating
(``domain.gif`` needs a rerun because full-domain frames are not saved)::

    python examples/sweet_parker/collapse_check.py --replot outputs/sweet_parker/<run>
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from mhx.config import MeshConfig  # noqa: E402
from mhx.diagnostics import (  # noqa: E402
    measure_sweet_parker_sheet,
    select_steady_window,
    sweet_parker_prediction,
)
from mhx.equations.reduced_mhd import current_density, reduced_mhd_rhs  # noqa: E402
from mhx.grids import CartesianGrid  # noqa: E402
from mhx.physics import PeriodicDoubleHarrisEquilibrium  # noqa: E402
from mhx.state import ReducedMHDParams, ReducedMHDState  # noqa: E402
from mhx.time_integrators import evolve_rk4  # noqa: E402

HISTORY_KEYS = (
    "current_x",
    "reconnection_rate",
    "normalized_rate",
    "delta",
    "length",
    "length_current",
    "b_upstream",
    "v_inflow",
    "v_outflow",
    "opening_angle_deg",
    "tilt_deg",
    "sheet_x_point_count",
    "island_width",
    "delta_over_dx",
    "lundquist",
    "x_point_y",
)
VERDICT_KEYS = (
    "x_point_collapse",
    "steady_window_found",
    "resolved",
    "alfvenic_outflow",
    "elongated_sheet",
    "high_lundquist",
)
MIN_OUTFLOW_OVER_ALFVEN = 0.5
MIN_WINDOW_ASPECT_RATIO = 10.0
MIN_WINDOW_LUNDQUIST = 500.0
GIF_MAX_POINTS = 384


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--nx", type=int, default=3072)
    parser.add_argument("--ny", type=int, default=384)
    parser.add_argument("--lx", type=float, default=8.0 * math.pi)
    parser.add_argument("--ly", type=float, default=4.0 * math.pi)
    parser.add_argument("--width", type=float, default=0.5, help="Harris half-width a")
    parser.add_argument("--eta", type=float, default=2.0e-3, help="resistivity")
    parser.add_argument("--pm", type=float, default=1.0, help="magnetic Prandtl nu/eta")
    parser.add_argument("--seed", type=float, default=1.0e-3, help="seed flux amplitude")
    parser.add_argument("--mode-y", type=int, default=1)
    parser.add_argument(
        "--hold-equilibrium",
        action="store_true",
        help="use eta*lap(psi - psi_eq) so the Harris sheet does not diffuse",
    )
    parser.add_argument("--t-end", type=float, default=300.0)
    parser.add_argument("--save-interval", type=float, default=0.5)
    parser.add_argument("--cfl", type=float, default=0.25, help="dt = cfl * min(dx, dy)")
    parser.add_argument("--snapshots", type=int, default=8)
    parser.add_argument(
        "--gif-interval", type=float, default=1.0, help="time between GIF frames (0 disables)"
    )
    parser.add_argument(
        "--outdir", type=Path, default=Path("outputs/sweet_parker/collapse_check")
    )
    parser.add_argument(
        "--replot",
        type=Path,
        default=None,
        help="finished run folder (histories.npz, summary.json) to re-assess and re-plot",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.replot is not None:
        replot(args.replot)
        return
    args.outdir.mkdir(parents=True, exist_ok=True)
    grid = CartesianGrid.from_mesh_config(
        MeshConfig(shape=(args.nx, args.ny), upper=(args.lx, args.ly))
    )
    dx, dy = grid.spacing
    lengths = grid.lengths
    viscosity = args.pm * args.eta
    ka = 2.0 * math.pi * args.mode_y * args.width / args.ly
    delta_prime_a = 2.0 * (1.0 / ka - ka)
    sheet_separation = 0.5 * args.lx

    save_every = max(1, math.ceil(args.save_interval / (args.cfl * min(dx, dy))))
    dt = args.save_interval / save_every
    n_saves = max(1, round(args.t_end / args.save_interval))
    print(
        f"[collapse] grid={args.nx}x{args.ny} L=({args.lx:.3f},{args.ly:.3f}) a={args.width} "
        f"ka={ka:.3f} Delta'a={delta_prime_a:.2f} eta={args.eta:g} nu={viscosity:g} "
        f"hold_equilibrium={args.hold_equilibrium} dt={dt:.3e} steps={n_saves * save_every}"
    )

    equilibrium = PeriodicDoubleHarrisEquilibrium(width=args.width, amplitude=1.0)
    state = PeriodicDoubleHarrisEquilibrium(
        width=args.width,
        amplitude=1.0,
        perturbation_amplitude=args.seed,
        perturbation_mode=(0, args.mode_y),
    ).initial_state(grid)
    params = ReducedMHDParams(resistivity=args.eta, viscosity=viscosity)
    # eta*lap(psi - psi_eq) = eta*lap(psi) + eta*j_eq.
    equilibrium_source = args.eta * current_density(
        equilibrium.initial_state(grid).psi, lengths=lengths
    )

    def rhs(current_state: ReducedMHDState) -> ReducedMHDState:
        tendency = reduced_mhd_rhs(current_state, params, lengths=lengths, dealiasing="two_thirds")
        if args.hold_equilibrium:
            tendency = tendency._replace(psi=tendency.psi + equilibrium_source)
        return tendency

    @jax.jit
    def advance(current_state: ReducedMHDState) -> ReducedMHDState:
        trajectory = evolve_rk4(current_state, rhs, dt=dt, steps=save_every, save_every=save_every)
        return jax.tree.map(lambda field: field[-1], trajectory.states)

    sheet_x = 0.25 * args.lx
    snapshot_indices = set(np.linspace(0, n_saves, args.snapshots).round().astype(int).tolist())
    times: list[float] = []
    history: dict[str, list[float]] = {key: [] for key in HISTORY_KEYS}
    snapshot_times: list[float] = []
    snapshots: list[np.ndarray] = []
    gif_stride = round(args.gif_interval / args.save_interval) if args.gif_interval > 0 else 0
    gif_steps = (max(1, args.nx // GIF_MAX_POINTS), max(1, args.ny // GIF_MAX_POINTS))
    gif_frames: list[tuple[float, np.ndarray, np.ndarray]] = []
    wall_start = time.perf_counter()

    for index in range(n_saves + 1):
        if index > 0:
            state = advance(state)
        t = index * args.save_interval
        psi, omega = np.asarray(state.psi), np.asarray(state.omega)
        if not (np.all(np.isfinite(psi)) and np.all(np.isfinite(omega))):
            print(f"[collapse] non-finite fields at t={t:.2f}; stopping")
            break
        times.append(t)
        try:
            values = measure_sweet_parker_sheet(
                psi, omega, lengths=lengths, resistivity=args.eta, sheet_x=sheet_x
            ).as_dict()
        except ValueError as error:
            print(f"[collapse] t={t:.2f}: measurement failed ({error})")
            values = {}
        for key in HISTORY_KEYS:
            history[key].append(float(values.get(key, np.nan)))
        wants_snapshot = index in snapshot_indices
        wants_gif = bool(gif_stride) and index % gif_stride == 0
        if wants_snapshot or wants_gif:
            shift = _centring_shift(history["x_point_y"][-1], args.ny, dy)
            current_field = np.roll(
                np.asarray(current_density(state.psi, lengths=lengths), np.float32), shift, axis=1
            )
            if wants_snapshot:
                snapshot_times.append(t)
                snapshots.append(current_field)
            if wants_gif:
                sx, sy = gif_steps
                gif_frames.append(
                    (
                        t,
                        current_field[::sx, ::sy].astype(np.float16),
                        np.roll(psi, shift, axis=1)[::sx, ::sy].astype(np.float16),
                    )
                )
        if index % 20 == 0:
            print(
                f"[collapse] t={t:7.2f} |j_X|={abs(history['current_x'][-1]):.3f} "
                f"delta={history['delta'][-1]:.4f} L_j={history['length_current'][-1]:.3f} "
                f"E*={history['normalized_rate'][-1]:.3e} "
                f"angle={history['opening_angle_deg'][-1]:.1f}deg "
                f"nX={history['sheet_x_point_count'][-1]:.0f} "
                f"W/sep={history['island_width'][-1] / sheet_separation:.3f} "
                f"wall={time.perf_counter() - wall_start:.0f}s"
            )

    t_arr = np.asarray(times)
    h = {key: np.asarray(values) for key, values in history.items()}
    summary = assess(
        t_arr,
        h,
        resistivity=args.eta,
        viscosity=viscosity,
        sheet_separation=sheet_separation,
    )
    summary.update(
        {
            "grid": [args.nx, args.ny],
            "lengths": list(lengths),
            "width": args.width,
            "ka": ka,
            "delta_prime_a": delta_prime_a,
            "resistivity": args.eta,
            "viscosity": viscosity,
            "seed": args.seed,
            "hold_equilibrium": args.hold_equilibrium,
            "sheet_separation": sheet_separation,
            "dt": dt,
            "t_final": float(t_arr[-1]),
            "wall_seconds": time.perf_counter() - wall_start,
        }
    )

    np.savez(args.outdir / "histories.npz", times=t_arr, **h)
    np.savez_compressed(
        args.outdir / "snapshots.npz",
        times=np.asarray(snapshot_times),
        current=np.asarray(snapshots),
        lengths=np.asarray(lengths),
        centred=np.asarray(True),
    )
    (args.outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    plot_histories(t_arr, h, summary, args.outdir / "histories.png")
    plot_snapshots(
        snapshot_times,
        snapshots,
        lengths,
        sheet_x,
        args.outdir / "current_snapshots.png",
        centred=True,
    )
    write_domain_gif(gif_frames, lengths, args.outdir / "domain.gif")
    print_verdict(summary, args.outdir)


def replot(run_dir: Path) -> None:
    """Re-assess and re-plot a finished run from ``histories.npz`` and ``summary.json``."""
    summary_path = run_dir / "summary.json"
    old = json.loads(summary_path.read_text())
    with np.load(run_dir / "histories.npz") as data:
        t = np.asarray(data["times"])
        h = {
            key: np.asarray(data[key]) if key in data.files else np.full(t.shape, np.nan)
            for key in HISTORY_KEYS
        }
    old_lengths = old.get("lengths", [2.0 * math.pi, 4.0 * math.pi])
    summary = assess(
        t,
        h,
        resistivity=old["resistivity"],
        viscosity=old["viscosity"],
        sheet_separation=old.get("sheet_separation", 0.5 * old_lengths[0]),
    )
    summary.update({key: value for key, value in old.items() if key not in summary})
    summary_path.write_text(json.dumps(summary, indent=2))
    plot_histories(t, h, summary, run_dir / "histories.png")
    snapshots_path = run_dir / "snapshots.npz"
    if snapshots_path.exists():
        with np.load(snapshots_path) as data:
            lengths = tuple(float(value) for value in data["lengths"])
            plot_snapshots(
                list(data["times"]),
                list(data["current"]),
                lengths,
                0.25 * lengths[0],
                run_dir / "current_snapshots.png",
                centred="centred" in data.files,
            )
    print_verdict(summary, run_dir)


def print_verdict(summary: dict, outdir: Path) -> None:
    print("\n[collapse] ---- verdict ----")
    for key in VERDICT_KEYS:
        print(f"  {key:22s}: {summary['checks'].get(key)}")
    print(json.dumps(summary["metrics"], indent=2))
    print(f"[collapse] outputs in {outdir}")


def assess(
    t: np.ndarray,
    h: dict[str, np.ndarray],
    *,
    resistivity: float,
    viscosity: float,
    sheet_separation: float,
) -> dict:
    current = np.abs(h["current_x"])
    delta = h["delta"]
    metrics: dict[str, float | str | list[float] | None] = {}
    checks: dict[str, bool] = {}

    # The equilibrium first diffuses (j_X falls, delta widens) before tearing
    # growth, so collapse is measured from the widest/weakest pre-collapse state.
    i_min = int(np.nanargmin(delta)) if np.any(np.isfinite(delta)) else 0
    i_peak = int(np.nanargmax(current)) if np.any(np.isfinite(current)) else 0
    delta0 = float(np.nanmax(delta[: i_min + 1]))
    current0 = float(np.nanmin(current[: i_peak + 1]))
    metrics["delta_reference"] = delta0
    metrics["current_reference"] = current0
    metrics["delta_min"] = float(delta[i_min])
    metrics["t_delta_min"] = float(t[i_min])
    metrics["thinning_ratio"] = float(delta[i_min] / delta0)
    metrics["current_amplification"] = float(current[i_peak] / current0)
    metrics["aspect_ratio_at_min_delta"] = float(h["length_current"][i_min] / delta[i_min])
    checks["x_point_collapse"] = bool(
        metrics["thinning_ratio"] <= 0.5
        and metrics["current_amplification"] >= 2.0
        and metrics["aspect_ratio_at_min_delta"] >= 5.0
    )

    # Clean collapse phase: strong X-point current, a single X-point on the
    # sheet, and an island narrower than half the sheet separation (NaN width
    # means the separatrix left the search range, i.e. the island is too wide).
    i_weakest = int(np.nanargmin(current[: i_peak + 1]))
    onset = np.flatnonzero(current[i_weakest:] >= 1.5 * current0)
    window = None
    for key in VERDICT_KEYS[1:]:
        checks[key] = False
    if onset.size:
        start = i_weakest + int(onset[0])
        end_reasons = {
            "secondary_x_point": h["sheet_x_point_count"] > 1,
            "island_width": ~(h["island_width"] <= 0.5 * sheet_separation),
            "current_drop": current < 1.5 * current0,
        }
        stop, reason = t.size, "end_of_run"
        for name, flags in end_reasons.items():
            hits = np.flatnonzero(flags[start:])
            if hits.size and start + int(hits[0]) < stop:
                stop, reason = start + int(hits[0]), name
        metrics["t_collapse_onset"] = float(t[start])
        metrics["t_phase_end"] = float(t[min(stop, t.size - 1)])
        metrics["phase_end_reason"] = reason
        phase = slice(start, stop)
        transit = np.nanmedian(h["length"][phase] / h["b_upstream"][phase]) if stop > start else 0
        metrics["alfven_transit"] = float(transit)
        if np.isfinite(transit) and transit > 0.0:
            window = select_steady_window(
                t[phase], h["reconnection_rate"][phase], min_duration=3.0 * transit
            )
    if window is not None:
        in_window = (t >= window[0]) & (t <= window[1])
        med = {key: float(np.nanmedian(values[in_window])) for key, values in h.items()}
        checks["steady_window_found"] = True
        checks["resolved"] = bool(np.nanmin(h["delta_over_dx"][in_window]) >= 8.0)
        theory = sweet_parker_prediction(
            resistivity=resistivity,
            viscosity=viscosity,
            length=med["length"],
            b_upstream=med["b_upstream"],
        )
        outflow_over_alfven = med["v_outflow"] / med["b_upstream"]
        aspect_ratio = med["length"] / med["delta"]
        checks["alfvenic_outflow"] = bool(outflow_over_alfven >= MIN_OUTFLOW_OVER_ALFVEN)
        checks["elongated_sheet"] = bool(aspect_ratio >= MIN_WINDOW_ASPECT_RATIO)
        checks["high_lundquist"] = bool(theory["lundquist"] >= MIN_WINDOW_LUNDQUIST)
        metrics.update(
            {
                "window": [window[0], window[1]],
                "window_rate_cv": window[2],
                "lundquist_window": theory["lundquist"],
                "normalized_rate_window": med["normalized_rate"],
                "ohm_ratio": med["reconnection_rate"] / (med["v_inflow"] * med["b_upstream"]),
                "mass_ratio": (med["v_inflow"] * med["length"])
                / (med["v_outflow"] * med["delta"]),
                "outflow_over_alfven": outflow_over_alfven,
                "aspect_ratio_window": aspect_ratio,
                "delta_coefficient": med["delta"] / theory["delta"],
                "rate_coefficient": med["normalized_rate"] / theory["normalized_rate"],
                "length_ratio_outflow_over_current": med["length"] / med["length_current"],
                "tilt_deg_window": med["tilt_deg"],
                "island_width_over_separation_max": float(
                    np.nanmax(h["island_width"][in_window]) / sheet_separation
                ),
                "delta_over_dx_window_min": float(np.nanmin(h["delta_over_dx"][in_window])),
            }
        )
    return {"checks": checks, "metrics": metrics}


def plot_histories(t: np.ndarray, h: dict[str, np.ndarray], summary: dict, path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
    panels = [
        ("|j_X|", [(np.abs(h["current_x"]), "|j_X|")], "linear"),
        (
            "sheet half-width / half-length / island width",
            [
                (h["delta"], "delta"),
                (h["length_current"], "L_j"),
                (h["length"], "L (outflow)"),
                (h["island_width"], "W (island)"),
            ],
            "log",
        ),
        ("normalized rate E* = eta|j_X|/B_up^2", [(h["normalized_rate"], "E*")], "log"),
        (
            "flows",
            [(h["v_inflow"], "v_in"), (h["v_outflow"], "v_out"), (h["b_upstream"], "B_up")],
            "linear",
        ),
        (
            "X-point opening angle [deg]",
            [(h["opening_angle_deg"], "angle"), (h["tilt_deg"], "tilt")],
            "linear",
        ),
        (
            "X-points on sheet / delta/dx",
            [(h["sheet_x_point_count"], "n_X"), (h["delta_over_dx"], "delta/dx")],
            "log",
        ),
    ]
    metrics = summary["metrics"]
    window = metrics.get("window")
    for axis, (title, series, scale) in zip(axes.flat, panels, strict=True):
        for values, label in series:
            axis.plot(t, values, label=label)
        axis.set_title(title)
        axis.set_yscale(scale)
        axis.set_xlabel("t")
        if "t_collapse_onset" in metrics:
            axis.axvspan(
                metrics["t_collapse_onset"], metrics["t_phase_end"], color="tab:orange", alpha=0.1
            )
        if window:
            axis.axvspan(*window, color="tab:green", alpha=0.2)
        if len(series) > 1:
            axis.legend(fontsize=8)
    checks = summary["checks"]
    fig.suptitle(
        "  ".join(f"{key}={checks.get(key)}" for key in VERDICT_KEYS)
        + f"\nphase end: {metrics.get('phase_end_reason', 'n/a')};"
        " orange = collapse phase, green = steady window"
    )
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_snapshots(
    times: list[float],
    snapshots: list[np.ndarray],
    lengths: tuple[float, float],
    sheet_x: float,
    path: Path,
    *,
    centred: bool,
) -> None:
    if not snapshots:
        return
    nx = snapshots[0].shape[0]
    dx = lengths[0] / nx
    half = max(1, int(0.125 * lengths[0] / dx))
    centre = int(sheet_x / dx)
    rows = slice(max(0, centre - half), centre + half)
    extent = (0.0, lengths[1], max(0, centre - half) * dx, (centre + half) * dx)
    fig, axes = plt.subplots(len(snapshots), 1, figsize=(10, 1.8 * len(snapshots)), sharex=True)
    for axis, t, field in zip(np.atleast_1d(axes), times, snapshots, strict=True):
        crop = field[rows]
        limit = max(float(np.max(np.abs(crop))), 1.0e-12)
        axis.imshow(
            crop,
            origin="lower",
            aspect="auto",
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
            extent=extent,
        )
        axis.set_ylabel(f"t={t:.1f}\nx", fontsize=8)
    np.atleast_1d(axes)[-1].set_xlabel("y - y_X + Ly/2" if centred else "y")
    fig.suptitle("j_z near the left sheet (x ~ Lx/4)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def write_domain_gif(
    frames: list[tuple[float, np.ndarray, np.ndarray]],
    lengths: tuple[float, float],
    path: Path,
) -> None:
    """Write full-domain ``j_z`` frames (per-frame colour scale) with ψ contours."""
    if not frames:
        return
    import imageio.v2 as imageio

    nx, ny = frames[0][1].shape
    x = (np.arange(nx) + 0.5) * lengths[0] / nx
    y = (np.arange(ny) + 0.5) * lengths[1] / ny
    height = float(np.clip(8.0 * lengths[0] / lengths[1] * 0.6, 4.0, 12.0))
    images = []
    for t, current, psi in frames:
        current = current.astype(np.float32)
        fig, axis = plt.subplots(figsize=(8.0, height), constrained_layout=True)
        limit = max(float(np.max(np.abs(current))), 1.0e-12)
        image = axis.imshow(
            current,
            origin="lower",
            aspect="auto",
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
            extent=(0.0, lengths[1], 0.0, lengths[0]),
        )
        axis.contour(y, x, psi.astype(np.float32), levels=32, colors="k", linewidths=0.4)
        axis.set_xlabel("y - y_X + Ly/2")
        axis.set_ylabel("x")
        axis.set_title(f"j_z with flux contours, t = {t:.1f}  (max |j_z| = {limit:.2f})")
        fig.colorbar(image, ax=axis, shrink=0.85)
        fig.canvas.draw()
        images.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        plt.close(fig)
    imageio.mimsave(path, images, duration=0.08)


def _centring_shift(x_point_y: float, ny: int, dy: float) -> int:
    """Return the ``y`` roll that moves the X-point to ``y = Ly/2``."""
    if not np.isfinite(x_point_y):
        return 0
    return ny // 2 - round(x_point_y / dy)


if __name__ == "__main__":
    main()
