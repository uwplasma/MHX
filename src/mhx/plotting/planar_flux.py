"""Plots of accepted tracked points and continuous paired flux/rates."""

from pathlib import Path

import numpy as np


def plot_tracked_points(axis, arrays, time, *, color="black", lengths=None):
    """Overlay accepted identities at their physical coordinates for one saved frame."""
    frame = np.isclose(arrays["point_time"], time, rtol=1e-12, atol=1e-12)
    frame &= arrays["point_accepted"]
    ylow, yhigh = axis.get_ylim()
    label_boxes = []
    pixels_per_point = axis.figure.dpi / 72
    for kind, marker in (("X", "x"), ("O", "o")):
        selected = frame & (arrays["point_kind"] == kind)
        positions = arrays["point_position"][selected].copy()
        if not len(positions):
            continue
        if lengths is not None and axis.images:
            extent = axis.images[0].get_extent()
            origin = np.asarray([extent[0], extent[2]])
            positions = (positions - origin) % lengths + origin
        style = {"color": color} if kind == "X" else {"facecolors": "none", "edgecolors": color}
        axis.scatter(positions[:, 0], positions[:, 1], marker=marker, s=28, label=kind, **style)
        for position, track_id in zip(positions, arrays["point_track_id"][selected], strict=True):
            high = position[1] > ylow + 0.9 * (yhigh - ylow)
            x, y = axis.transData.transform(position)
            width, height = (4.5 * len(str(track_id)) + 3) * pixels_per_point, 10 * pixels_per_point
            offsets = ((4, -5 if high else 5), (4, 5 if high else -5), (-4, 5), (-4, -5))
            for ox, oy in offsets:
                left = x + ox * pixels_per_point - (width if ox < 0 else 0)
                bottom = y + oy * pixels_per_point - (height if oy < 0 else 0)
                box = (left, bottom, left + width, bottom + height)
                if not any(
                    box[0] < other[2]
                    and box[2] > other[0]
                    and box[1] < other[3]
                    and box[3] > other[1]
                    for other in label_boxes
                ):
                    label_boxes.append(box)
                    break
            else:
                continue  # The marker and its archived identity remain available.
            axis.annotate(
                str(track_id),
                position,
                xytext=(ox, oy),
                textcoords="offset points",
                fontsize=7,
                color=color,
                va="top" if oy < 0 else "bottom",
                ha="right" if ox < 0 else "left",
                clip_on=True,
            )


def plot_planar_flux(result, path, *, psi_final=None, lengths, lower=(0.0, 0.0)):
    import matplotlib.pyplot as plt

    arrays = result.arrays
    has_pairs = len(arrays["pair_names"]) > 0
    figure, axes = plt.subplots(
        1,
        3 if has_pairs else 1,
        figsize=(16, 4.5) if has_pairs else (7, 6),
        constrained_layout=True,
        squeeze=False,
    )
    axes = axes[0]
    if psi_final is not None:
        psi_final = np.asarray(psi_final)
        dx, dy = np.asarray(lengths) / np.asarray(psi_final.shape)
        image = axes[0].imshow(
            psi_final.T,
            origin="lower",
            aspect="equal",
            cmap="RdBu_r",
            extent=(
                lower[0] - dx / 2,
                lower[0] + lengths[0] - dx / 2,
                lower[1] - dy / 2,
                lower[1] + lengths[1] - dy / 2,
            ),
        )
        figure.colorbar(image, ax=axes[0], label=r"$\psi$")
        if np.ptp(psi_final) > 0:
            x = lower[0] + np.arange(psi_final.shape[0]) * dx
            y = lower[1] + np.arange(psi_final.shape[1]) * dy
            axes[0].contour(
                x, y, psi_final.T, levels=48, colors="black", linewidths=0.4, alpha=0.55
            )
    axes[0].set(
        xlabel="x",
        ylabel="y",
        title=f"Accepted X/O points and IDs, t = {arrays['time'][-1]:.2f}",
    )
    figure.canvas.draw()
    plot_tracked_points(axes[0], arrays, arrays["time"][-1], lengths=lengths)
    if axes[0].get_legend_handles_labels()[0]:
        axes[0].legend(facecolor="white", framealpha=0.9)
    for j, name in enumerate(arrays["pair_names"]):
        axes[1].plot(arrays["time"], arrays["paired_flux"][:, j], label=name)
        axes[2].plot(arrays["time"], arrays["reconnection_rate"][:, j], label=f"{name}: d flux/dt")
        axes[2].plot(
            arrays["time"],
            arrays["electric_field_rate"][:, j],
            linestyle="--",
            label=f"{name}: E(X)-E(O)",
        )
    if has_pairs:
        for axis, title, ylabel in (
            (axes[1], "Signed paired flux; gaps are invalid", "psi(O)-psi(X)"),
            (axes[2], "Continuous rates; endpoints invalid", "flux / time"),
        ):
            axis.set(xlabel="time (code units)", ylabel=ylabel, title=title)
            axis.grid(alpha=0.25)
            if axis.get_legend_handles_labels()[0]:
                axis.legend(frameon=False, fontsize=8)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output
