r"""Track energies, a spectrum snapshot, and the isometric current cube of the 3-D Orszag--Tang run.

Reads trajectory.npz written by examples/gallery/09_orszag_tang_3d.py (run that
first) and streams it frame by frame, so large runs fit in memory. It writes, to
<npz folder>/track/ unless OUTDIR is set: energy.png (dE_B/W0 and dE_K/W0 with
W0 = E_K(0) + E_M(0)), spectrum.png (shell spectra of the magnetic and kinetic
energy at the snapshot time T_SNAP, a power-law fit, the compensated k^(5/3)
E_B(k) and the fitted slope of every frame, all stopping at k = N/3), cube_final.png
and cube.gif (|J| or |B| on the three periodic faces at x=0, y=0, z=0 in an
isometric view), and history.csv and spectrum.csv.

The NPZ stores neither t=0 nor the Orszag--Tang beta, so the script rebuilds the
initial fields analytically (beta from BETA) and prepends them as the t=0 frame.
Runtime is seconds to a minute for the gallery run (64^3, 21 frames); lower
GIF_FRAMES to speed up the cube animation. Run from the repository root:

    python examples/orszag_tang/track_orszag_tang_3d.py
"""

import ast
import csv
import zipfile
from collections.abc import Iterator
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

# --- Input parameters ------------------------------------------------------
NPZ = Path("outputs/gallery/orszag_tang_3d/trajectory.npz")  # trajectory to analyse
OUTDIR = None  # output folder; None means <npz folder>/track
FIELD = "J"  # field on the cube faces: "J" or "B"
BETA = 0.8  # Orszag--Tang beta (not stored in the NPZ)
T_SNAP = 2.0  # time of the spectrum snapshot (nearest saved state)
KMIN = 3  # lower edge of the fit band
KMAX = None  # upper edge of the fit band; None or anything above N/3 is capped at N/3
GIF_FRAMES = 40  # frames of the cube animation

# --- Helpers ---------------------------------------------------------------


def open_frames(path: Path, member: str) -> tuple[tuple[int, ...], Iterator[np.ndarray]]:
    """Return one NPZ array's shape and an iterator over its leading-axis frames."""
    archive = zipfile.ZipFile(path)
    stream = archive.open(f"{member}.npy")
    version = np.lib.format.read_magic(stream)
    if version == (1, 0):
        shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(stream)
    else:
        shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(stream)
    if fortran_order:
        raise ValueError(f"{member} is stored in Fortran order")
    frame_shape = shape[1:]
    frame_bytes = int(np.prod(frame_shape)) * np.dtype(dtype).itemsize

    def frames() -> Iterator[np.ndarray]:
        with archive, stream:
            for _ in range(shape[0]):
                buffer = bytearray()
                while len(buffer) < frame_bytes:
                    chunk = stream.read(frame_bytes - len(buffer))
                    if not chunk:
                        raise EOFError(f"{member} ended early")
                    buffer.extend(chunk)
                yield np.frombuffer(bytes(buffer), dtype=dtype).reshape(frame_shape)

    return shape, frames()


def orszag_tang_fields(n: int, beta: float) -> tuple[np.ndarray, np.ndarray]:
    """Same fields as OrszagTang3DEquilibrium.initial_fields, in NumPy."""
    axis = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    zero = np.zeros_like(x)
    velocity = np.stack((-2.0 * np.sin(y), 2.0 * np.sin(x), zero))
    magnetic = beta * np.stack(
        (
            -2.0 * np.sin(2.0 * y) + np.sin(z),
            2.0 * np.sin(x) + np.sin(z),
            np.sin(x) + np.sin(y),
        )
    )
    return velocity, magnetic


def analyze(
    velocity: np.ndarray,
    magnetic: np.ndarray,
    *,
    kvec: np.ndarray,
    shell: np.ndarray,
    nshell: int,
    face_field: str | None = None,
) -> dict:
    """Energies, shell spectra, and optional cube-face values for one state."""
    norm = float(velocity.shape[-1]) ** 6  # (N^3)^2, the Parseval normalization
    vh = np.fft.fftn(velocity, axes=(1, 2, 3))
    bh = np.fft.fftn(magnetic, axes=(1, 2, 3))

    kinetic = 0.5 * np.sum(np.abs(vh) ** 2) / norm
    magnetic_energy = 0.5 * np.sum(np.abs(bh) ** 2) / norm
    kinetic_per_mode = 0.5 * np.sum(np.abs(vh) ** 2, axis=0).ravel() / norm
    magnetic_per_mode = 0.5 * np.sum(np.abs(bh) ** 2, axis=0).ravel() / norm

    faces = None
    if face_field == "J":
        current_h = 1j * np.cross(kvec, bh, axis=0)
        current = np.fft.ifftn(current_h, axes=(1, 2, 3)).real
        field = np.sqrt(np.sum(current**2, axis=0))
    elif face_field == "B":
        field = np.sqrt(np.sum(magnetic**2, axis=0))
    if face_field is not None:
        faces = (field[0, :, :], field[:, 0, :], field[:, :, 0])

    return {
        "kinetic": float(kinetic),
        "magnetic": float(magnetic_energy),
        "spectrum_k": np.bincount(shell, weights=kinetic_per_mode, minlength=nshell),
        "spectrum_b": np.bincount(shell, weights=magnetic_per_mode, minlength=nshell),
        "faces": faces,
    }


# --- Run -------------------------------------------------------------------
if FIELD not in ("J", "B"):
    raise ValueError("FIELD must be 'J' or 'B'")
npz = NPZ.resolve()
outdir = OUTDIR if OUTDIR is not None else npz.parent / "track"
outdir.mkdir(parents=True, exist_ok=True)
print(f"Tracking {npz}, writing to {outdir}", flush=True)

with np.load(npz) as data:
    times_file = np.asarray(data["times"], dtype=float)
    config = ast.literal_eval(data["config"].item())
if config.get("equilibrium", "orszag_tang_3d") != "orszag_tang_3d":
    raise ValueError(f"NPZ was run from {config.get('equilibrium')!r}, not orszag_tang_3d")

velocity_shape, velocity_frames = open_frames(npz, "velocity")
magnetic_shape, magnetic_frames = open_frames(npz, "magnetic")
if len(magnetic_shape) != 5 or magnetic_shape[1] != 3 or velocity_shape != magnetic_shape:
    raise ValueError(f"expected (frames, 3, N, N, N) arrays, got {magnetic_shape}")
n = magnetic_shape[-1]

prepend = bool(times_file[0] > 0.0)
if not prepend:
    print("NPZ already starts at t=0; using its first frame as the initial state")
times = np.concatenate(([0.0], times_file)) if prepend else times_file
total = len(times)
if total != magnetic_shape[0] + int(prepend):
    raise ValueError("times and frame counts disagree")

# Wavevectors on the L = 2 pi box are integers, and shells are |k| rounded.
k1 = np.fft.fftfreq(n, d=1.0 / n)
kvec = np.stack(np.meshgrid(k1, k1, k1, indexing="ij"))
shell = np.rint(np.sqrt(np.sum(kvec**2, axis=0))).astype(int).ravel()
nshell = int(shell.max()) + 1

def all_frames():
    if prepend:
        yield orszag_tang_fields(n, BETA)
    yield from zip(velocity_frames, magnetic_frames, strict=True)

gif_set = set(
    np.unique(np.linspace(0, total - 1, min(total, GIF_FRAMES), dtype=int)).tolist()
)
kinetic, magnetic_energy, spectrum_k, spectrum_b, faces = [], [], [], [], {}
print(f"{total} states, N={n}")
for index, (velocity, magnetic) in enumerate(all_frames()):
    result = analyze(
        velocity, magnetic, kvec=kvec, shell=shell, nshell=nshell,
        face_field=FIELD if index in gif_set else None,
    )
    kinetic.append(result["kinetic"])
    magnetic_energy.append(result["magnetic"])
    spectrum_k.append(result["spectrum_k"])
    spectrum_b.append(result["spectrum_b"])
    if result["faces"] is not None:
        faces[index] = result["faces"]
    print(f"  state {index + 1}/{total}  t={times[index]:.2f}", flush=True)

kinetic = np.asarray(kinetic)
magnetic_energy = np.asarray(magnetic_energy)
spectrum_k = np.asarray(spectrum_k)
spectrum_b = np.asarray(spectrum_b)
if prepend:
    print(
        f"initial E_K = {kinetic[0]:.6f} (expect 2), "
        f"E_M = {magnetic_energy[0]:.6f} (expect 1.92 at beta 0.8)"
    )

# Energy, normalized as in vencels_kinetic_comparison.py.
w0 = kinetic[0] + magnetic_energy[0]
d_magnetic = (magnetic_energy - magnetic_energy[0]) / w0
d_kinetic = (kinetic - kinetic[0]) / w0

# Spectra: one snapshot fit, plus the slope of every frame.
k = np.arange(nshell)
kcut = n // 3
kmax = min(KMAX or kcut, kcut)
lo = max(1, KMIN)
band = (k >= lo) & (k <= kmax)
slope_t = np.full(total, np.nan)
for i in range(total):
    mask = band & (spectrum_b[i] > 0.0)
    if mask.sum() >= 2:
        slope_t[i] = np.polyfit(np.log10(k[mask]), np.log10(spectrum_b[i][mask]), 1)[0]

snap = int(np.argmin(np.abs(times - T_SNAP)))
t_snap = float(times[snap])
if abs(t_snap - T_SNAP) > 1e-6:
    print(f"note: no saved state exactly at t={T_SNAP}; using t={t_snap:.3f}")
e_b = spectrum_b[snap]
e_k = spectrum_k[snap]
fit = band & (e_b > 0.0)
if fit.sum() < 2:
    raise ValueError("fit band has fewer than two points")
slope, intercept = np.polyfit(np.log10(k[fit]), np.log10(e_b[fit]), 1)
compensated = k ** (5.0 / 3.0) * e_b
plateau = float(np.mean(compensated[fit]))
nrmse = float(np.sqrt(np.mean((compensated[fit] - plateau) ** 2)) / plateau)

# Energy figure.
fig, ax = plt.subplots(figsize=(7.5, 5.0), constrained_layout=True)
ax.plot(times, d_magnetic, color="navy", lw=2, label=r"$\delta E_B/W_0$ (magnetic)")
ax.plot(times, d_kinetic, color="crimson", lw=2, label=r"$\delta E_K/W_0$ (kinetic)")
ax.axhline(0.0, color="0.6", lw=1)
ax.set_title("Energy partitioning, 3-D Orszag--Tang")
ax.set_xlabel("t")
ax.set_ylabel(r"$\delta W / W_0$")
ax.legend(frameon=False)
fig.savefig(outdir / "energy.png", dpi=180)
plt.close(fig)

# Spectrum figure.
kk = k[1 : kmax + 1]
fig, (ax_s, ax_c, ax_t) = plt.subplots(1, 3, figsize=(16.0, 4.8), constrained_layout=True)
ax_s.loglog(kk, e_b[1 : kmax + 1], color="teal", lw=2, label=r"$E_B(k)$")
ax_s.loglog(kk, e_k[1 : kmax + 1], color="firebrick", lw=1.6, label=r"$E_K(k)$")
fit_k = np.geomspace(lo, kmax, 100)
ax_s.loglog(
    fit_k, 10.0**intercept * fit_k**slope, "k--", lw=1.5,
    label=rf"fit $E_B \propto k^{{{slope:.2f}}}$ on $[{lo}, {kmax}]$",
)
ax_s.axvline(kcut, color="0.4", ls=":", lw=1.2, label=r"2/3 cutoff, $k=N/3$")
ax_s.set_xlim(1, kcut)
ax_s.set_xlabel(r"wavenumber $k$")
ax_s.set_ylabel(r"energy spectrum $E(k)$")
ax_s.set_title(f"Shell spectra at t = {t_snap:.2f}")
ax_s.legend(frameon=False, fontsize="small")
ax_c.semilogx(kk, compensated[1 : kmax + 1], color="#d95f02", lw=2, label=r"$k^{5/3}E_B(k)$")
ax_c.hlines(plateau, lo, kmax, colors="k", linestyles="--",
            label=f"constant fit, NRMSE {nrmse:.1%}")
for boundary in (lo, kmax):
    ax_c.axvline(boundary, color="0.5", ls=":", lw=1.0)
ax_c.axvline(kcut, color="0.4", ls=":", lw=1.2)
ax_c.set_xlim(1, kcut)
ax_c.set_xlabel(r"wavenumber $k$")
ax_c.set_ylabel(r"$k^{5/3}E_B(k)$")
ax_c.set_title("Compensated spectrum")
ax_c.legend(frameon=False, fontsize="small")
tracked = np.isfinite(slope_t)
ax_t.plot(times[tracked], slope_t[tracked], "o-", color="teal", ms=3, label="fitted slope")
ax_t.axvline(t_snap, color="0.5", ls=":", lw=1.0, label="snapshot")
ax_t.set_xlabel("t")
ax_t.set_ylabel(r"slope of $E_B(k)$ on $[k_{\min}, N/3]$")
ax_t.set_title("Spectral slope tracked in time")
ax_t.legend(frameon=False, fontsize="small")
fig.savefig(outdir / "spectrum.png", dpi=180)
plt.close(fig)

with (outdir / "history.csv").open("w", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow([
        "t", "E_K", "E_M", "dE_B_over_W0", "dE_K_over_W0", "slope_E_B",
    ])
    rows = zip(
        times, kinetic, magnetic_energy, d_magnetic, d_kinetic, slope_t, strict=False
    )
    for row in rows:
        writer.writerow(row)
with (outdir / "spectrum.csv").open("w", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(["k", "E_B", "E_K", "k53_E_B"])
    rows = zip(
        kk, e_b[1 : kmax + 1], e_k[1 : kmax + 1], compensated[1 : kmax + 1], strict=False
    )
    for row in rows:
        writer.writerow(row)

print(f"W0 = {w0:.6f}")
print(f"final: dE_B/W0 = {d_magnetic[-1]:+.4f}, dE_K/W0 = {d_kinetic[-1]:+.4f}")
print(f"snapshot t = {t_snap:.2f}: fit E_B ~ k^{slope:.3f} on k in [{lo}, {kmax}]")
print(f"compensated plateau = {plateau:.4g}, NRMSE = {nrmse:.1%}")

# Isometric cube of the three faces x=0, y=0, z=0.
if faces:
    keys = sorted(faces)
    values = np.concatenate([np.concatenate([f.ravel() for f in faces[key]]) for key in keys])
    vmax = float(np.percentile(values, 99.5))
    vmax = vmax if vmax > 0.0 else 1.0
    norm = Normalize(vmin=0.0, vmax=vmax)
    cmap = plt.get_cmap("magma")
    label = "|J|" if FIELD == "J" else "|B|"
    coords = np.linspace(0.0, 1.0, n)
    grid_a, grid_b = np.meshgrid(coords, coords, indexing="ij")
    zeros = np.zeros_like(grid_a)
    stride = max(1, n // 96)

    def draw_cube(axis, key: int) -> None:
        axis.clear()
        x_face, y_face, z_face = faces[key]
        surfaces = (
            (zeros, grid_a, grid_b, x_face),  # x = 0 face, indexed (y, z)
            (grid_a, zeros, grid_b, y_face),  # y = 0 face, indexed (x, z)
            (grid_a, grid_b, zeros, z_face),  # z = 0 face, indexed (x, y)
        )
        for xs, ys, zs, field in surfaces:
            axis.plot_surface(
                xs, ys, zs,
                facecolors=cmap(norm(field)),
                rstride=stride, cstride=stride,
                shade=False, linewidth=0, antialiased=False,
            )
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(0.0, 1.0)
        axis.set_zlim(0.0, 1.0)
        axis.set_box_aspect((1.0, 1.0, 1.0))
        axis.view_init(elev=-35.26, azim=225.0)  # isometric view of the origin corner
        axis.set_xlabel("x / L")
        axis.set_ylabel("y / L")
        axis.set_zlabel("z / L")
        axis.set_title(f"{label} on the periodic faces, t = {times[key]:.2f}")

    figure = plt.figure(figsize=(7.5, 7.0))
    axis = figure.add_subplot(projection="3d")
    figure.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=axis, shrink=0.7, label=label)
    draw_cube(axis, keys[-1])
    figure.savefig(outdir / "cube_final.png", dpi=150)
    animation = FuncAnimation(
        figure, lambda i: draw_cube(axis, keys[i]), frames=len(keys), interval=120,
    )
    animation.save(outdir / "cube.gif", writer=PillowWriter(fps=8), dpi=90)
    plt.close(figure)

for written in sorted(outdir.iterdir()):
    print(f"wrote {written}")
