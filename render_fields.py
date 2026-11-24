"""Poster-quality flow pictures from a side-view run's snapshots.npz.

    python render_fields.py runs/an1_p_amb101325 --out poster/ [--frames 4] [--movie]

Writes  <out>/stills.png   (speed / vorticity / pressure at a few phases)
        <out>/hero.png     (one wide speed+vorticity frame)
        <out>/cycle.mp4    (one reed cycle, if --movie and ffmpeg is available)
The non-uniform finite-volume fields are resampled onto a uniform 25 µm
raster (linear interpolation between cell centres); solids and the tongue are
drawn as crisp polygons on top.
"""
import argparse
import os
import shutil
import subprocess
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Polygon
from scipy.interpolate import RegularGridInterpolator

BG = "#0d1420"        # field background (still air reads as this)
SOLID = "#4a5566"     # aluminium plate / chamber walls
STEEL = "#d9dee6"     # the tongue
INK = "#c9d2de"

SPEED = LinearSegmentedColormap.from_list(
    "speed", ["#0d1420", "#1b2a4a", "#334d8c", "#5b6fd0", "#b46bd1", "#f08a6b", "#ffd27a", "#fff7e0"])
VORT = LinearSegmentedColormap.from_list(
    "vort", ["#7fd4ff", "#2b7bd6", "#173764", "#0d1420", "#5a1f3f", "#d6457a", "#ffb0c4"])
PRES = LinearSegmentedColormap.from_list(
    "pres", ["#5fc6ff", "#2a6fb3", "#162a46", "#0d1420", "#4a2a12", "#c27a2c", "#ffd58a"])


def load(run):
    s = np.load(os.path.join(run, "snapshots.npz"))
    d = {k: s[k] for k in s.files}
    d["xc"] = 0.5 * (d["zf"][1:] + d["zf"][:-1])
    d["yc"] = 0.5 * (d["yf"][1:] + d["yf"][:-1])
    return d


def vorticity(d, k):
    u, v = d["u"][k].astype(np.float64), d["v"][k].astype(np.float64)
    dvdx = np.gradient(v, d["xc"], axis=0)
    dudy = np.gradient(u, d["yc"], axis=1)
    return dvdx - dudy


def raster(d, q, k, box, res):
    """resample cell-centred q (nx, ny) to a uniform grid inside box; NaN in solids/tongue."""
    x0, x1, y0, y1 = box
    X = np.arange(x0, x1, res) + 0.5 * res
    Y = np.arange(y0, y1, res) + 0.5 * res
    bad = d["solid"] | (d["alpha"][k] < 0.5)
    qq = np.where(bad, np.nan, q)
    f = RegularGridInterpolator((d["xc"], d["yc"]), qq, bounds_error=False, fill_value=np.nan)
    m = RegularGridInterpolator((d["xc"], d["yc"]), bad.astype(float), bounds_error=False, fill_value=1.0)
    XX, YY = np.meshgrid(X, Y, indexing="ij")
    pts = np.stack([XX.ravel(), YY.ravel()], -1)
    out = f(pts).reshape(XX.shape)
    # near walls linear interpolation sees a NaN neighbour: fall back to nearest fluid value
    fn = RegularGridInterpolator((d["xc"], d["yc"]), np.where(bad, 0.0, q), method="nearest",
                                 bounds_error=False, fill_value=0.0)
    hole = np.isnan(out) & (m(pts).reshape(XX.shape) < 0.5)
    out[hole] = fn(pts).reshape(XX.shape)[hole]
    return out.T  # (ny, nx) for imshow


def solid_rects(d):
    """merge solid cells into row runs -> list of rectangles (x0, x1, y0, y1)."""
    zf, yf, sol = d["zf"], d["yf"], d["solid"]
    rects = []
    for j in range(sol.shape[1]):
        col = sol[:, j]
        i = 0
        while i < len(col):
            if col[i]:
                i0 = i
                while i < len(col) and col[i]:
                    i += 1
                rects.append((zf[i0], zf[i], yf[j], yf[j + 1]))
            else:
                i += 1
    return rects


def draw_geometry(ax, d, k, rects, s=1e3):
    for x0, x1, y0, y1 in rects:
        ax.add_patch(plt.Rectangle((x0 * s, y0 * s), (x1 - x0) * s, (y1 - y0) * s, fc=SOLID, ec="none", lw=0))
    xc, L, h = d["xc"], float(d["L"]), float(d["h"])
    m = (xc >= 0) & (xc <= L)
    x = np.concatenate([[0.0], xc[m], [L]])
    yb = np.interp(x, xc[m], d["Y"][k][m])
    yb[0] = 0.0
    poly = np.concatenate([np.stack([x, yb + h], 1), np.stack([x[::-1], yb[::-1]], 1)])
    ax.add_patch(Polygon(poly * s, closed=True, fc=STEEL, ec="#ffffff", lw=0.4, zorder=5))


def style(ax, box, labels=True):
    x0, x1, y0, y1 = (v * 1e3 for v in box)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_facecolor(SOLID)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(colors=INK, labelsize=7, length=2)
    if not labels:
        ax.set_xticks([])
        ax.set_yticks([])


def field_layers(d, k, box, res):
    sp = np.hypot(d["u"][k], d["v"][k]).astype(np.float64)
    return dict(speed=raster(d, sp, k, box, res), vort=raster(d, vorticity(d, k), k, box, res),
                pres=raster(d, d["p"][k].astype(np.float64), k, box, res))


def limits(d, idx):
    fl = ~d["solid"]
    sp = np.hypot(d["u"][idx], d["v"][idx])[:, fl]
    vmax = float(np.percentile(sp, 99.7))
    w = np.concatenate([np.abs(vorticity(d, k)[fl]) for k in idx])
    wmax = float(np.percentile(w, 99.0))
    pmax = float(np.percentile(np.abs(d["p"][idx][:, fl]), 99.5))
    return vmax, wmax, pmax


CHAMBER = (-4.5e-3, 34.0e-3, -36.0e-3, 4.0e-3)
TIP = (15.0e-3, 27.5e-3, -5.0e-3, 2.0e-3)


def stills(d, out, n=4, res=40e-6):
    nt = len(d["t"])
    idx = (np.arange(n) * nt // n).astype(int)
    vmax, wmax, pmax = limits(d, np.arange(nt))
    rects = solid_rects(d)
    T = d["t"][-1] - d["t"][0]
    T = T * nt / (nt - 1)
    fig, ax = plt.subplots(3, n, figsize=(3.0 * n, 8.6), facecolor=BG)
    fig.subplots_adjust(hspace=0.04, wspace=0.03)
    rows = [("speed", SPEED, dict(vmin=0, vmax=vmax), "|u|  [m/s]"),
            ("vort", VORT, dict(norm=TwoSlopeNorm(0, -wmax, wmax)), "vorticity  [1/s]"),
            ("pres", PRES, dict(norm=TwoSlopeNorm(0, -pmax, pmax)), "p − p_amb  [Pa]")]
    ims = [None] * 3
    for c, k in enumerate(idx):
        F = field_layers(d, k, CHAMBER, res)
        for r, (key, cmap, kw, _) in enumerate(rows):
            a = ax[r, c]
            ims[r] = a.imshow(np.ma.masked_invalid(F[key]), origin="lower", cmap=cmap, interpolation="bilinear",
                              extent=[v * 1e3 for v in CHAMBER], **kw)
            draw_geometry(a, d, k, rects)
            style(a, CHAMBER, labels=False)
        ph = (d["t"][k] - d["t"][idx[0]]) / T
        ax[0, c].set_title(f"phase {ph:.2f}", color=INK, fontsize=10)
    for r, (_, _, _, lab) in enumerate(rows):
        cb = fig.colorbar(ims[r], ax=ax[r, :], shrink=0.85, pad=0.01, aspect=18)
        cb.set_label(lab, color=INK, fontsize=9)
        cb.ax.tick_params(colors=INK, labelsize=7)
        cb.outline.set_visible(False)
    fig.savefig(os.path.join(out, "stills.png"), dpi=150, facecolor=BG, bbox_inches="tight")
    plt.close(fig)


def frame(d, k, vmax, wmax, rects, path, res=30e-6, size=(12.8, 7.2), trace=None):
    fig = plt.figure(figsize=size, facecolor=BG)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.25], height_ratios=[1, 1], left=0.03, right=0.985,
                          top=0.93, bottom=0.07, wspace=0.08, hspace=0.18)
    a0 = fig.add_subplot(gs[:, 0])
    a1 = fig.add_subplot(gs[0, 1])
    a2 = fig.add_subplot(gs[1, 1])
    F = field_layers(d, k, CHAMBER, res)
    a0.imshow(np.ma.masked_invalid(F["speed"]), origin="lower", cmap=SPEED, vmin=0, vmax=vmax,
              interpolation="bilinear", extent=[v * 1e3 for v in CHAMBER])
    draw_geometry(a0, d, k, rects)
    style(a0, CHAMBER)
    a0.set_title("air speed  |u|", color=INK, fontsize=11, loc="left")
    tres = res / 3
    sp = raster(d, np.hypot(d["u"][k], d["v"][k]).astype(np.float64), k, TIP, tres)
    w = raster(d, vorticity(d, k), k, TIP, tres)
    a1.imshow(np.ma.masked_invalid(sp), origin="lower", cmap=SPEED, vmin=0, vmax=vmax, interpolation="bilinear",
              extent=[v * 1e3 for v in TIP])
    a2.imshow(np.ma.masked_invalid(w), origin="lower", cmap=VORT, norm=TwoSlopeNorm(0, -wmax, wmax),
              interpolation="bilinear", extent=[v * 1e3 for v in TIP])
    for a, t in ((a1, "tip close-up: speed"), (a2, "tip close-up: vorticity")):
        draw_geometry(a, d, k, rects)
        style(a, TIP)
        a.set_title(t, color=INK, fontsize=11, loc="left")
    a0.add_patch(plt.Rectangle((TIP[0] * 1e3, TIP[2] * 1e3), (TIP[1] - TIP[0]) * 1e3, (TIP[3] - TIP[2]) * 1e3,
                               fc="none", ec=INK, lw=0.7, ls=(0, (3, 2)), zorder=6))
    if trace is not None:
        tt, yy = trace
        ins = a0.inset_axes([0.62, 0.035, 0.35, 0.13], facecolor="#0d1420cc")
        ins.plot(tt, yy, color=STEEL, lw=1)
        ins.plot([tt[k]], [yy[k]], "o", color="#ffd27a", ms=4)
        ins.set_xticks([])
        ins.set_yticks([])
        for spn in ins.spines.values():
            spn.set_color("#33415a")
        ins.set_title("tip position over one cycle", color=INK, fontsize=7, pad=2)
    fig.text(0.985, 0.965, f"t = {d['t'][k] * 1e3:7.3f} ms", color=INK, ha="right", fontsize=10, family="monospace")
    fig.savefig(path, dpi=100, facecolor=BG)
    plt.close(fig)


def movie(d, out, fps=None, loops=3):
    nt = len(d["t"])
    fps = fps or int(np.clip(nt // 3, 8, 24))  # ~1.5-3 s per reed cycle
    vmax, wmax, _ = limits(d, np.arange(nt))
    rects = solid_rects(d)
    xc, L = d["xc"], float(d["L"])
    itip = int(np.argmin(np.abs(xc - (L - 0.2e-3))))
    trace = (d["t"], d["Y"][:, itip] * 1e3)
    tmp = tempfile.mkdtemp()
    for k in range(nt):
        frame(d, k, vmax, wmax, rects, os.path.join(tmp, f"f{k:04d}.png"), trace=trace)
    frame(d, nt // 3, vmax, wmax, rects, os.path.join(out, "hero.png"), size=(12.8, 7.2), trace=trace)
    lst = os.path.join(tmp, "list.txt")
    with open(lst, "w") as fh:
        for _ in range(loops):
            for k in range(nt):
                fh.write(f"file '{tmp}/f{k:04d}.png'\nduration {1 / fps}\n")
    if shutil.which("ffmpeg"):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst,
                        "-vf", f"fps={fps},format=yuv420p", "-c:v", "libx264", "-crf", "22", "-preset", "slow",
                        "-movflags", "+faststart", os.path.join(out, "cycle.mp4")], check=True)
        # GitHub READMEs can't embed mp4 from the repo: also write a palette-optimised GIF (one cycle, looping)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", os.path.join(out, "cycle.mp4"), "-t",
                        f"{nt / fps:.4f}", "-vf", f"fps={fps},scale=800:-1:flags=lanczos,split[a][b];"
                        "[a]palettegen=max_colors=128:stats_mode=full[p];[b][p]paletteuse=dither=bayer:bayer_scale=4",
                        "-loop", "0", os.path.join(out, "flow.gif")], check=True)
    shutil.rmtree(tmp)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--out", default=None)
    ap.add_argument("--frames", type=int, default=4)
    ap.add_argument("--movie", action="store_true")
    a = ap.parse_args()
    out = a.out or a.run
    os.makedirs(out, exist_ok=True)
    d = load(a.run)
    stills(d, out, a.frames)
    if a.movie:
        movie(d, out)
