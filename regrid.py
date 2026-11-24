#!/usr/bin/env python3
"""
regrid.py -- build a solution-adapted grid from a pilot run's resolution maps.

A side-view run writes runs/<name>/diag.npz with per-cell running maxima of
  re_cell : |u| * cell size / nu          (under-resolved shear if >> 10)
  dvel    : largest velocity jump to a neighbouring cell [m/s]
For every existing cell we compute a target size
  h_t = h * clip( min(Re_target / re_cell, jump_target / dvel), shrink, grow )
then take the envelope (min over the other direction) per column / row, since
the grid is tensor-product, smooth it to a max growth ratio, and re-integrate
faces, keeping the geometric lines exact (x = 0, L, L + gap_tip; y = 0, -t_p).

    python regrid.py runs/<pilot> [more pilots...] --out grid_adapted.npz
    python reed_side.py --grid-file grid_adapted.npz ...

Note: the time step is set by the smallest cell anywhere, so this mostly cuts
the cell count (cost per step), not the number of steps.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np


def envelope(f, h, red_axis):
    return np.min(f, axis=red_axis) if f.size else h


def smooth_spacing(h, ratio):
    h = h.copy()
    for _ in range(2):
        for k in range(1, len(h)):
            h[k] = min(h[k], h[k - 1] * ratio)
        for k in range(len(h) - 2, -1, -1):
            h[k] = min(h[k], h[k + 1] * ratio)
    return h


def faces_from_spacing(centers, h_t, fixed, lo, hi):
    """integrate x' = h(x) between consecutive fixed points and rescale each segment."""
    pts = [lo] + sorted(p for p in fixed if lo < p < hi) + [hi]
    out = [lo]
    for a, b in zip(pts[:-1], pts[1:]):
        xs = [a]
        while xs[-1] < b - 1e-15:
            xs.append(xs[-1] + float(np.interp(xs[-1], centers, h_t)))
        n = max(1, len(xs) - 1)
        if n > 1 and (xs[-1] - b) > 0.5 * (xs[-1] - xs[-2]):
            n -= 1  # drop the overshooting cell
        seg = np.array(xs[: n + 1])
        seg = a + (seg - a) * (b - a) / (seg[-1] - a)
        out.extend(seg[1:])
    f = np.array(out)
    f[-1] = hi
    return f


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--re-target", type=float, default=15.0)
    ap.add_argument("--jump-target", type=float, default=1.5, help="max velocity jump per cell [m/s]")
    ap.add_argument("--shrink", type=float, default=0.5)
    ap.add_argument("--grow", type=float, default=3.0)
    ap.add_argument("--h-min", type=float, default=12.5e-6)
    ap.add_argument("--h-max", type=float, default=0.8e-3)
    ap.add_argument("--ratio", type=float, default=1.10)
    ap.add_argument("--out", default="grid_adapted.npz")
    a = ap.parse_args()
    D = [np.load(os.path.join(r, "diag.npz")) for r in a.runs]
    cfg = json.load(open(os.path.join(a.runs[0], "config.json")))["config"]
    info = json.load(open(os.path.join(a.runs[0], "config.json")))["info"]
    xf, yf, solid = D[0]["xf"], D[0]["yf"], D[0]["solid"]
    re = np.maximum.reduce([d["re_cell"] for d in D])
    dv = np.maximum.reduce([d["dvel"] for d in D])
    dx, dy = np.diff(xf), np.diff(yf)
    xc, yc = 0.5 * (xf[1:] + xf[:-1]), 0.5 * (yf[1:] + yf[:-1])
    fac = np.minimum(a.re_target / np.maximum(re, 1e-9), a.jump_target / np.maximum(dv, 1e-9))
    fac = np.clip(fac, a.shrink, a.grow)
    fac[solid] = a.grow
    hx = np.clip(np.min(dx[:, None] * fac, axis=1), a.h_min, a.h_max)
    hy = np.clip(np.min(dy[None, :] * fac, axis=0), a.h_min, a.h_max)
    hx, hy = smooth_spacing(hx, a.ratio), smooth_spacing(hy, a.ratio)
    L, gt, tp = cfg["reed_L"], cfg["gap_tip"], info["plate_t_eff"]
    xf2 = faces_from_spacing(xc, hx, [0.0, L, L + gt], xf[0], xf[-1])
    yf2 = faces_from_spacing(yc, hy, [0.0, -tp], yf[0], yf[-1])
    np.savez(a.out, xf=xf2, yf=yf2)
    n0, n1 = len(dx) * len(dy), (len(xf2) - 1) * (len(yf2) - 1)
    print(f"old grid {len(dx)}x{len(dy)} = {n0} cells, min {dx.min() * 1e6:.1f}/{dy.min() * 1e6:.1f} um")
    print(f"new grid {len(xf2) - 1}x{len(yf2) - 1} = {n1} cells ({n1 / n0:.2f}x), "
          f"min {np.diff(xf2).min() * 1e6:.1f}/{np.diff(yf2).min() * 1e6:.1f} um -> wrote {a.out}")
    print(f"cells with Re_cell > {a.re_target}: {(re > a.re_target).mean() * 100:.1f}% ; max Re_cell {re.max():.0f}")


if __name__ == "__main__":
    main()
