#!/usr/bin/env python3
"""
compare.py -- compare reed pitch across runs (e.g. an ambient-pressure sweep).

    python compare.py runs/ex1_* [--ref-amp 0.40] [--out runs/ex1_compare]

Pitch depends on vibration amplitude, and in a finite-length simulation the
amplitude is still drifting (decaying or growing). So runs are compared at a
common reference amplitude: each run's per-half-cycle (amplitude, pitch) curve is
fitted locally and evaluated at --ref-amp (default: the middle of the amplitude
range shared by the runs of each blowing pressure).
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from analysis import analyze_run, cents  # noqa: E402


def pitch_at_amp(s, a_ref, skip=2, halfwidth=0.08):
    """Cents vs vacuum at amplitude a_ref [mm]: local quadratic fit of the
    (amp, cents) samples within +-halfwidth mm (skipping the first `skip` start-up
    estimates). Returns (cents, n_points) or (nan, 0)."""
    a = np.array(s.get("amp_cycles_mm", []))[skip:]
    c = cents(np.array(s.get("f_cycles", []))[skip:], s["f_vacuum"]) if len(a) else np.array([])
    m = np.abs(a - a_ref) < halfwidth
    if m.sum() < 3:
        return np.nan, 0
    deg = 2 if m.sum() >= 6 else 1
    p = np.polyfit(a[m] - a_ref, c[m], deg)
    return float(p[-1]), int(m.sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--ref-amp", type=float, default=None, help="reference amplitude [mm]")
    ap.add_argument("--out", default=None)
    ap.add_argument("--reanalyze", action="store_true")
    a = ap.parse_args()
    runs = [r for r in a.runs if os.path.isdir(r) and os.path.exists(os.path.join(r, "series.npz"))]
    summ = []
    for r in runs:
        sj = os.path.join(r, "summary.json")
        if a.reanalyze or not os.path.exists(sj):
            analyze_run(r, verbose=False)
        with open(sj) as f:
            summ.append(json.load(f))
    out = a.out or os.path.commonprefix([os.path.normpath(r) for r in runs]).rstrip("_") + "_compare"
    groups = {}
    for s in summ:
        groups.setdefault(s["dp"], []).append(s)

    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    report = {}
    for gi, (dp, ss) in enumerate(sorted(groups.items())):
        ss = sorted(ss, key=lambda s: s["p_amb"])
        lab = "plucked (still air)" if dp == 0 else f"blown dp={dp:.0f} Pa"
        amps = [np.array(s.get("amp_cycles_mm", []))[2:] for s in ss]
        if a.ref_amp is None:
            lo = max((x.min() for x in amps if len(x)), default=np.nan)
            hi = min((x.max() for x in amps if len(x)), default=np.nan)
            a_ref = 0.5 * (lo + hi) if hi > lo else np.nan
        else:
            a_ref = a.ref_amp
        rows = []
        for s in ss:
            c_ref, npts = pitch_at_amp(s, a_ref) if np.isfinite(a_ref) else (np.nan, 0)
            rows.append(dict(name=s["name"], p_amb=s["p_amb"], rho0=s["rho0"], cents_at_ref=c_ref,
                             n_fit=npts, cents_last=s.get("cents_vs_vacuum"), amp_last=s.get("amp_last_mm"),
                             f_last=s.get("f_reed")))
            if s.get("f_cycles"):
                ax[0].plot(s["amp_cycles_mm"][2:], cents(np.array(s["f_cycles"][2:]), s["f_vacuum"]), ".-",
                           lw=0.7, ms=3, label=f"{lab}, {s['p_amb'] / 1e3:.1f} kPa")
                ax[1].plot(np.array(s["t_cycles"]) * 1e3, s["amp_cycles_mm"], ".-", lw=0.7, ms=3,
                           label=f"{lab}, {s['p_amb'] / 1e3:.1f} kPa")
        p = np.array([r["p_amb"] for r in rows])
        rho = np.array([r["rho0"] for r in rows])
        c = np.array([r["cents_at_ref"] for r in rows])
        ok = np.isfinite(c)
        fit = np.polyfit(p[ok], c[ok], 1) if ok.sum() >= 2 else [np.nan, np.nan]
        ax[2].plot(p[ok] / 1e3, c[ok], "o-", label=f"{lab} @ {a_ref:.3f} mm")
        report[f"dp={dp:.0f}"] = dict(ref_amp_mm=a_ref, runs=rows,
                                      slope_cents_per_10kPa=float(fit[0] * 1e4),
                                      intercept_cents_at_0Pa=float(fit[1]))
        if ok.sum() >= 2:
            print(f"{lab}: at amplitude {a_ref:.3f} mm, pitch changes {fit[0] * 1e4:+.3f} cents per +10 kPa "
                  f"ambient (extrapolated vacuum offset {fit[1]:+.2f} c)")
        for r in rows:
            print(f"   {r['p_amb'] / 1e3:7.2f} kPa rho0={r['rho0']:.3f}: {r['cents_at_ref']:+8.3f} c "
                  f"(n={r['n_fit']}), last-cycle {r['cents_last'] if r['cents_last'] is not None else float('nan'):+.3f} c "
                  f"at {r['amp_last'] or float('nan'):.3f} mm")
    ax[0].set(xlabel="amplitude [mm]", ylabel="cents vs in-vacuo reed", title="pitch vs amplitude (each half-cycle)")
    ax[1].set(xlabel="t [ms]", ylabel="amplitude [mm]", title="amplitude envelope")
    ax[2].set(xlabel="ambient pressure [kPa]", ylabel="cents vs in-vacuo reed", title="pitch at matched amplitude")
    ax[2].axvline(101.325, color="0.7", lw=0.8)
    for x in ax:
        x.legend(fontsize=7)
        x.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out + ".png", dpi=110)
    with open(out + ".json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"wrote {out}.png / .json")


if __name__ == "__main__":
    main()
