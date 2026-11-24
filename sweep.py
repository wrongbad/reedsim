#!/usr/bin/env python3
"""
sweep.py -- ambient-pressure sweep for the 2-D accordion reed model.

    python sweep.py --pressures 70e3 85e3 101.325e3 115e3 --dp 0 1000 --t-end 0.012

Each (p_amb, dp) pair is one reed2d run (runs/<prefix>_dp<dp>_p<kPa>/). Runs are
launched as separate processes (--parallel at a time, each with --threads numba
threads). Afterwards every run is analysed and a summary is written to
runs/<prefix>_summary.{png,json} plus runs/<prefix>_listen.wav (the looped
steady tone of each run, in order of increasing ambient pressure).

Extra arguments after `--` are passed through to reed2d.py, e.g.
    python sweep.py --pressures 80e3 101325 -- --kick 0.4e-3 --gap 0.04e-3
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from analysis import FS, analyze_run, cents, load, loop_tone, reed_cycles, write_wav  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    argv = sys.argv[1:]
    extra = []
    if "--" in argv:
        k = argv.index("--")
        argv, extra = argv[:k], argv[k + 1:]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pressures", type=float, nargs="+", default=[70e3, 85e3, 101325.0, 115e3])
    ap.add_argument("--dp", type=float, nargs="+", default=[1000.0])
    ap.add_argument("--t-end", type=float, default=0.012)
    ap.add_argument("--prefix", default="sweep")
    ap.add_argument("--out", default=os.path.join(HERE, "runs"))
    ap.add_argument("--parallel", type=int, default=2)
    ap.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--mic", default=None, help="mic used for the listening file (default: last-but-one)")
    ap.add_argument("--analyze-only", action="store_true")
    a = ap.parse_args(argv)

    jobs = []
    for dp in a.dp:
        for p in sorted(a.pressures):
            name = f"{a.prefix}_dp{dp:.0f}_p{p / 1e3:.1f}k"
            jobs.append((name, p, dp))

    if not a.analyze_only:
        os.makedirs(a.out, exist_ok=True)
        env = dict(os.environ, NUMBA_NUM_THREADS=str(a.threads))
        pending = list(jobs)
        running = []
        t0 = time.time()
        while pending or running:
            while pending and len(running) < a.parallel:
                name, p, dp = pending.pop(0)
                cmd = [sys.executable, os.path.join(HERE, "reed2d.py"), "--name", name,
                       "--p-amb", repr(p), "--dp", repr(dp), "--t-end", repr(a.t_end),
                       "--out", a.out, "--no-analysis"] + extra
                log = open(os.path.join(a.out, name + ".log"), "w")
                running.append((name, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env), log))
                print(f"started {name}", flush=True)
            time.sleep(2)
            for r in list(running):
                if r[1].poll() is not None:
                    r[2].close()
                    running.remove(r)
                    print(f"finished {r[0]} (exit {r[1].returncode}) after {time.time() - t0:.0f}s", flush=True)

    summaries = []
    for name, p, dp in jobs:
        rd = os.path.join(a.out, name)
        if not os.path.exists(os.path.join(rd, "series.npz")):
            print(f"missing {rd}; see {rd}.log")
            continue
        summaries.append(analyze_run(rd, verbose=True))
    summarize(summaries, jobs, a)


def summarize(summaries, jobs, a):
    if not summaries:
        return
    out = os.path.join(a.out, a.prefix)
    by_dp = {}
    for s in summaries:
        by_dp.setdefault(s["dp"], []).append(s)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    report = {}
    for dp, ss in sorted(by_dp.items()):
        ss = [s for s in ss if "f_reed" in s]
        if not ss:
            continue
        p = np.array([s["p_amb"] for s in ss])
        rho = np.array([s["rho0"] for s in ss])
        c = np.array([s["cents_vs_vacuum"] for s in ss])
        lab = "plucked, still air" if dp == 0 else f"blown, dp={dp:.0f} Pa"
        ax[0].plot(p / 1e3, c, "o-", label=lab)
        fit = np.polyfit(rho, c, 1) if len(ss) > 1 else [np.nan, np.nan]
        report[f"dp={dp:.0f}"] = dict(
            p_amb=p.tolist(), rho0=rho.tolist(), cents_vs_vacuum=c.tolist(),
            f_reed=[s["f_reed"] for s in ss], amp_mm=[s["amp_last_mm"] for s in ss],
            slope_cents_per_kg_m3=float(fit[0]), intercept_cents=float(fit[1]),
            slope_cents_per_10kPa=float(fit[0] * 1e4 * rho[0] / p[0]))
        for s in ss:
            fc = np.array(s["f_cycles"])
            ax[1].plot(s["amp_cycles_mm"], cents(fc, s["f_vacuum"]), ".-",
                       label=f"{lab}, {s['p_amb'] / 1e3:.0f} kPa")
            ax[2].plot(np.array(s["t_cycles"]) * 1e3, s["amp_cycles_mm"], ".-",
                       label=f"{lab}, {s['p_amb'] / 1e3:.0f} kPa")
    ax[0].set(xlabel="ambient pressure [kPa]", ylabel="pitch shift vs in-vacuo reed [cents]",
              title="reed pitch vs ambient pressure")
    ax[0].axvline(101.325, color="0.7", lw=0.8)
    ax[0].legend()
    ax[1].set(xlabel="cycle amplitude [mm]", ylabel="cents vs vacuum", title="pitch vs amplitude, per cycle")
    ax[1].legend(fontsize=7)
    ax[2].set(xlabel="t [ms]", ylabel="amplitude [mm]", title="amplitude envelope")
    fig.tight_layout()
    fig.savefig(out + "_summary.png", dpi=110)
    plt.close(fig)
    with open(out + "_summary.json", "w") as f:
        json.dump(report, f, indent=2)
    for k, v in report.items():
        print(f"{k}: slope {v['slope_cents_per_10kPa']:+.3f} cents per +10 kPa ambient "
              f"({v['slope_cents_per_kg_m3']:+.3f} cents per kg/m^3)")

    # listening file: looped tone of each run, increasing pressure, common gain
    tones = []
    for name, p, dp in jobs:
        rd = os.path.join(a.out, name)
        if not os.path.exists(os.path.join(rd, "series.npz")):
            continue
        d = load(rd)
        names = [str(s) for s in d["mic_names"]]
        mic = a.mic if a.mic in names else names[max(0, len(names) - 2)]
        tc = reed_cycles(d["t"], d["Y"], d["V"])[0]
        lt = loop_tone(d["t"], d["mics"][:, names.index(mic)], tc, dur=1.5)
        if lt is not None:
            tones += [lt, np.zeros(int(0.3 * FS))]
    if tones:
        y = np.concatenate(tones)
        write_wav(out + "_listen.wav", y)
        print(f"wrote {out}_listen.wav")


if __name__ == "__main__":
    main()
