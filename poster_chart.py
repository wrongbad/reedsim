"""Headline chart for the README: pitch change vs ambient pressure / altitude.

    python poster_chart.py runs_gpu/sw2_dp_compare.json --out poster
Writes altitude_light.png / altitude_dark.png and prints the fitted shifts.
"""
import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

P0 = 101325.0


def p_at_ft(ft):  # ICAO standard atmosphere, troposphere
    return P0 * (1 - 2.25577e-5 * ft * 0.3048) ** 5.25588


THEMES = {
    "light": dict(bg="#ffffff", fg="#1f2430", grid="#e3e6ec", muted="#6b7280", c=["#2f5fb3", "#c2702a"]),
    "dark": dict(bg="#0d1117", fg="#e6edf3", grid="#262c36", muted="#8b949e", c=["#79a6ff", "#f0a35e"]),
}


def fit(js):
    out = {}
    for key, g in js.items():
        p = np.array([r["p_amb"] for r in g["runs"]])
        c = np.array([r["cents_at_ref"] for r in g["runs"]])
        ok = np.isfinite(c)
        a, b = np.polyfit(p[ok], c[ok], 1)
        sea = a * P0 + b
        out[key] = dict(p=p[ok], rel=c[ok] - sea, slope=a, amp=g["ref_amp_mm"],
                        shift=lambda q, a=a: a * (q - P0))
    return out


def draw(F, theme, path):
    T = THEMES[theme]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig, ax = plt.subplots(figsize=(9, 5.2), facecolor=T["bg"])
    ax.set_facecolor(T["bg"])
    pp = np.linspace(66e3, 133e3, 50)
    for k, (key, f) in enumerate(F.items()):
        col = T["c"][k]
        dp = key.split("=")[1]
        ax.plot(pp / 1e3, f["shift"](pp), color=col, lw=1.6, alpha=0.55)
        ax.plot(f["p"] / 1e3, f["rel"], "o", color=col, ms=7, mec=T["bg"], mew=1.5,
                label=f"blown at {int(dp) / 1000:g} kPa  ({f['slope'] * 1e4:+.1f} cents per 10 kPa)")
    for ft, name in ((0, "sea level"), (5000, "5,000 ft"), (10000, "10,000 ft")):
        p = p_at_ft(ft) / 1e3
        ax.axvline(p, color=T["grid"], lw=1.2, zorder=0)
        ax.text(p, 9.3, name, ha="center", va="bottom", color=T["fg"], fontsize=10, fontweight="bold")
        if ft:
            s = [f["shift"](p * 1e3) for f in F.values()]
            ax.text(p + 0.8, max(s) + 0.5, f"+{min(s):.1f} to +{max(s):.1f} ¢", color=T["fg"], fontsize=12,
                    fontweight="bold", ha="right", va="bottom")
    ax.axhline(0, color=T["muted"], lw=0.8, ls=(0, (4, 3)))
    ax.set_xlim(133, 66)  # altitude increases to the right
    ax.set_ylim(-8.5, 9.2)
    ax.set_xlabel("ambient air pressure  [kPa]   (higher altitude →)", color=T["fg"])
    ax.set_ylabel("pitch change vs sea level  [cents, + = sharper]", color=T["fg"])
    ax.grid(color=T["grid"], lw=0.6)
    ax.tick_params(colors=T["muted"])
    for s in ax.spines.values():
        s.set_visible(False)
    leg = ax.legend(loc="upper left", frameon=False, fontsize=10)
    for t in leg.get_texts():
        t.set_color(T["fg"])
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=T["bg"])
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("compare_json")
    ap.add_argument("--out", default="poster")
    a = ap.parse_args()
    F = fit(json.load(open(a.compare_json)))
    os.makedirs(a.out, exist_ok=True)
    for th in THEMES:
        draw(F, th, os.path.join(a.out, f"altitude_{th}.png"))
    for key, f in F.items():
        print(key, f"slope {f['slope'] * 1e4:+.2f} c/10kPa, ref amp {f['amp']:.2f} mm",
              " ".join(f"{ft}ft:{f['shift'](p_at_ft(ft)):+.2f}" for ft in (5000, 10000)),
              "points:", np.round(f["rel"], 2).tolist())
