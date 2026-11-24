#!/usr/bin/env python3
"""
analysis.py -- pitch extraction, audio rendering and plots for reed2d runs.

    python analysis.py runs/<name> [runs/<name2> ...]

Per run writes into the run directory:
  summary.json            pitch (Hz, cents vs. the in-vacuo reed), amplitude, ...
  mic_<name>.wav          the simulated pressure at each microphone, 48 kHz
  mic_<name>_loop.wav     the last whole reed cycles tiled to a 2 s tone (for listening)
  overview.png            reed motion, force, slot flow, mic pressures, spectrum, pitch
  fields.png              pressure / speed snapshots over the last reed period
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import signal  # noqa: E402
from scipy.io import wavfile  # noqa: E402

FS = 48000


def cents(f, f_ref):
    return 1200.0 * np.log2(np.asarray(f) / f_ref)


def load(run_dir):
    d = dict(np.load(os.path.join(run_dir, "series.npz")))
    # drop any trailing never-written records (t must increase)
    t = d["t"]
    bad = np.nonzero(np.diff(t) <= 0)[0]
    if len(bad):
        n = bad[0] + 1
        for k, v in d.items():
            if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == len(t):
                d[k] = v[:n]
    with open(os.path.join(run_dir, "config.json")) as f:
        d["meta"] = json.load(f)
    return d


def reed_cycles(t, Y, V):
    """Cycle-by-cycle frequency and amplitude of the reed.

    Uses every zero crossing of the reed velocity (displacement extrema), located
    by linear interpolation; velocity is smooth (it integrates the force), so the
    timing is precise to ~1e-9 s. Each full period is measured between crossings
    two apart, so a new estimate is available every half cycle.
    Returns (tc_up, t_mid, f, amp, mean): tc_up are the upward crossings
    (displacement minima, used to cut whole cycles), the rest are per estimate."""
    sgn = np.sign(V)
    k = np.where(sgn[:-1] * sgn[1:] < 0)[0]
    e = (np.array([]),) * 5
    if len(k) < 3:
        return e
    tc = t[k] - V[k] * (t[k + 1] - t[k]) / (V[k + 1] - V[k])
    up = V[k] < 0
    f = 1.0 / (tc[2:] - tc[:-2])
    tmid = 0.5 * (tc[2:] + tc[:-2])
    amp = np.array([0.5 * (Y[a:c + 1].max() - Y[a:c + 1].min()) for a, c in zip(k[:-2], k[2:])])
    mean = np.array([Y[a:c + 1].mean() for a, c in zip(k[:-2], k[2:])])
    return tc[up], tmid, f, amp, mean


def lowpass_resample(t, x, fs=FS, f_hi=18000.0):
    fs_raw = 1.0 / np.median(np.diff(t))
    sos = signal.butter(6, f_hi, "low", fs=fs_raw, output="sos")
    y = signal.sosfiltfilt(sos, x)
    tt = np.arange(t[0], t[-1], 1.0 / fs)
    return tt, np.interp(tt, t, y)


def loop_tone(t, x, tc, n_cycles=None, dur=2.0, fs=FS):
    """Tile the last whole reed cycles of x(t) into a `dur`-second tone."""
    if len(tc) < 3:
        return None
    # Loop exactly ONE reed cycle by default: tiling several cycles of a signal whose
    # amplitude is still drifting puts a step at every seam, i.e. an amplitude
    # modulation at f/n that is heard as a low "undertone" (sidebands at f(1 +- k/n)).
    n = n_cycles or 1
    ta, tb = tc[-1 - n], tc[-1]
    m = (t >= ta) & (t <= tb)
    ts, xs = t[m], x[m]
    xs = xs - (xs[0] + (xs[-1] - xs[0]) * (ts - ts[0]) / (ts[-1] - ts[0]))  # close the loop
    xs = xs - xs.mean()
    Tseg = tb - ta
    tt = np.arange(0, dur, 1.0 / fs)
    # anti-alias the segment first (periodic filtering via tiling three copies)
    fs_raw = 1.0 / np.median(np.diff(ts))
    sos = signal.butter(6, 18000.0, "low", fs=fs_raw, output="sos")
    x3 = signal.sosfiltfilt(sos, np.concatenate([xs, xs, xs]))[len(xs):2 * len(xs)]
    y = np.interp(ta + np.mod(tt, Tseg), ts, x3)
    fade = int(0.02 * fs)
    w = np.ones_like(y)
    w[:fade] = np.linspace(0, 1, fade)
    w[-fade:] = np.linspace(1, 0, fade)
    return y * w


def write_wav(path, y, peak=None):
    peak = peak or (np.max(np.abs(y)) + 1e-30)
    wavfile.write(path, FS, (0.5 * y / peak).astype(np.float32))
    return peak


def mic_pitch(t, x, f_guess):
    """Pitch of a mic signal from the autocorrelation peak near 1/f_guess."""
    tt, y = lowpass_resample(t, x, fs=400000.0, f_hi=8000.0)
    n = len(y) // 2
    y = y[-n:] - y[-n:].mean()
    lag0 = int(round(400000.0 / f_guess))
    lo, hi = int(lag0 * 0.8), int(lag0 * 1.25)
    ac = np.array([np.dot(y[:-L], y[L:]) / (len(y) - L) for L in range(lo, hi)])
    k = int(np.argmax(ac))
    if 0 < k < len(ac) - 1:
        a, b, c = ac[k - 1], ac[k], ac[k + 1]
        k = k + 0.5 * (a - c) / (a - 2 * b + c)
    return 400000.0 / (lo + k)


def analyze_run(run_dir, make_plots=True, verbose=True):
    d = load(run_dir)
    meta = d["meta"]
    cfg, info = meta["config"], meta["info"]
    t, Y, V, F, Q = d["t"], d["Y"], d["V"], d["F"], d["Qslot"]
    f_vac = float(d["f_vacuum"])
    names = [str(s) for s in d["mic_names"]]
    tc, tmid, fcyc, amp, ymean = reed_cycles(t, Y, V)

    summ = dict(name=cfg["name"], p_amb=cfg["p_amb"], dp=cfg["dp"], rho0=float(d["rho0"]),
                f_vacuum=f_vac, t_end=float(t[-1]), n_cycles=int(len(fcyc)))
    if len(fcyc) >= 2:
        last = fcyc[2:] if len(fcyc) > 3 else fcyc  # skip the first (start-up) cycle
        nl = max(1, min(len(last), 6))
        f_est = float(np.mean(last[-nl:]))
        summ.update(
            f_reed=f_est, cents_vs_vacuum=float(cents(f_est, f_vac)),
            cents_vs_A440=float(cents(f_est, 440.0)),
            f_cycles=fcyc.tolist(), amp_cycles_mm=(1e3 * amp).tolist(),
            t_cycles=tmid.tolist(),
            amp_last_mm=float(1e3 * amp[-1]), amp_growth_per_cycle=float(amp[-1] / amp[-2]) if len(amp) > 1 else None,
            mean_disp_last_mm=float(1e3 * ymean[-1]),
            f_spread_last_cents=float(np.ptp(cents(last[-nl:], f_vac))),
        )
    # --- audio ---
    mics = d["mics"]
    summ["mics"] = {}
    for k, nm in enumerate(names):
        x = mics[:, k]
        tt, y = lowpass_resample(t, x)
        y = y - y.mean()
        sos = signal.butter(2, 20.0, "high", fs=FS, output="sos")
        y = signal.sosfiltfilt(sos, y) if len(y) > 30 else y
        pk = write_wav(os.path.join(run_dir, f"mic_{nm}.wav"), y)
        entry = dict(peak_Pa=float(pk), rms_Pa=float(np.std(y[len(y) // 2:])))
        lt = loop_tone(t, x, tc)
        if lt is not None:
            write_wav(os.path.join(run_dir, f"mic_{nm}_loop.wav"), lt)
        if len(fcyc) >= 4:
            try:
                entry["f_autocorr"] = float(mic_pitch(t, x, summ.get("f_reed", f_vac)))
            except Exception:
                pass
        summ["mics"][nm] = entry
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summ, f, indent=2)
    if verbose:
        if "f_reed" in summ:
            print(f"[{cfg['name']}] p_amb={cfg['p_amb'] / 1e3:.2f} kPa dp={cfg['dp']:.0f} Pa: "
                  f"f={summ['f_reed']:.3f} Hz ({summ['cents_vs_vacuum']:+.2f} c vs vacuum, "
                  f"{summ['cents_vs_A440']:+.2f} c vs A440), amp={summ['amp_last_mm']:.3f} mm, "
                  f"{len(fcyc)} cycles")
        else:
            print(f"[{cfg['name']}] not enough reed cycles to estimate pitch")
    if make_plots:
        plot_overview(run_dir, d, summ, tc, tmid, fcyc, amp)
        if os.path.exists(os.path.join(run_dir, "snapshots.npz")):
            plot_fields(run_dir)
    return summ


def plot_overview(run_dir, d, summ, tc, tmid, fcyc, amp):
    t, Y, F, Q = d["t"], d["Y"], d["F"], d["Qslot"]
    names = [str(s) for s in d["mic_names"]]
    f_vac = float(d["f_vacuum"])
    tp = d["meta"]["info"]["plate_t_eff"]
    fig, ax = plt.subplots(3, 2, figsize=(13, 10))
    a = ax[0, 0]
    a.plot(t * 1e3, Y * 1e3, lw=0.8)
    a.axhspan(-tp * 1e3 - d["meta"]["config"]["set_gap"] * 1e3, -d["meta"]["config"]["set_gap"] * 1e3,
              color="0.85", label="plate (slot walls)")
    a.set(xlabel="t [ms]", ylabel="reed displacement [mm]", title="reed (section) motion")
    a.legend(loc="upper right", fontsize=8)
    a = ax[0, 1]
    a.plot(t * 1e3, F, lw=0.6, label="air force on reed [N/m]")
    a2 = a.twinx()
    a2.plot(t * 1e3, Q * 1e3, lw=0.6, color="C1", label="slot flow [L/s per m]")
    a.set(xlabel="t [ms]", ylabel="force per length [N/m]", title="air force & volume flow through slot")
    a2.set_ylabel("volume flow per length [1e-3 m^2/s]", color="C1")
    a = ax[1, 0]
    n_show = (t > t[-1] - 3.0 / f_vac)
    for k, nm in enumerate(names):
        a.plot(t[n_show] * 1e3, d["mics"][n_show, k], lw=0.8, label=nm)
    a.set(xlabel="t [ms]", ylabel="p - p_amb [Pa]", title="microphones (last 3 periods)")
    a.legend(fontsize=8)
    a = ax[1, 1]
    for k, nm in enumerate(names):
        lt = loop_tone(t, d["mics"][:, k], tc)
        if lt is None:
            continue
        spec = np.abs(np.fft.rfft(lt * np.hanning(len(lt))))
        fr = np.fft.rfftfreq(len(lt), 1 / FS)
        a.semilogy(fr, spec / spec.max() + 1e-9, lw=0.7, label=nm)
    a.set(xlim=(0, 12000), ylim=(1e-5, 2), xlabel="Hz", ylabel="rel. magnitude",
          title="mic spectra (looped steady part)")
    a.legend(fontsize=8)
    a = ax[2, 0]
    if len(fcyc):
        a.plot(tmid * 1e3, cents(fcyc, f_vac), "o-")
        a.set(xlabel="t [ms]", ylabel="cents vs in-vacuo reed", title="cycle-by-cycle reed pitch")
        a.axhline(0, color="k", lw=0.5)
    a = ax[2, 1]
    if len(fcyc):
        a.plot(amp * 1e3, cents(fcyc, f_vac), "o-")
        a.set(xlabel="cycle amplitude [mm]", ylabel="cents vs in-vacuo reed",
              title="pitch vs amplitude (backbone)")
    cfg = d["meta"]["config"]
    fig.suptitle(f"{cfg['name']}: p_amb={cfg['p_amb'] / 1e3:.2f} kPa, dp={cfg['dp']:.0f} Pa, "
                 f"rho0={float(d['rho0']):.3f} kg/m^3"
                 + (f"  ->  f={summ['f_reed']:.3f} Hz ({summ['cents_vs_vacuum']:+.2f} c)" if "f_reed" in summ else ""))
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "overview.png"), dpi=110)
    plt.close(fig)


def _mirror(zf, q, odd=False):
    zfull = np.concatenate([-zf[::-1], zf[1:]])
    qq = np.concatenate([(-q if odd else q)[::-1], q], axis=0)
    return zfull, qq


def plot_fields(run_dir, n_frames=6, zoom=(4.0e-3, -4.5e-3, 2.5e-3)):
    s = np.load(os.path.join(run_dir, "snapshots.npz"))
    zf, yf = s["zf"], s["yf"]
    side = "side" in s.files
    if side:  # side view: x along the tongue, zoom on the tip region, no mirroring
        L = float(s["L"])
        zoom = None
    nsn = len(s["t"])
    idx = np.linspace(0, nsn - 1, min(n_frames, nsn)).astype(int)
    solid = s["solid"]
    fig, ax = plt.subplots(2, len(idx) + 1, figsize=((4.2 if "side" in s.files else 3.2) * (len(idx) + 1), 8.5),
                           gridspec_kw=dict(width_ratios=[1] * len(idx) + [1.3]))
    ax = np.atleast_2d(ax)
    pmax = np.percentile(np.abs(s["p"][:, ~solid]), 99.5)
    vmax = np.percentile(np.hypot(s["u"], s["v"])[:, ~solid], 99.8)
    mir = (lambda zz, q: (zz, q)) if side else _mirror
    if side:
        xlim, ylim = (L - 6e-3, L + 2e-3), (-3.5e-3, 1.5e-3)
    else:
        zm, ylo, yhi = zoom
        xlim, ylim = (-zm, zm), (ylo, yhi)
    for c, k in enumerate(idx):
        mask = s["alpha"][k] < 0.5
        p = np.ma.array(s["p"][k], mask=mask)
        sp = np.ma.array(np.hypot(s["u"][k], s["v"][k]), mask=mask)
        Z, P = mir(zf, p)
        _, SP = mir(zf, sp)
        im0 = ax[0, c].pcolormesh(Z * 1e3, yf * 1e3, P.T, cmap="RdBu_r", vmin=-pmax, vmax=pmax, shading="flat")
        im1 = ax[1, c].pcolormesh(Z * 1e3, yf * 1e3, SP.T, cmap="magma", vmin=0, vmax=vmax, shading="flat")
        for r in (0, 1):
            ax[r, c].set(xlim=(xlim[0] * 1e3, xlim[1] * 1e3), ylim=(ylim[0] * 1e3, ylim[1] * 1e3), aspect="equal")
            ax[r, c].set_facecolor("0.55")
        ax[0, c].set_title(f"t={s['t'][k] * 1e3:.3f} ms")
    # full-domain pressure at the last frame
    k = idx[-1]
    mask = s["alpha"][k] < 0.5
    Z, P = mir(zf, np.ma.array(s["p"][k], mask=mask))
    _, SP = mir(zf, np.ma.array(np.hypot(s["u"][k], s["v"][k]), mask=mask))
    ax[0, -1].pcolormesh(Z * 1e3, yf * 1e3, P.T, cmap="RdBu_r", vmin=-pmax, vmax=pmax)
    ax[1, -1].pcolormesh(Z * 1e3, yf * 1e3, SP.T, cmap="magma", vmin=0, vmax=vmax)
    for r in (0, 1):
        ax[r, -1].set_aspect("equal")
        ax[r, -1].set_facecolor("0.55")
    ax[0, -1].set_title("whole domain")
    fig.colorbar(im0, ax=ax[0, :], label="p - p_amb [Pa]", shrink=0.8)
    fig.colorbar(im1, ax=ax[1, :], label="|u| [m/s]", shrink=0.8)
    for a in ax[:, 0]:
        a.set_ylabel("y [mm]")
    for a in ax[1, :]:
        a.set_xlabel("x along tongue [mm]" if side else "z [mm]")
    fig.savefig(os.path.join(run_dir, "fields.png"), dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    for rd in sys.argv[1:]:
        analyze_run(rd)
