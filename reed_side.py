#!/usr/bin/env python3
"""
reed_side.py -- side-view 2-D model of an accordion free reed (GPU, numba-cuda).

The slice runs ALONG the tongue (x = rivet -> tip, y = normal to the plate):

      y ^            bellows  (p_amb + dp)
        |   heel  ___________________________ tongue (Euler-Bernoulli cantilever,
        |  [####]/                  set ->   \   rest "set" w0(x) = s_tip (x/L)^2)
   -----+--------+-------- slot ---------------+  +-------  reed plate
        |        |                             tip clearance (resolved)
        |   +----+         reed chamber            +----+
        |   |                                           |
        |   +-----------------+    outlet    +----------+
        |                       room (p_amb)
        +--------------------------------------------------> x

What a 2-D side view cannot contain -- the curved/triangular side openings
along both edges of the tongue -- is supplied by a sub-grid "leak" model: at
every column x along the tongue, air passes straight through the tongue (from
the fluid cell just above it to the one just below) with a local gap-flow
equation
      rho l_eff dU/dt = dp - rho U|U| / (2 Cd^2) - 12 mu l_v U / a^2
      a   = sqrt(g^2 + d^2)   corner-to-corner opening per side
            (d = how far the tongue edge sits above the plate, or below it)
      l_v = vertical overlap of the tongue edge with the slot wall
      l_eff = l_v + a      (gap inertia incl. end correction)
and volume flow 2 a U per unit tongue length.

The 2-D slab represents a depth equal to the tongue width b (the slot width), so
the chamber height is scaled to give the right chamber volume per unit length.

Fluid numerics are the same as reed2d / reed_gpu (FV, MUSCL + Rusanov/Thornber,
SSP-RK2, mixed precision, cut cells for the moving tongue, CUDA graphs); here
every column along the tongue has its own height.

    python reed_side.py --prefix s1 --t-end 0.03 --p-amb 101325 --dp 500 1000
    python reed_side.py --quiet-check
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import time
from dataclasses import asdict, dataclass, field

import numpy as np
from numba import cuda, float32, float64, int32

from reed_gpu import H, ONE, TWO, Z, d_mc, d_mix, d_prim, d_rusanov, d_search, f32, g_prims_all

BETAL = (1.8751040687, 4.6940911330, 7.8547574382, 10.9955407349)


# ----------------------------------------------------------------------------
@dataclass
class SideConfig:
    name: str = "side"
    # air
    p_amb: float = 101325.0
    T: float = 293.15
    gamma: float = 1.4
    R_air: float = 287.05
    mu: float = 1.81e-5
    dp: float = 1000.0          # bellows over-pressure [Pa]
    t_ramp: float = 1.0e-3      # dp ramp time [s]
    # tongue
    reed_L: float = 23.75e-3
    reed_b: float = 2.5e-3
    reed_h: float = 0.30e-3
    E: float = 206e9
    rho_s: float = 7850.0
    Q_mech: float = 800.0
    n_modes: int = 3
    set_tip: float = 0.30e-3    # rest clearance under the tongue at the tip
    set_exp: float = 2.0        # set profile w0 = set_tip (x/L)^set_exp
    kick: float = 0.0           # initial tip displacement of mode 1 [m] (neg = into slot)
    # plate / slot / leak model
    plate_t: float = 1.5e-3
    gap: float = 0.05e-3        # side clearance (sub-grid)
    gap_tip: float = 0.05e-3    # tip clearance (resolved)
    Cd: float = 0.7             # discharge coefficient of the side openings
    jet: float = 1.0            # side-jet momentum: 1 = physical (leaked mass enters at the jet
                                # velocity U/Cd from the gap equation, along the slot); 0 = no jet momentum
    heel: float = 3.0e-3        # riveted heel length
    # chamber (slab depth = reed width, so height ~ real width*depth / b)
    chamber: bool = True
    ch_x0: float = -2.0e-3
    ch_x1_extra: float = 4.0e-3  # chamber end beyond tip
    chamber_h: float = 20e-3
    outlet_x0_extra: float = -6.0e-3  # outlet start relative to tip
    outlet_w: float = 8e-3
    wall_t: float = 2e-3
    # domain / grid
    x_lo: float = -8e-3
    x_hi_extra: float = 14e-3
    y_top: float = 14e-3
    room_below: float = 14e-3
    dx_body: float = 0.10e-3
    dx_tip: float = 0.025e-3
    tip_zone: float = 0.6e-3
    dy_fine: float = 0.03e-3
    core_up: float = 2.5e-3
    core_dn: float = 2.0e-3
    d_coarse: float = 0.6e-3
    stretch: float = 1.08
    sponge: float = 5e-3
    sponge_sigma: float = 1.0e6     # strong enough to hold reservoir pressures under throughflow
    grid_file: str = ""         # optional npz with xf, yf (e.g. from regrid.py)
    sponge_mom: float = 0.03    # momentum damping in sponges, as fraction of sponge_sigma
    # time
    t_end: float = 0.03
    cfl: float = 0.8
    rec_every: int = 8
    diag_every: int = 200
    snap_cycles: float = 1.0
    snap_per_cycle: int = 12
    mics: list = field(default_factory=list)

    def rho0(self):
        return self.p_amb / (self.R_air * self.T)

    def c0(self):
        return math.sqrt(self.gamma * self.R_air * self.T)


# ----------------------------------------------------------------------------
# grid helpers
# ----------------------------------------------------------------------------
def _grow(n_len, h0, hmax, ratio):
    """cell sizes starting at h0, growing by `ratio` up to hmax, filling >= n_len."""
    s, tot, h = [], 0.0, h0
    while tot < n_len - 1e-15:
        s.append(h)
        tot += h
        h = min(h * ratio, hmax)
    return np.array(s)


def segment(a, b, ha, hb, ratio):
    """faces strictly inside (a, b] going from spacing ha at a to hb at b."""
    L = b - a
    if abs(ha - hb) < 1e-15:
        n = max(1, int(round(L / ha)))
        sizes = np.full(n, L / n)
    elif ha < hb:
        sizes = _grow(L, ha, hb, ratio)
        sizes *= L / sizes.sum()
    else:
        sizes = _grow(L, hb, ha, ratio)[::-1]
        sizes *= L / sizes.sum()
    return a + np.cumsum(sizes)


def axis_from_knots(knots, ratio):
    """knots: list of (position, spacing), increasing positions."""
    f = [knots[0][0]]
    for (a, ha), (b, hb) in zip(knots[:-1], knots[1:]):
        f.extend(segment(a, b, ha, hb, ratio))
    f = np.array(f)
    f[-1] = knots[-1][0]
    assert np.all(np.diff(f) > 0)
    return f


def cantilever_modes(x, L, n):
    """clamped-free mode shapes, tip-normalised (phi(L) = 1)."""
    out = []
    for k in range(n):
        bl = BETAL[k]
        s = (math.cosh(bl) + math.cos(bl)) / (math.sinh(bl) + math.sin(bl))
        def raw(xx):
            z = bl * xx / L
            return np.cosh(z) - np.cos(z) - s * (np.sinh(z) - np.sin(z))
        out.append(raw(x) / raw(np.array(L)))
    return np.array(out)


# ----------------------------------------------------------------------------
# device kernels
# arrays:  colY[b, s, i] tongue underside height at stage s (0: state n, 1: stage-1)
#          colV[b, s, i] its y-velocity;  colJ[b, 2s(+1), i] top/bottom cut-cell rows
#          leak[b, s, i] side-gap velocity;  mq[b, 0..3, n] = q, qd, q1, qd1
#          bpar[b] = (rho0, p0, c0sq, g1h, rho_up)   bp32 = (rho0, c0sq, g1h)
# ----------------------------------------------------------------------------
@cuda.jit(device=True, inline=True)
def d_interval(alpha, isr, colY, h, yf, b, s, i, j):
    """fluid y-interval of cell (i, j); empty if hi <= lo."""
    a = alpha[b, i, j]
    lo = yf[j]
    hi = yf[j + 1]
    if a <= 0.0:
        return lo, lo
    if isr[i] != 0 and a < 1.0:
        Yb = colY[b, s, i]
        if Yb > lo:
            hi = min(hi, Yb)
        else:
            lo = max(lo, Yb + h)
    return lo, hi


@cuda.jit
def g_flux_x(P, alpha, isr, colY, colV, h, yf, hx, idx_, idxc, FL, FR, bp32, mu, stage):
    j, i, b = cuda.grid(3)
    nb, nx, ny = alpha.shape
    if j >= ny or i > nx or b >= nb:
        return
    rho0 = bp32[b, 0]
    c0sq = bp32[b, 1]
    g1h = bp32[b, 2]
    dyj = yf[j + 1] - yf[j]
    if i > 0:
        loL, hiL = d_interval(alpha, isr, colY, h, yf, b, stage, i - 1, j)
    else:
        loL = yf[j]
        hiL = yf[j]
    if i < nx:
        loR, hiR = d_interval(alpha, isr, colY, h, yf, b, stage, i, j)
    else:
        loR = yf[j]
        hiR = yf[j]
    fL = max(0.0, hiL - loL) / dyj
    fR = max(0.0, hiR - loR) / dyj
    beta = max(0.0, min(hiL, hiR) - max(loL, loR)) / dyj
    l0 = Z
    l1 = Z
    l2 = Z
    r0 = Z
    r1 = Z
    r2 = Z
    if beta > 1e-12:
        bf = float32(beta)
        L = i - 1
        Rr = i
        aL = alpha[b, L, j]
        aR = alpha[b, Rr, j]
        ic = idxc[L]
        g0 = (P[b, 0, Rr, j] - P[b, 0, L, j]) * ic
        g1 = (P[b, 1, Rr, j] - P[b, 1, L, j]) * ic
        g2 = (P[b, 2, Rr, j] - P[b, 2, L, j]) * ic
        rL = P[b, 0, L, j]
        nL = P[b, 1, L, j]
        tL = P[b, 2, L, j]
        if L >= 1 and aL == 1.0 and aR == 1.0 and alpha[b, L - 1, j] == 1.0:
            il = idxc[L - 1]
            x = hx[L]
            rL += x * d_mc((P[b, 0, L, j] - P[b, 0, L - 1, j]) * il, g0)
            nL += x * d_mc((P[b, 1, L, j] - P[b, 1, L - 1, j]) * il, g1)
            tL += x * d_mc((P[b, 2, L, j] - P[b, 2, L - 1, j]) * il, g2)
        rR = P[b, 0, Rr, j]
        nR = P[b, 1, Rr, j]
        tR = P[b, 2, Rr, j]
        if Rr <= nx - 2 and aL == 1.0 and aR == 1.0 and alpha[b, Rr + 1, j] == 1.0:
            ir = idxc[Rr]
            x = hx[Rr]
            rR -= x * d_mc(g0, (P[b, 0, Rr + 1, j] - P[b, 0, Rr, j]) * ir)
            nR -= x * d_mc(g1, (P[b, 1, Rr + 1, j] - P[b, 1, Rr, j]) * ir)
            tR -= x * d_mc(g2, (P[b, 2, Rr + 1, j] - P[b, 2, Rr, j]) * ir)
        f0, f1, f2 = d_rusanov(rL, nL, tL, P[b, 4, L, j], P[b, 5, L, j],
                               rR, nR, tR, P[b, 4, Rr, j], P[b, 5, Rr, j], rho0, c0sq, g1h)
        f1 -= mu * g1
        f2 -= mu * g2
        l0 = bf * f0
        l1 = bf * f1
        l2 = bf * f2
        r0 = l0
        r1 = l1
        r2 = l2
    wl = fL - beta
    if wl > 1e-12:  # wall on +x side of the left cell
        w = float32(wl)
        ii = i - 1
        un = P[b, 1, ii, j]
        pw = P[b, 3, ii, j] + (rho0 + P[b, 0, ii, j]) * P[b, 4, ii, j] * un
        if i == nx:
            l1 += w * pw
        else:
            wt = Z
            if isr[i] != 0 and alpha[b, i, j] < 1.0:
                wt = float32(colV[b, stage, i])
            k = TWO * mu * idx_[ii]
            l1 += w * (pw + k * un)
            l2 += w * (k * (P[b, 2, ii, j] - wt))
    wr = fR - beta
    if wr > 1e-12:  # wall on -x side of the right cell
        w = float32(wr)
        un = P[b, 1, i, j]
        pw = P[b, 3, i, j] - (rho0 + P[b, 0, i, j]) * P[b, 4, i, j] * un
        if i == 0:
            r1 += w * pw
        else:
            wt = Z
            if isr[i - 1] != 0 and alpha[b, i - 1, j] < 1.0:
                wt = float32(colV[b, stage, i - 1])
            k = TWO * mu * idx_[i]
            r1 += w * (pw - k * un)
            r2 -= w * k * (P[b, 2, i, j] - wt)
    FL[b, 0, i, j] = l0
    FL[b, 1, i, j] = l1
    FL[b, 2, i, j] = l2
    FR[b, 0, i, j] = r0
    FR[b, 1, i, j] = r1
    FR[b, 2, i, j] = r2


@cuda.jit(device=True, inline=True)
def d_yflux_s(P, alpha, yf, idy, idyc, hy, b, i, j, ny, closed, rho0, c0sq, g1h, mu):
    b0 = Z
    b1 = Z
    b2 = Z
    t0 = Z
    t1 = Z
    t2 = Z
    if not closed:
        aB = alpha[b, i, j - 1] if j > 0 else 0.0
        aT = alpha[b, i, j] if j < ny else 0.0
        if aB > 0.0 and aT > 0.0:
            B = j - 1
            T = j
            ic = idyc[B]
            g0 = (P[b, 0, i, T] - P[b, 0, i, B]) * ic
            g1 = (P[b, 1, i, T] - P[b, 1, i, B]) * ic
            g2 = (P[b, 2, i, T] - P[b, 2, i, B]) * ic
            rB = P[b, 0, i, B]
            tB = P[b, 1, i, B]
            nB = P[b, 2, i, B]
            if B >= 1 and aB == 1.0 and aT == 1.0 and alpha[b, i, B - 1] == 1.0:
                il = idyc[B - 1]
                x = hy[B]
                rB += x * d_mc((P[b, 0, i, B] - P[b, 0, i, B - 1]) * il, g0)
                tB += x * d_mc((P[b, 1, i, B] - P[b, 1, i, B - 1]) * il, g1)
                nB += x * d_mc((P[b, 2, i, B] - P[b, 2, i, B - 1]) * il, g2)
            rT = P[b, 0, i, T]
            tT = P[b, 1, i, T]
            nT = P[b, 2, i, T]
            if T <= ny - 2 and aB == 1.0 and aT == 1.0 and alpha[b, i, T + 1] == 1.0:
                ir = idyc[T]
                x = hy[T]
                rT -= x * d_mc(g0, (P[b, 0, i, T + 1] - P[b, 0, i, T]) * ir)
                tT -= x * d_mc(g1, (P[b, 1, i, T + 1] - P[b, 1, i, T]) * ir)
                nT -= x * d_mc(g2, (P[b, 2, i, T + 1] - P[b, 2, i, T]) * ir)
            f0, f1, f2 = d_rusanov(rB, nB, tB, P[b, 4, i, B], P[b, 5, i, B],
                                   rT, nT, tT, P[b, 4, i, T], P[b, 5, i, T], rho0, c0sq, g1h)
            b0 = f0
            b1 = f2 - mu * g1
            b2 = f1 - mu * g2
            t0 = b0
            t1 = b1
            t2 = b2
        elif aB > 0.0:
            jj = j - 1
            un = P[b, 2, i, jj]
            pw = P[b, 3, i, jj] + (rho0 + P[b, 0, i, jj]) * P[b, 4, i, jj] * un
            if j == ny:
                b2 = pw
            else:
                k = TWO * mu * idy[jj]
                b2 = pw + k * un
                b1 = k * P[b, 1, i, jj]
        elif aT > 0.0:
            un = P[b, 2, i, j]
            pw = P[b, 3, i, j] - (rho0 + P[b, 0, i, j]) * P[b, 4, i, j] * un
            if j == 0:
                t2 = pw
            else:
                k = TWO * mu * idy[j]
                t2 = pw - k * un
                t1 = -k * P[b, 1, i, j]
    return b0, b1, b2, t0, t1, t2


@cuda.jit
def g_update(P, Pn, Ucur, Uold, Unew, ca, cb, cc, dt, FL, FR, alpha, isr, colY, colJ, h, yf, hy32,
             idy32, idyc32, idx, idy, sig, sigm, is_up, bpar, bp32, ist, mu, jpb, qslot, stage, sig_fix):
    j, i, b = cuda.grid(3)
    nb, nx, ny = alpha.shape
    if j >= ny or i >= nx or b >= nb:
        return
    rho0 = bpar[b, 0]
    c0sq = bpar[b, 2]
    g1h = bpar[b, 3]
    rho0f = bp32[b, 0]
    c0sqf = bp32[b, 1]
    g1hf = bp32[b, 2]
    reed = isr[i] != 0
    Yb = 0.0
    Yt = 0.0
    jt = -100
    jb = -100
    if reed:
        Yb = colY[b, stage, i]
        Yt = Yb + h
        jt = colJ[b, 2 * stage, i]
        jb = colJ[b, 2 * stage + 1, i]
    closed0 = reed and yf[j] >= Yb and yf[j] <= Yt
    closed1 = reed and yf[j + 1] >= Yb and yf[j + 1] <= Yt
    _, _, _, ft0, ft1, ft2 = d_yflux_s(P, alpha, yf, idy32, idyc32, hy32, b, i, j, ny, closed0,
                                       rho0f, c0sqf, g1hf, mu)
    b0, b1, b2, t0, _, _ = d_yflux_s(P, alpha, yf, idy32, idyc32, hy32, b, i, j + 1, ny, closed1,
                                     rho0f, c0sqf, g1hf, mu)
    if j + 1 == jpb:
        qslot[b, i] = t0
    if alpha[b, i, j] > 0.0:
        ix = idx[i]
        iy = idy[j]
        q0 = -float64(FL[b, 0, i + 1, j] - FR[b, 0, i, j]) * ix - float64(b0 - ft0) * iy
        q1 = -float64(FL[b, 1, i + 1, j] - FR[b, 1, i, j]) * ix - float64(b1 - ft1) * iy
        q2 = -float64(FL[b, 2, i + 1, j] - FR[b, 2, i, j]) * ix - float64(b2 - ft2) * iy
        s = sig[i, j]
        if s > 0.0 and alpha[b, i, j] == 1.0:
            rt = bpar[b, 4] if is_up[i, j] else rho0
            q0 += s * (rt - Ucur[b, 0, i, j])
            sm = sigm[i, j]
            q1 -= sm * Ucur[b, 1, i, j]
            q2 -= sm * Ucur[b, 2, i, j]
        cdt = cc * dt
        Unew[b, 0, i, j] = ca * Uold[b, 0, i, j] + cb * Ucur[b, 0, i, j] + cdt * q0
        Unew[b, 1, i, j] = ca * Uold[b, 1, i, j] + cb * Ucur[b, 1, i, j] + cdt * q1
        Unew[b, 2, i, j] = ca * Uold[b, 2, i, j] + cb * Ucur[b, 2, i, j] + cdt * q2
        if s >= sig_fix and alpha[b, i, j] == 1.0:
            # outer band of the sponge: hold the reservoir density exactly (constant-pressure
            # bellows / room), so the interior sees the true dp minus genuine flow losses
            Unew[b, 0, i, j] = bpar[b, 4] if is_up[i, j] else rho0
    else:
        Unew[b, 0, i, j] = 0.0
        Unew[b, 1, i, j] = 0.0
        Unew[b, 2, i, j] = 0.0
    if not (reed and j >= jb - 2 and j <= jt + 2):
        if not d_prim(Unew, alpha, Pn, b, i, j, rho0, c0sq, g1h, 0, 0.0):
            ist[b, 0] = 1


NT_REED = 256


@cuda.jit
def g_reed(P, Pn, Unew, alpha, isr, colY, colV, colJ, leak, mq, w0, phi, dxc, xr0, nr,
           modal, h, bw, tp, lpar, yf, dy, bpar, ist, mu, dt, stage,
           rec, ctr, rec_every, mic_i, mic_j, mic_w):
    """One block per batch member. Reed columns are i = xr0 .. xr0+nr-1."""
    b = cuda.blockIdx.x
    tid = cuda.threadIdx.x
    nt = cuda.blockDim.x
    ny = dy.shape[0]
    nm = phi.shape[0]
    sQ = cuda.shared.array((4, NT_REED), float64)
    sL = cuda.shared.array(NT_REED, float64)
    sM = cuda.shared.array(8, float64)
    nstep = ctr[0]
    rho0 = bpar[b, 0]
    p0 = bpar[b, 1]
    c0sq = bpar[b, 2]
    g1h = bpar[b, 3]
    cdt = dt if stage == 0 else 0.5 * dt
    so = 1 if stage == 0 else 0
    gap = lpar[b, 0]
    Cd = lpar[b, 1]
    jf = lpar[b, 2]
    for n in range(4):
        sQ[n, tid] = 0.0
    sL[tid] = 0.0
    # ---------- A: forces on the tongue, side-gap leak ----------
    for k in range(tid, nr, nt):
        i = xr0 + k
        Yb = colY[b, stage, i]
        Yt = Yb + h
        V = colV[b, stage, i]
        jt = colJ[b, 2 * stage, i]
        jb = colJ[b, 2 * stage + 1, i]
        # top face
        hd = 0.5 * dy[jt]
        rt = rho0 + float64(P[b, 0, i, jt])
        vv = float64(P[b, 2, i, jt])
        pwt = float64(P[b, 3, i, jt]) + rt * float64(P[b, 4, i, jt]) * (V - vv)
        fw = pwt - mu * (vv - V) / hd
        Unew[b, 2, i, jt] += cdt * fw / dy[jt]
        Unew[b, 1, i, jt] -= cdt * mu * float64(P[b, 1, i, jt]) / hd / dy[jt]
        f = -fw
        # bottom face
        hd = 0.5 * dy[jb]
        rb = rho0 + float64(P[b, 0, i, jb])
        vv = float64(P[b, 2, i, jb])
        pwb = float64(P[b, 3, i, jb]) + rb * float64(P[b, 4, i, jb]) * (vv - V)
        fw = pwb + mu * (vv - V) / hd
        Unew[b, 2, i, jb] -= cdt * fw / dy[jb]
        Unew[b, 1, i, jb] -= cdt * mu * float64(P[b, 1, i, jb]) / hd / dy[jb]
        f += fw
        f *= bw                                   # force per unit tongue length [N/m]
        for n in range(nm):
            sQ[n, tid] += f * phi[n, k] * dxc[k]
        # side-gap leak (sub-grid): per side opening a, overlap l_v
        d = max(0.0, Yb) + max(0.0, -tp - Yt)
        a = math.sqrt(gap * gap + d * d)
        lv = max(0.0, min(Yt, 0.0) - max(Yb, -tp))
        leff = lv + a
        U = leak[b, stage, i]
        dpl = float64(P[b, 3, i, jt]) - float64(P[b, 3, i, jb])
        rm = 0.5 * (rt + rb)
        dUdt = (dpl - rm * U * abs(U) / (2.0 * Cd * Cd) - 12.0 * mu * lv * U / (a * a)) / (rm * leff)
        if stage == 0:
            leak[b, 1, i] = U + dt * dUdt
        else:
            leak[b, 0, i] = 0.5 * (leak[b, 0, i] + U + dt * dUdt)
        # mass moved per unit time per unit tongue length, per unit slab depth
        if U > 0.0:
            jd = jt
            rd = rt
        else:
            jd = jb
            rd = rb
        mflux = rd * 2.0 * a * U / bw
        ud = float64(P[b, 1, i, jd])
        vd = float64(P[b, 2, i, jd])
        # donor loses its own momentum; the receiver gets the leaked mass at the jet
        # (vena-contracta) velocity U/Cd from the gap equation, directed along the slot
        vj = jf * (-U / Cd) + (1.0 - jf) * vd
        if U > 0.0:
            vt = vd
            vbm = vj
        else:
            vt = vj
            vbm = vd
        Unew[b, 0, i, jt] -= cdt * mflux / dy[jt]
        Unew[b, 1, i, jt] -= cdt * mflux * ud / dy[jt]
        Unew[b, 2, i, jt] -= cdt * mflux * vt / dy[jt]
        Unew[b, 0, i, jb] += cdt * mflux / dy[jb]
        Unew[b, 1, i, jb] += cdt * mflux * ud / dy[jb]
        Unew[b, 2, i, jb] += cdt * mflux * vbm / dy[jb]
        sL[tid] += 2.0 * a * U * dxc[k]           # leak volume flow [m^3/s]
    cuda.syncthreads()
    # ---------- B: modal ODE (thread 0) ----------
    if tid == 0:
        for n in range(nm):
            Qn = 0.0
            for k in range(nt):
                Qn += sQ[n, k]
            M = modal[b, n, 0]
            K = modal[b, n, 1]
            C = modal[b, n, 2]
            if stage == 0:
                q = mq[b, 0, n]
                qd = mq[b, 1, n]
                A = (Qn - C * qd - K * q) / M
                mq[b, 2, n] = q + dt * qd
                mq[b, 3, n] = qd + dt * A
                sM[2 * n] = mq[b, 2, n]
                sM[2 * n + 1] = mq[b, 3, n]
            else:
                q = mq[b, 2, n]
                qd = mq[b, 3, n]
                A = (Qn - C * qd - K * q) / M
                mq[b, 0, n] = 0.5 * (mq[b, 0, n] + q + dt * qd)
                mq[b, 1, n] = 0.5 * (mq[b, 1, n] + qd + dt * A)
                sM[2 * n] = mq[b, 0, n]
                sM[2 * n + 1] = mq[b, 1, n]
            if n == 0:
                sM[7] = Qn
        if stage == 0 and (nstep % rec_every) == 0:
            irec = nstep // rec_every
            if irec < rec.shape[1]:
                ytip = 0.0
                vtip = 0.0
                for n in range(nm):
                    ytip += mq[b, 0, n]
                    vtip += mq[b, 1, n]
                lk = 0.0
                for k in range(nt):
                    lk += sL[k]
                rec[b, irec, 0] = nstep * dt
                rec[b, irec, 1] = ytip
                rec[b, irec, 2] = vtip
                rec[b, irec, 3] = sM[7]
                rec[b, irec, 5] = lk
                for n in range(nm):
                    rec[b, irec, 6 + n] = mq[b, 0, n]
                for m in range(mic_w.shape[0]):
                    s = 0.0
                    for q in range(4):
                        s += mic_w[m, q] * float64(P[b, 3, mic_i[m, q], mic_j[m, q]])
                    rec[b, irec, 10 + m] = p0 + s
    cuda.syncthreads()
    # ---------- C: new column heights, cut cells, merging, primitives ----------
    for k in range(tid, nr, nt):
        i = xr0 + k
        Yn = w0[b, k]
        Vn = 0.0
        for n in range(nm):
            Yn += sM[2 * n] * phi[n, k]
            Vn += sM[2 * n + 1] * phi[n, k]
        colY[b, so, i] = Yn
        colV[b, so, i] = Vn
        jt = colJ[b, 2 * stage, i]
        jb = colJ[b, 2 * stage + 1, i]
        jbn = d_search(yf, Yn, False) - 1
        jtn = d_search(yf, Yn + h, True) - 1
        colJ[b, 2 * so, i] = jtn
        colJ[b, 2 * so + 1, i] = jbn
        Ytn = Yn + h
        j0 = max(jbn - 3, 0)
        j1 = min(jtn + 4, ny)
        for j in range(j0, j1):
            if j > jbn and j < jtn:
                a = 0.0
            else:
                ov = min(yf[j + 1], Ytn) - max(yf[j], Yn)
                a = 1.0 - ov / dy[j] if ov > 0.0 else 1.0
                if a < 0.0:
                    a = 0.0
            alpha[b, i, j] = a
        ymid = Yn + 0.5 * h
        for j in range(jbn + 1, jtn):
            if Unew[b, 0, i, j] != 0.0 or Unew[b, 1, i, j] != 0.0 or Unew[b, 2, i, j] != 0.0:
                tgt = jtn if 0.5 * (yf[j] + yf[j + 1]) > ymid else jbn
                fct = dy[j] / dy[tgt]
                for q in range(3):
                    Unew[b, q, i, tgt] += Unew[b, q, i, j] * fct
                    Unew[b, q, i, j] = 0.0
        if alpha[b, i, jtn] < 0.5:
            d_mix(Unew, alpha, dy, b, i, jtn, jtn + 1)
        if alpha[b, i, jbn] < 0.5:
            d_mix(Unew, alpha, dy, b, i, jbn, jbn - 1)
        r0 = max(min(jb, jbn) - 4, 0)
        r1 = min(max(jt, jtn) + 5, ny)
        for j in range(r0, r1):
            if not d_prim(Unew, alpha, Pn, b, i, j, rho0, c0sq, g1h, 0, 0.0):
                ist[b, 0] = 1
    if stage == 1 and b == 0:
        cuda.syncthreads()
        if tid == 0:
            ctr[0] = nstep + 1


@cuda.jit
def g_slotflux(qslot, dxc_all, rec, ctr, rec_every, rho0s, bw, n0, n1):
    """record the volume flow through the plate underside (slot exit)."""
    b = cuda.grid(1)
    if b >= qslot.shape[0]:
        return
    nstep = ctr[0]
    if (nstep % rec_every) == 0:
        irec = nstep // rec_every
        if irec < rec.shape[1]:
            s = 0.0
            for i in range(n0, n1):
                s += qslot[b, i] * dxc_all[i]
            rec[b, irec, 4] = s * bw / rho0s[b]


@cuda.jit
def g_diag(P, alpha, D, idx, idy, nu):
    """running max of cell-Reynolds number |u| d / nu and of the normalised
    velocity jump to the neighbours -- a resolution map for regridding."""
    j, i, b = cuda.grid(3)
    nb, nx, ny = alpha.shape
    if j >= ny or i >= nx or b >= nb or alpha[b, i, j] < 1.0:
        return
    u = float64(P[b, 1, i, j])
    v = float64(P[b, 2, i, j])
    d = 1.0 / max(idx[i], idy[j])
    re = math.sqrt(u * u + v * v) * d / nu[b]
    if re > D[b, 0, i, j]:
        D[b, 0, i, j] = re
    jmp = 0.0
    if i + 1 < nx and alpha[b, i + 1, j] == 1.0:
        jmp = max(jmp, abs(float64(P[b, 1, i + 1, j]) - u) + abs(float64(P[b, 2, i + 1, j]) - v))
    if j + 1 < ny and alpha[b, i, j + 1] == 1.0:
        jmp = max(jmp, abs(float64(P[b, 1, i, j + 1]) - u) + abs(float64(P[b, 2, i, j + 1]) - v))
    if jmp > D[b, 1, i, j]:
        D[b, 1, i, j] = jmp


# ----------------------------------------------------------------------------
# host side
# ----------------------------------------------------------------------------
def column_geometry(Yb, h, yf, dy):
    """fluid fractions of one tongue column and its top/bottom cut-cell rows."""
    Yt = Yb + h
    jb = int(np.searchsorted(yf, Yb, side="left")) - 1
    jt = int(np.searchsorted(yf, Yt, side="right")) - 1
    a = np.ones(len(dy))
    ov = np.minimum(yf[1:], Yt) - np.maximum(yf[:-1], Yb)
    m = ov > 0
    a[m] = np.clip(1.0 - ov[m] / dy[m], 0.0, 1.0)
    a[jb + 1:jt] = 0.0
    return a, jt, jb


class SideBatch:
    TPB = (32, 4, 1)

    def __init__(self, cfgs):
        self.cfgs = cfgs
        C = cfgs[0]
        for c in cfgs[1:]:
            for k in ("reed_L", "reed_b", "reed_h", "plate_t", "gap_tip", "dx_body", "dx_tip", "dy_fine", "grid_file",
                      "chamber", "chamber_h", "x_lo", "y_top", "mu", "n_modes", "rec_every"):
                assert getattr(c, k) == getattr(C, k), f"batch members must share {k}"
        self.nb = nb = len(cfgs)
        L, h, b_w = C.reed_L, C.reed_h, C.reed_b
        dy0 = C.dy_fine
        tp = round(C.plate_t / dy0) * dy0
        self.tp, self.L, self.h, self.bw = tp, L, h, b_w
        gt = C.gap_tip
        xk = [(C.x_lo, C.d_coarse), (-3e-3, C.dx_body), (0.0, C.dx_body), (L - C.tip_zone, C.dx_body),
              (L - 0.1e-3, C.dx_tip), (L, C.dx_tip), (L + gt, C.dx_tip), (L + gt + 0.1e-3, C.dx_tip),
              (L + gt + C.tip_zone, C.dx_body), (L + 2e-3, C.dx_body), (L + C.x_hi_extra, C.d_coarse)]
        self.xf = axis_from_knots(xk, C.stretch)
        H_c = C.chamber_h
        yb_ch = -tp - H_c
        y_bot = (yb_ch - C.wall_t if C.chamber else -tp) - C.room_below
        yk = [(y_bot, C.d_coarse), (-tp - C.core_dn, dy0), (-tp, dy0), (0.0, dy0), (C.core_up, dy0),
              (C.y_top, C.d_coarse)]
        self.yf = axis_from_knots(yk, C.stretch)
        if C.grid_file:
            gfile = np.load(C.grid_file)
            self.xf, self.yf = gfile["xf"], gfile["yf"]
            for x in (0.0, L, L + gt):
                assert np.min(abs(self.xf - x)) < 1e-12, "grid must contain x = 0, L, L+gap_tip"
            for y in (0.0, -tp):
                assert np.min(abs(self.yf - y)) < 1e-12, "grid must contain y = 0, -t_p"
        xf, yf = self.xf, self.yf
        self.xc, self.yc = 0.5 * (xf[1:] + xf[:-1]), 0.5 * (yf[1:] + yf[:-1])
        self.dx, self.dy = np.diff(xf), np.diff(yf)
        nx, ny = len(self.xc), len(self.yc)
        self.nx, self.ny = nx, ny
        self.jpb = int(np.argmin(abs(yf + tp)))
        assert abs(yf[self.jpb] + tp) < 1e-12
        X, Yc = np.meshgrid(self.xc, self.yc, indexing="ij")
        slot = (X > 0) & (X < L + gt)
        solid = (Yc > -tp) & (Yc < 0) & ~slot
        solid |= (X > -C.heel) & (X < 0) & (Yc > 0) & (Yc < h)
        if C.chamber:
            x0, x1, wt = C.ch_x0, L + gt + C.ch_x1_extra, C.wall_t
            o0 = L + C.outlet_x0_extra
            ywall = (Yc > yb_ch - wt) & (Yc < -tp)
            solid |= ywall & (((X > x0 - wt) & (X < x0)) | ((X > x1) & (X < x1 + wt)))
            solid |= (X > x0 - wt) & (X < x1 + wt) & (Yc > yb_ch - wt) & (Yc < yb_ch) & \
                ~((X > o0) & (X < o0 + C.outlet_w))
            self.outlet = (o0, o0 + C.outlet_w, yb_ch - wt)
        self.solid = solid
        isr = (self.xc > 0) & (self.xc < L)
        self.xr0 = int(np.argmax(isr))
        self.nr = int(isr.sum())
        assert np.all(isr[self.xr0:self.xr0 + self.nr])
        xr = self.xc[isr]
        dxr = self.dx[isr]
        assert abs(xf[self.xr0] - 0) < 1e-12 and abs(xf[self.xr0 + self.nr] - L) < 1e-12
        nm = C.n_modes
        phi = cantilever_modes(xr, L, nm)
        EI = C.E * b_w * h ** 3 / 12.0
        mlin = C.rho_s * b_w * h
        om = np.array([(BETAL[n] / L) ** 2 * math.sqrt(EI / mlin) for n in range(nm)])
        Mn = mlin * (phi ** 2 * dxr).sum(axis=1)
        self.f_modes = om / (2 * math.pi)
        self.f_vac = self.f_modes[0]
        self.Mn = Mn
        modal = np.zeros((nb, nm, 3))
        w0 = np.zeros((nb, self.nr))
        mq = np.zeros((nb, 4, nm))
        for bi, c in enumerate(cfgs):
            modal[bi, :, 0] = Mn
            modal[bi, :, 1] = Mn * om ** 2
            modal[bi, :, 2] = Mn * om / c.Q_mech
            w0[bi] = c.set_tip * (xr / L) ** c.set_exp
            mq[bi, 0, 0] = c.kick
            mq[bi, 2, 0] = c.kick
        # fluid state
        self.rho0 = np.array([c.rho0() for c in cfgs])
        c0sq = np.array([c.c0() ** 2 for c in cfgs])
        g1h = np.array([0.5 * (c.gamma - 1) / r for c, r in zip(cfgs, self.rho0)])
        rho_up = np.array([r * ((c.p_amb + c.dp) / c.p_amb) ** (1 / c.gamma) for c, r in zip(cfgs, self.rho0)])
        self.rho_up = rho_up
        alpha = np.where(solid, 0.0, 1.0)[None].repeat(nb, 0)
        colY = np.zeros((nb, 2, nx))
        colJ = np.zeros((nb, 4, nx), np.int32)
        U = np.zeros((nb, 3, nx, ny))
        for bi, c in enumerate(cfgs):
            Ycol = w0[bi] + mq[bi, 0, :nm] @ phi
            for k in range(self.nr):
                i = self.xr0 + k
                a, jt, jb = column_geometry(Ycol[k], h, yf, self.dy)
                alpha[bi, i] = np.where(solid[i], 0.0, a)
                colY[bi, :, i] = Ycol[k]
                colJ[bi, :, i] = (jt, jb, jt, jb)
            rinit = self.rho0[bi] + (rho_up[bi] - self.rho0[bi]) * (Yc > 0) * (c.t_ramp <= 0)
            U[bi, 0] = alpha[bi] * rinit
        # sponge
        def ramp(d):
            return np.clip((C.sponge - d) / C.sponge, 0, 1) ** 2
        sig = np.maximum.reduce([ramp(C.y_top - Yc), ramp(Yc - yf[0]), ramp(X - xf[0]), ramp(xf[-1] - X)])
        sig = C.sponge_sigma * sig
        is_up = Yc > 0
        sigm = np.where(is_up, 0.0, C.sponge_mom * sig)
        # mics
        if not C.mics:
            o0, o1, yfl = self.outlet if C.chamber else (L, L + 5e-3, -tp)
            C.mics = [["chamber", 0.5 * L, -tp - 0.5 * H_c], ["outlet", 0.5 * (o0 + o1), yfl - 4e-3],
                      ["room", 0.5 * (o0 + o1) + 8e-3, yfl - 7e-3], ["bellows", 0.5 * L, 5e-3]]
        self.mic_names = [m[0] for m in C.mics]
        nmic = len(C.mics)
        mi, mj, mw = np.zeros((nmic, 4), np.int64), np.zeros((nmic, 4), np.int64), np.zeros((nmic, 4))
        for k, (_, xm, ym) in enumerate(C.mics):
            i1 = int(np.clip(np.searchsorted(self.xc, xm), 1, nx - 1))
            j1 = int(np.clip(np.searchsorted(self.yc, ym), 1, ny - 1))
            i0, j0 = i1 - 1, j1 - 1
            wx = np.clip((xm - self.xc[i0]) / (self.xc[i1] - self.xc[i0]), 0, 1)
            wy = np.clip((ym - self.yc[j0]) / (self.yc[j1] - self.yc[j0]), 0, 1)
            ii, jj = [i0, i1, i0, i1], [j0, j0, j1, j1]
            if any(solid[a, q] for a, q in zip(ii, jj)) or any(isr[a] for a in ii) and -3e-3 < ym < 3e-3:
                raise ValueError(f"mic {C.mics[k][0]} is in/next to a solid or the tongue")
            mi[k], mj[k], mw[k] = ii, jj, [(1 - wx) * (1 - wy), wx * (1 - wy), (1 - wx) * wy, wx * wy]
        # time step
        cmax = math.sqrt(c0sq.max() * (1 + 2 * (g1h * (rho_up - self.rho0)).max()))
        u_est = 1.4 * math.sqrt(2 * max(c.dp for c in cfgs) / self.rho0.min()) + 30.0
        self.dt = C.cfl / ((cmax + u_est) * (1 / self.dx.min() + 1 / self.dy.min()))
        self.t_end = max(c.t_end for c in cfgs)
        self.nsteps = int(math.ceil(self.t_end / self.dt))
        self.rec_every = C.rec_every
        nrec = (self.nsteps - 1) // self.rec_every + 1
        # device arrays
        g = lambda a: cuda.to_device(np.ascontiguousarray(a))
        self.U, self.U1 = g(U), g(np.zeros_like(U))
        self.alpha = g(alpha)
        self.P = g(np.zeros((nb, 6, nx, ny), f32))
        self.Pn = g(np.zeros((nb, 6, nx, ny), f32))
        self.FL = g(np.zeros((nb, 3, nx + 1, ny), f32))
        self.FR = g(np.zeros((nb, 3, nx + 1, ny), f32))
        self.qslot = g(np.zeros((nb, nx)))
        self.D = g(np.zeros((nb, 2, nx, ny)))
        self.isr = g(isr.astype(np.uint8))
        self.colY, self.colV, self.colJ = g(colY), g(np.zeros((nb, 2, nx))), g(colJ)
        self.leak = g(np.zeros((nb, 2, nx)))
        self.mq = g(mq)
        self.w0, self.phi, self.dxr, self.modal = g(w0), g(phi), g(dxr), g(modal)
        self.hx32, self.idx32, self.idxc32 = g((0.5 * self.dx).astype(f32)), g((1 / self.dx).astype(f32)), \
            g((1 / np.diff(self.xc)).astype(f32))
        self.hy32, self.idy32, self.idyc32 = g((0.5 * self.dy).astype(f32)), g((1 / self.dy).astype(f32)), \
            g((1 / np.diff(self.yc)).astype(f32))
        self.idx, self.idy = g(1 / self.dx), g(1 / self.dy)
        self.yf_d, self.dy_d, self.dx_d = g(yf), g(self.dy), g(self.dx)
        self.sig, self.sigm, self.is_up = g(sig), g(sigm), g(is_up.astype(np.uint8))
        self.sig_fix = 0.5 * C.sponge_sigma
        self.bpar_h = np.stack([self.rho0, [c.p_amb for c in cfgs], c0sq, g1h, rho_up], 1)
        self.bpar = g(self.bpar_h)
        self.bp32 = g(np.stack([self.rho0, c0sq, g1h], 1).astype(f32))
        self.lpar = g(np.array([[c.gap, c.Cd, c.jet] for c in cfgs]))
        self.ist = g(np.zeros((nb, 1), np.int32))
        self.rec = g(np.zeros((nb, nrec, 10 + nmic)))
        self.mic = (g(mi), g(mj), g(mw))
        self.nu = g(np.array([c.mu / r for c, r in zip(cfgs, self.rho0)]))
        self.ctr = g(np.zeros(1, np.int64))
        self.rho0_d = g(self.rho0)
        self.mu, self.mu32 = C.mu, f32(C.mu)
        self.s = cuda.stream()
        self._graphs = {}
        t = self.TPB
        self.grid_x = ((ny + t[0] - 1) // t[0], (nx + 1 + t[1] - 1) // t[1], nb)
        self.grid_c = ((ny + t[0] - 1) // t[0], (nx + t[1] - 1) // t[1], nb)
        g_prims_all[self.grid_c, self.TPB, self.s](self.U, self.alpha, self.P, self.bpar, 0, 0.0)
        self.s.synchronize()

    def info(self):
        C = self.cfgs[0]
        return dict(nx=self.nx, ny=self.ny, cells=self.nx * self.ny, dt=self.dt, nsteps=self.nsteps,
                    reed_columns=self.nr, f_modes=self.f_modes.tolist(), f_vacuum=self.f_vac,
                    plate_t_eff=self.tp, modal_mass=self.Mn.tolist())

    def step(self):
        s, dt, C = self.s, self.dt, self.cfgs[0]
        P, Pn = self.P, self.Pn
        for stage in (0, 1):
            if stage == 0:
                Ucur, Uold, Unew, ca, cb, cc = self.U, self.U, self.U1, 0.0, 1.0, 1.0
            else:
                Ucur, Uold, Unew, ca, cb, cc = self.U1, self.U, self.U, 0.5, 0.5, 0.5
            g_flux_x[self.grid_x, self.TPB, s](P, self.alpha, self.isr, self.colY, self.colV, self.h,
                                               self.yf_d, self.hx32, self.idx32, self.idxc32,
                                               self.FL, self.FR, self.bp32, self.mu32, stage)
            g_update[self.grid_c, self.TPB, s](P, Pn, Ucur, Uold, Unew, ca, cb, cc, dt, self.FL, self.FR,
                                               self.alpha, self.isr, self.colY, self.colJ, self.h,
                                               self.yf_d, self.hy32, self.idy32, self.idyc32,
                                               self.idx, self.idy, self.sig, self.sigm, self.is_up,
                                               self.bpar, self.bp32, self.ist, self.mu32, self.jpb,
                                               self.qslot, stage, self.sig_fix)
            if stage == 0:
                g_slotflux[1, 32, s](self.qslot, self.dx_d, self.rec, self.ctr, self.rec_every,
                                     self.rho0_d, self.bw, self.xr0, self.xr0 + self.nr + 2)
            g_reed[self.nb, NT_REED, s](P, Pn, Unew, self.alpha, self.isr, self.colY, self.colV, self.colJ,
                                        self.leak, self.mq, self.w0, self.phi, self.dxr, self.xr0, self.nr,
                                        self.modal, self.h, self.bw, self.tp, self.lpar, self.yf_d,
                                        self.dy_d, self.bpar, self.ist, self.mu, dt, stage, self.rec,
                                        self.ctr, self.rec_every, *self.mic)
            P, Pn = Pn, P

    def steps(self, count, graph_len=200):
        from cuda.bindings import driver as drv
        hs = self.s.handle
        cs = drv.CUstream(hs.value if hasattr(hs, "value") else int(hs))
        if count >= graph_len and graph_len not in self._graphs:
            self.step()
            self.s.synchronize()
            count -= 1
            err, = drv.cuStreamBeginCapture(cs, drv.CUstreamCaptureMode.CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
            assert err == drv.CUresult.CUDA_SUCCESS, err
            for _ in range(graph_len):
                self.step()
            err, graph = drv.cuStreamEndCapture(cs)
            assert err == drv.CUresult.CUDA_SUCCESS, err
            err, ge = drv.cuGraphInstantiate(graph, 0)
            assert err == drv.CUresult.CUDA_SUCCESS, err
            self._graphs[graph_len] = (graph, ge)
        while count >= graph_len:
            drv.cuGraphLaunch(self._graphs[graph_len][1], cs)
            count -= graph_len
        for _ in range(count):
            self.step()

    def diag(self):
        g_diag[self.grid_c, self.TPB, self.s](self.P, self.alpha, self.D, self.idx, self.idy, self.nu)

    def run(self, out_root, verbose=True, chunk=200):
        C = self.cfgs[0]
        T0 = 1.0 / self.f_vac
        snap_dt = T0 / C.snap_per_cycle
        next_snap = self.t_end - C.snap_cycles * T0
        snaps = [[] for _ in range(self.nb)]
        wall0 = last = time.time()
        n = 0
        ramp = max(c.t_ramp for c in self.cfgs) > 0
        while n < self.nsteps:
            if ramp and n * self.dt <= max(c.t_ramp for c in self.cfgs) + 2 * chunk * self.dt:
                for bi, c in enumerate(self.cfgs):
                    f = min(1.0, n * self.dt / c.t_ramp) if c.t_ramp > 0 else 1.0
                    f = f * f * (3 - 2 * f)
                    self.bpar_h[bi, 4] = self.rho0[bi] * ((c.p_amb + c.dp * f) / c.p_amb) ** (1 / c.gamma)
                self.s.synchronize()
                self.bpar.copy_to_device(self.bpar_h)
            cnt = min(chunk, self.nsteps - n)
            self.steps(cnt)
            n += cnt
            if (n // chunk) % 20 == 0:
                self.s.synchronize()   # keep the host from queueing the whole run
            if n % C.diag_every < chunk:
                self.diag()
            t = n * self.dt
            if t >= next_snap and next_snap <= self.t_end:
                self.s.synchronize()
                Ph, ah, cy = self.P.copy_to_host(), self.alpha.copy_to_host(), self.colY.copy_to_host()
                for bi in range(self.nb):
                    snaps[bi].append(dict(t=t, Y=cy[bi, 0].copy(), p=Ph[bi, 3].copy(), u=Ph[bi, 1].copy(),
                                          v=Ph[bi, 2].copy(), alpha=ah[bi].astype(np.float32)))
                next_snap += snap_dt
            now = time.time()
            if (now - last > 15 or n >= self.nsteps):
                self.s.synchronize()
                if self.ist.copy_to_host().any():
                    bad = [self.cfgs[b].name for b in np.nonzero(self.ist.copy_to_host()[:, 0])[0]]
                    raise RuntimeError(f"blow-up at step {n} in {bad}")
                last = now
                if verbose:
                    mq = self.mq.copy_to_host()
                    rate = n / (now - wall0)
                    ys = " ".join(f"{1e3 * mq[bi, 0].sum():+.3f}" for bi in range(self.nb))
                    print(f"t={t * 1e3:8.3f} ms step {n}/{self.nsteps} {rate:.0f} steps/s "
                          f"eta {(self.nsteps - n) / rate:.0f}s  tip[mm]: {ys}", flush=True)
        self.s.synchronize()
        self.wall_time = time.time() - wall0
        return self.save(out_root, snaps)

    def save(self, out_root, snaps):
        rec = self.rec.copy_to_host()
        D = self.D.copy_to_host()
        dirs = []
        for bi, c in enumerate(self.cfgs):
            od = os.path.join(out_root, c.name)
            os.makedirs(od, exist_ok=True)
            cfg = asdict(c)
            cfg["set_gap"] = c.set_tip
            info = self.info()
            info["wall_time_s"] = getattr(self, "wall_time", None)
            info["model"] = "side-view"
            with open(os.path.join(od, "config.json"), "w") as f:
                json.dump(dict(config=cfg, info=info), f, indent=2)
            r = rec[bi]
            np.savez_compressed(
                os.path.join(od, "series.npz"),
                t=r[:, 0], Y=r[:, 1], V=r[:, 2], F=r[:, 3], Qslot=r[:, 4], Qleak=r[:, 5],
                q=r[:, 6:6 + c.n_modes], mics=r[:, 10:] - c.p_amb, mic_names=np.array(self.mic_names),
                mic_pos=np.array([[m[1], m[2]] for m in self.cfgs[0].mics]), f_vacuum=self.f_vac,
                f_modes=self.f_modes, p_amb=c.p_amb, dp=c.dp, rho0=self.rho0[bi], c0=c.c0(), reed_L=c.reed_L)
            np.savez_compressed(os.path.join(od, "diag.npz"), xf=self.xf, yf=self.yf, re_cell=D[bi, 0],
                                dvel=D[bi, 1], solid=self.solid)
            if snaps[bi]:
                sn = snaps[bi]
                np.savez_compressed(
                    os.path.join(od, "snapshots.npz"), side=True, zf=self.xf, yf=self.yf,
                    t=np.array([q["t"] for q in sn]), Y=np.array([q["Y"] for q in sn]), h=self.h,
                    b=self.bw, tp=self.tp, L=self.L, p=np.stack([q["p"] for q in sn]),
                    u=np.stack([q["u"] for q in sn]), v=np.stack([q["v"] for q in sn]),
                    alpha=np.stack([q["alpha"] for q in sn]), solid=self.solid)
            dirs.append(od)
        return dirs


def quiet_check(nsteps=2000):
    """no blowing, no kick: nothing may move."""
    sb = SideBatch([SideConfig(name="q", dp=0.0, kick=0.0, t_ramp=0.0, t_end=1.0)])
    m0 = sb.U.copy_to_host()[0, 0].sum()
    sb.steps(nsteps)
    sb.s.synchronize()
    P = sb.P.copy_to_host()[0]
    a = sb.alpha.copy_to_host()[0] > 0
    print(f"{sb.nx}x{sb.ny} cells, dt={sb.dt:.3e}, reed columns {sb.nr}, modes {sb.f_modes.round(1)} Hz")
    print("max|u|", np.abs(P[1][a]).max(), "max|v|", np.abs(P[2][a]).max(), "max|p'|", np.abs(P[3][a]).max(),
          "mass drift", sb.U.copy_to_host()[0, 0].sum() / m0 - 1, "q", sb.mq.copy_to_host()[0, 0])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    multi = {"p_amb", "dp", "kick", "Q_mech", "set_tip", "t_end", "Cd", "gap", "jet"}
    for k, v in asdict(SideConfig()).items():
        if k in ("mics", "name"):
            continue
        flag = "--" + k.replace("_", "-")
        if k in multi:
            ap.add_argument(flag, type=float, nargs="+", default=[v])
        elif isinstance(v, bool):
            ap.add_argument(flag, type=lambda s: s.lower() in ("1", "true", "yes"), default=v)
        else:
            ap.add_argument(flag, type=type(v), default=v)
    ap.add_argument("--prefix", default="side")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--grid", choices=["product", "zip"], default="product")
    ap.add_argument("--quiet-check", action="store_true")
    ap.add_argument("--no-analysis", action="store_true")
    a = ap.parse_args()
    if a.quiet_check:
        quiet_check()
        return
    base = {k: getattr(a, k) for k in asdict(SideConfig()) if k not in ("mics", "name") and k not in multi}
    lists = {k: getattr(a, k) for k in multi}
    keys = sorted(lists)
    if a.grid == "zip":
        n = max(len(v) for v in lists.values())
        combos = [{k: (lists[k][i] if len(lists[k]) > 1 else lists[k][0]) for k in keys} for i in range(n)]
    else:
        combos = [dict(zip(keys, vals)) for vals in itertools.product(*[lists[k] for k in keys])]
    cfgs = []
    for c in combos:
        tag = "_".join(f"{k}{c[k]:g}" for k in keys if len(lists[k]) > 1) or "run"
        cfgs.append(SideConfig(name=f"{a.prefix}_{tag}", **base, **c))
    sb = SideBatch(cfgs)
    print(json.dumps(sb.info()))
    print(f"batch of {sb.nb}: " + ", ".join(c.name for c in cfgs), flush=True)
    dirs = sb.run(a.out)
    print(f"wall time {sb.wall_time:.0f}s")
    if not a.no_analysis:
        from analysis import analyze_run
        for d in dirs:
            analyze_run(d)


if __name__ == "__main__":
    main()
