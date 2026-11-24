#!/usr/bin/env python3
"""
reed_gpu.py -- batched CUDA (numba-cuda) version of the reed2d solver.

Same numerics as reed2d.py (and the same Config / grid / output format), but
B simulations that share a grid -- e.g. an ambient-pressure sweep -- advance
together in one set of kernel launches. Each batch member has its own air
density, blowing pressure, reed state and fluid fractions.

    python reed_gpu.py --p-amb 70e3 85e3 101325 115e3 --dp 1000 --t-end 0.05 --prefix sw
    python reed_gpu.py --check      # compare against the CPU solver for a few hundred steps

Any reed2d Config field can be given (single value, or one value per run where
it makes sense: p_amb, dp, kick, Q_mech, set_gap, t_end).
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from dataclasses import asdict, replace

import numpy as np
from numba import cuda, float64, int32

import reed2d as R2

# ----------------------------------------------------------------------------
# device helpers (mirror reed2d.py)
# ----------------------------------------------------------------------------


@cuda.jit(device=True, inline=True)
def d_pres(r, rho0, p0, c0sq, g1h):
    d = r - rho0
    return p0 + c0sq * d * (1.0 + g1h * d)


@cuda.jit(device=True, inline=True)
def d_mc(a, b):
    m = min(2.0 * abs(a), 2.0 * abs(b), 0.5 * abs(a + b))
    return 0.5 * (math.copysign(1.0, a) + math.copysign(1.0, b)) * m


@cuda.jit(device=True, inline=True)
def d_rusanov(rL, nL, tL, cL, icL, rR, nR, tR, cR, icR, rho0, p0, c0sq, g1h):
    pL = d_pres(rL, rho0, p0, c0sq, g1h)
    pR = d_pres(rR, rho0, p0, c0sq, g1h)
    m2 = max((nL * nL + tL * tL) * icL * icL, (nR * nR + tR * tR) * icR * icR)
    zz = math.sqrt(m2) if m2 < 1.0 else 1.0
    nm = 0.5 * (nL + nR)
    dn = 0.5 * zz * (nL - nR)
    tm = 0.5 * (tL + tR)
    dtt = 0.5 * zz * (tL - tR)
    nL2 = nm + dn
    nR2 = nm - dn
    tL2 = tm + dtt
    tR2 = tm - dtt
    a = max(abs(nL), abs(nR)) + max(cL, cR)
    f0 = 0.5 * (rL * nL2 + rR * nR2) - 0.5 * a * (rR - rL)
    f1 = 0.5 * (rL * nL2 * nL2 + pL + rR * nR2 * nR2 + pR) - 0.5 * a * (rR * nR2 - rL * nL2)
    f2 = 0.5 * (rL * nL2 * tL2 + rR * nR2 * tR2) - 0.5 * a * (rR * tR2 - rL * tL2)
    return f0, f1, f2


@cuda.jit(device=True, inline=True)
def d_prim(U, alpha, P, b, i, j, rho0, p0, c0sq, g1h, nreed, Vr):
    a = alpha[b, i, j]
    ok = True
    if a > 0.0:
        m = U[b, 0, i, j]
        r = m / a
        P[b, 0, i, j] = r
        P[b, 1, i, j] = U[b, 1, i, j] / m
        P[b, 2, i, j] = U[b, 2, i, j] / m
        ok = (r > 0.3 * rho0) and (r < 3.0 * rho0)
        if not ok:
            r = rho0
    else:
        r = rho0
        P[b, 0, i, j] = r
        P[b, 1, i, j] = 0.0
        P[b, 2, i, j] = Vr if i < nreed else 0.0
    d = r - rho0
    P[b, 3, i, j] = p0 + c0sq * d * (1.0 + g1h * d)
    c = math.sqrt(c0sq * (1.0 + 2.0 * g1h * d))
    P[b, 4, i, j] = c
    P[b, 5, i, j] = 1.0 / c
    return ok


@cuda.jit(device=True, inline=True)
def d_yflux(P, alpha, yf, idy, idyc, hy, b, i, j, ny, nreed, Yb, Yt, rho0, p0, c0sq, g1h, mu):
    b0 = 0.0
    b1 = 0.0
    b2 = 0.0
    t0 = 0.0
    t1 = 0.0
    t2 = 0.0
    closed = (i < nreed) and (yf[j] >= Yb) and (yf[j] <= Yt)
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
                                   rT, nT, tT, P[b, 4, i, T], P[b, 5, i, T], rho0, p0, c0sq, g1h)
            b0 = f0
            b1 = f2 - mu * g1
            b2 = f1 - mu * g2
            t0 = b0
            t1 = b1
            t2 = b2
        elif aB > 0.0:
            jj = j - 1
            un = P[b, 2, i, jj]
            pw = P[b, 3, i, jj] + P[b, 0, i, jj] * P[b, 4, i, jj] * un
            if j == ny:
                b2 = pw
            else:
                k = 2.0 * mu * idy[jj]
                b2 = pw + k * un
                b1 = k * P[b, 1, i, jj]
        elif aT > 0.0:
            un = P[b, 2, i, j]
            pw = P[b, 3, i, j] - P[b, 0, i, j] * P[b, 4, i, j] * un
            if j == 0:
                t2 = pw
            else:
                k = 2.0 * mu * idy[j]
                t2 = pw - k * un
                t1 = -k * P[b, 1, i, j]
    return b0, b1, b2, t0, t1, t2


@cuda.jit(device=True, inline=True)
def d_mix(U, alpha, dy, b, i, a, c):
    for k in range(3):
        tot = U[b, k, i, a] * dy[a] + U[b, k, i, c] * dy[c]
        vol = alpha[b, i, a] * dy[a] + alpha[b, i, c] * dy[c]
        q = tot / vol
        U[b, k, i, a] = alpha[b, i, a] * q
        U[b, k, i, c] = alpha[b, i, c] * q


@cuda.jit(device=True, inline=True)
def d_search(yf, x, right):
    """number of faces < x (right=False) or <= x (right=True); like np.searchsorted."""
    lo = 0
    hi = yf.shape[0]
    while lo < hi:
        mid = (lo + hi) // 2
        if (yf[mid] <= x) if right else (yf[mid] < x):
            lo = mid + 1
        else:
            hi = mid
    return lo


# ----------------------------------------------------------------------------
# kernels.  rst[b] = (Y, V, Y1, V1);  ist[b] = (jt, jb, jt1, jb1, bad)
# bpar[b] = (rho0, p0, c0sq, g1h, rho_up, Y_rest, m, k, c, amix)
# ----------------------------------------------------------------------------
@cuda.jit
def g_flux_z(P, alpha, hz, idz, idzc, FL, FR, bpar, rst, mu, nreed, stage):
    j, i, b = cuda.grid(3)
    nb, nz, ny = alpha.shape
    if j >= ny or i > nz or b >= nb:
        return
    d_flux_z(P, alpha, hz, idz, idzc, FL, FR, bpar, rst, mu, nreed, stage, b, i, j)


@cuda.jit(device=True)
def d_flux_z(P, alpha, hz, idz, idzc, FL, FR, bpar, rst, mu, nreed, stage, b, i, j):
    nz = alpha.shape[1]
    rho0 = bpar[b, 0]
    p0 = bpar[b, 1]
    c0sq = bpar[b, 2]
    g1h = bpar[b, 3]
    Vr = rst[b, 1] if stage == 0 else rst[b, 3]
    aL = alpha[b, i - 1, j] if i > 0 else 0.0
    aR = alpha[b, i, j] if i < nz else 0.0
    beta = min(aL, aR)
    l0 = 0.0
    l1 = 0.0
    l2 = 0.0
    r0 = 0.0
    r1 = 0.0
    r2 = 0.0
    if beta > 0.0:
        L = i - 1
        Rr = i
        ic = idzc[L]
        g0 = (P[b, 0, Rr, j] - P[b, 0, L, j]) * ic
        g1 = (P[b, 1, Rr, j] - P[b, 1, L, j]) * ic
        g2 = (P[b, 2, Rr, j] - P[b, 2, L, j]) * ic
        rL = P[b, 0, L, j]
        nL = P[b, 1, L, j]
        tL = P[b, 2, L, j]
        if L >= 1 and aL == 1.0 and aR == 1.0 and alpha[b, L - 1, j] == 1.0:
            il = idzc[L - 1]
            x = hz[L]
            rL += x * d_mc((P[b, 0, L, j] - P[b, 0, L - 1, j]) * il, g0)
            nL += x * d_mc((P[b, 1, L, j] - P[b, 1, L - 1, j]) * il, g1)
            tL += x * d_mc((P[b, 2, L, j] - P[b, 2, L - 1, j]) * il, g2)
        rR = P[b, 0, Rr, j]
        nR = P[b, 1, Rr, j]
        tR = P[b, 2, Rr, j]
        if Rr <= nz - 2 and aL == 1.0 and aR == 1.0 and alpha[b, Rr + 1, j] == 1.0:
            ir = idzc[Rr]
            x = hz[Rr]
            rR -= x * d_mc(g0, (P[b, 0, Rr + 1, j] - P[b, 0, Rr, j]) * ir)
            nR -= x * d_mc(g1, (P[b, 1, Rr + 1, j] - P[b, 1, Rr, j]) * ir)
            tR -= x * d_mc(g2, (P[b, 2, Rr + 1, j] - P[b, 2, Rr, j]) * ir)
        f0, f1, f2 = d_rusanov(rL, nL, tL, P[b, 4, L, j], P[b, 5, L, j],
                               rR, nR, tR, P[b, 4, Rr, j], P[b, 5, Rr, j], rho0, p0, c0sq, g1h)
        f1 -= mu * g1
        f2 -= mu * g2
        l0 = beta * f0
        l1 = beta * f1
        l2 = beta * f2
        r0 = l0
        r1 = l1
        r2 = l2
    wl = aL - beta
    if wl > 0.0:
        ii = i - 1
        un = P[b, 1, ii, j]
        pw = P[b, 3, ii, j] + P[b, 0, ii, j] * P[b, 4, ii, j] * un
        if i == nz:
            l1 += wl * pw
        else:
            k = 2.0 * mu * idz[ii]
            l1 += wl * (pw + k * un)
            l2 += wl * (k * P[b, 2, ii, j])
    wr = aR - beta
    if wr > 0.0:
        un = P[b, 1, i, j]
        pw = P[b, 3, i, j] - P[b, 0, i, j] * P[b, 4, i, j] * un
        if i == 0:
            r1 += wr * pw
        else:
            k = 2.0 * mu * idz[i]
            wt = Vr if i == nreed else 0.0
            r1 += wr * (pw - k * un)
            r2 -= wr * k * (P[b, 2, i, j] - wt)
    FL[b, 0, i, j] = l0
    FL[b, 1, i, j] = l1
    FL[b, 2, i, j] = l2
    FR[b, 0, i, j] = r0
    FR[b, 1, i, j] = r1
    FR[b, 2, i, j] = r2


@cuda.jit
def g_update(P, Pn, Ucur, Uold, Unew, ca, cb, cc, dt, FL, FR, alpha, yf, hy, idz, idy, idyc,
             sig, sigm, is_up, bpar, rst, ist, mu, nreed, h, jpb, nslot, qslot, stage):
    j, i, b = cuda.grid(3)
    nb, nz, ny = alpha.shape
    if j >= ny or i >= nz or b >= nb:
        return
    d_update(P, Pn, Ucur, Uold, Unew, ca, cb, cc, dt, FL, FR, alpha, yf, hy, idz, idy, idyc,
             sig, sigm, is_up, bpar, rst, ist, mu, nreed, h, jpb, nslot, qslot, stage, b, i, j)


@cuda.jit(device=True)
def d_update(P, Pn, Ucur, Uold, Unew, ca, cb, cc, dt, FL, FR, alpha, yf, hy, idz, idy, idyc,
             sig, sigm, is_up, bpar, rst, ist, mu, nreed, h, jpb, nslot, qslot, stage, b, i, j):
    ny = alpha.shape[2]
    rho0 = bpar[b, 0]
    p0 = bpar[b, 1]
    c0sq = bpar[b, 2]
    g1h = bpar[b, 3]
    if stage == 0:
        Yb = rst[b, 0]
        Vr = rst[b, 1]
        jt = ist[b, 0]
        jb = ist[b, 1]
    else:
        Yb = rst[b, 2]
        Vr = rst[b, 3]
        jt = ist[b, 2]
        jb = ist[b, 3]
    Yt = Yb + h
    _, _, _, ft0, ft1, ft2 = d_yflux(P, alpha, yf, idy, idyc, hy, b, i, j, ny, nreed, Yb, Yt,
                                     rho0, p0, c0sq, g1h, mu)
    b0, b1, b2, t0, _, _ = d_yflux(P, alpha, yf, idy, idyc, hy, b, i, j + 1, ny, nreed, Yb, Yt,
                                   rho0, p0, c0sq, g1h, mu)
    if j + 1 == jpb and i < nslot:
        qslot[b, i] = t0
    if alpha[b, i, j] > 0.0:
        iz = idz[i]
        iy = idy[j]
        q0 = -(FL[b, 0, i + 1, j] - FR[b, 0, i, j]) * iz - (b0 - ft0) * iy
        q1 = -(FL[b, 1, i + 1, j] - FR[b, 1, i, j]) * iz - (b1 - ft1) * iy
        q2 = -(FL[b, 2, i + 1, j] - FR[b, 2, i, j]) * iz - (b2 - ft2) * iy
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
    else:
        Unew[b, 0, i, j] = 0.0
        Unew[b, 1, i, j] = 0.0
        Unew[b, 2, i, j] = 0.0
    if not (i < nreed and j >= jb - 2 and j <= jt + 2):
        if not d_prim(Unew, alpha, Pn, b, i, j, rho0, p0, c0sq, g1h, nreed, Vr):
            ist[b, 4] = 1


@cuda.jit
def g_reed(P, Pn, Uold, Unew, alpha, dz, dy, yf, bpar, rst, ist, mu, nreed, h, dt, stage,
           rec, rec_on, irec, t_now, mic_i, mic_j, mic_w, qslot, nslot):
    d_reed(P, Pn, Uold, Unew, alpha, dz, dy, yf, bpar, rst, ist, mu, nreed, h, dt, stage,
           rec, rec_on, irec, t_now, mic_i, mic_j, mic_w, qslot, nslot,
           cuda.blockIdx.x, cuda.threadIdx.x, cuda.blockDim.x)


@cuda.jit(device=True)
def d_reed(P, Pn, Uold, Unew, alpha, dz, dy, yf, bpar, rst, ist, mu, nreed, h, dt, stage,
           rec, rec_on, irec, t_now, mic_i, mic_j, mic_w, qslot, nslot, b, tid, nt):
    """One block per batch member: wall forces on the reed, reed ODE, fluid-fraction
    update, covered-cell transfer, small-cell merging, primitives near the reed."""
    ny = dy.shape[0]
    sF = cuda.shared.array(256, float64)
    sY = cuda.shared.array(2, float64)
    sJ = cuda.shared.array(2, int32)
    rho0 = bpar[b, 0]
    p0 = bpar[b, 1]
    c0sq = bpar[b, 2]
    g1h = bpar[b, 3]
    if stage == 0:
        Y = rst[b, 0]
        V = rst[b, 1]
        jt = ist[b, 0]
        jb = ist[b, 1]
        cdt = dt
    else:
        Y = rst[b, 2]
        V = rst[b, 3]
        jt = ist[b, 2]
        jb = ist[b, 3]
        cdt = 0.5 * dt
    # ---- 1. wall fluxes on the reed's top/bottom faces, force ----
    acc = 0.0
    for i in range(tid, nreed, nt):
        hd = 0.5 * dy[jt]
        vv = P[b, 2, i, jt]
        pw = P[b, 3, i, jt] + P[b, 0, i, jt] * P[b, 4, i, jt] * (V - vv)
        fw = pw - mu * (vv - V) / hd
        Unew[b, 2, i, jt] += cdt * fw / dy[jt]
        Unew[b, 1, i, jt] -= cdt * mu * P[b, 1, i, jt] / hd / dy[jt]
        acc -= fw * dz[i]
        hd = 0.5 * dy[jb]
        vv = P[b, 2, i, jb]
        pw = P[b, 3, i, jb] + P[b, 0, i, jb] * P[b, 4, i, jb] * (vv - V)
        fw = pw + mu * (vv - V) / hd
        Unew[b, 2, i, jb] -= cdt * fw / dy[jb]
        Unew[b, 1, i, jb] -= cdt * mu * P[b, 1, i, jb] / hd / dy[jb]
        acc += fw * dz[i]
    hdz = 0.5 * dz[nreed]
    for j in range(jb + tid, jt + 1, nt):
        w = 1.0 - alpha[b, 0, j]
        if w > 0.0 and alpha[b, nreed, j] > 0.0:
            acc += mu * (P[b, 2, nreed, j] - V) / hdz * w * dy[j]
    sF[tid] = acc
    cuda.syncthreads()
    # ---- 2. reed ODE (thread 0) ----
    if tid == 0:
        F = 0.0
        for k in range(nt):
            F += sF[k]
        mr = bpar[b, 6]
        kr = bpar[b, 7]
        cr = bpar[b, 8]
        Yr = bpar[b, 5]
        A = (2.0 * F - cr * V - kr * (Y - Yr)) / mr
        if stage == 0:
            Yn = Y + dt * V
            Vn = V + dt * A
            rst[b, 2] = Yn
            rst[b, 3] = Vn
            if rec_on:
                rec[b, irec, 0] = t_now
                rec[b, irec, 1] = Y
                rec[b, irec, 2] = V
                rec[b, irec, 3] = 2.0 * F
                qs = 0.0
                for i in range(nslot):
                    qs += qslot[b, i] * dz[i]
                rec[b, irec, 4] = 2.0 * qs / rho0
                for m in range(mic_w.shape[0]):
                    s = 0.0
                    for q in range(4):
                        s += mic_w[m, q] * P[b, 3, mic_i[m, q], mic_j[m, q]]
                    rec[b, irec, 5 + m] = s
        else:
            Yn = 0.5 * (rst[b, 0] + Y + dt * V)
            Vn = 0.5 * (rst[b, 1] + V + dt * A)
            rst[b, 0] = Yn
            rst[b, 1] = Vn
        jbn = d_search(yf, Yn, False) - 1
        jtn = d_search(yf, Yn + h, True) - 1
        if stage == 0:
            ist[b, 2] = jtn
            ist[b, 3] = jbn
        else:
            ist[b, 0] = jtn
            ist[b, 1] = jbn
        sY[0] = Yn
        sY[1] = Vn
        sJ[0] = jtn
        sJ[1] = jbn
    cuda.syncthreads()
    Yn = sY[0]
    Vn = sY[1]
    jtn = sJ[0]
    jbn = sJ[1]
    Yt = Yn + h
    amix = bpar[b, 9]
    j0 = max(jbn - 3, 0)
    j1 = min(jtn + 4, ny)
    r0 = max(min(jb, jbn) - 4, 0)
    r1 = min(max(jt, jtn) + 5, ny)
    ymid = Yn + 0.5 * h
    # ---- 3. per reed column: fluid fractions, covered cells, merging, primitives ----
    for i in range(tid, nreed, nt):
        for j in range(j0, j1):
            if j > jbn and j < jtn:
                a = 0.0
            else:
                ov = min(yf[j + 1], Yt) - max(yf[j], Yn)
                a = 1.0 - ov / dy[j] if ov > 0.0 else 1.0
                if a < 0.0:
                    a = 0.0
            alpha[b, i, j] = a
        for j in range(jbn + 1, jtn):
            if Unew[b, 0, i, j] != 0.0 or Unew[b, 1, i, j] != 0.0 or Unew[b, 2, i, j] != 0.0:
                tgt = jtn if 0.5 * (yf[j] + yf[j + 1]) > ymid else jbn
                f = dy[j] / dy[tgt]
                for k in range(3):
                    Unew[b, k, i, tgt] += Unew[b, k, i, j] * f
                    Unew[b, k, i, j] = 0.0
        if alpha[b, i, jtn] < amix:
            d_mix(Unew, alpha, dy, b, i, jtn, jtn + 1)
        if alpha[b, i, jbn] < amix:
            d_mix(Unew, alpha, dy, b, i, jbn, jbn - 1)
        for j in range(r0, r1):
            if not d_prim(Unew, alpha, Pn, b, i, j, rho0, p0, c0sq, g1h, nreed, Vn):
                ist[b, 4] = 1


@cuda.jit
def g_persistent(P, Pn, U, U1, alpha, hz, idz, idzc, FL, FR, yf, hy, idy, idyc, dz, dy,
                 sig, sigm, is_up, bpar, rst, ist, mu, nreed, h, jpb, nslot, qslot, dt,
                 rec, rec_every, n0, nsteps, mic_i, mic_j, mic_w):
    """Advance `nsteps` time steps in ONE cooperative launch (grid-wide syncs
    between the phases) -- removes per-step kernel-launch overhead."""
    grid = cuda.cg.this_grid()
    nb, nz, ny = alpha.shape
    gt = cuda.grid(1)
    T = cuda.gridsize(1)
    tid = cuda.threadIdx.x
    nt = cuda.blockDim.x
    blk = cuda.blockIdx.x
    nzf = (nz + 1) * ny
    ncz = nz * ny
    for s in range(nsteps):
        n = n0 + s
        rec_on = (n % rec_every) == 0
        irec = n // rec_every
        # ---- stage 1 ----
        for idx in range(gt, nb * nzf, T):
            b = idx // nzf
            r = idx - b * nzf
            d_flux_z(P, alpha, hz, idz, idzc, FL, FR, bpar, rst, mu, nreed, 0, b, r // ny, r % ny)
        grid.sync()
        for idx in range(gt, nb * ncz, T):
            b = idx // ncz
            r = idx - b * ncz
            d_update(P, Pn, U, U, U1, 0.0, 1.0, 1.0, dt, FL, FR, alpha, yf, hy, idz, idy, idyc,
                     sig, sigm, is_up, bpar, rst, ist, mu, nreed, h, jpb, nslot, qslot, 0,
                     b, r // ny, r % ny)
        grid.sync()
        if blk < nb:
            d_reed(P, Pn, U, U1, alpha, dz, dy, yf, bpar, rst, ist, mu, nreed, h, dt, 0,
                   rec, rec_on, irec, n * dt, mic_i, mic_j, mic_w, qslot, nslot, blk, tid, nt)
        grid.sync()
        # ---- stage 2 (buffers swapped) ----
        for idx in range(gt, nb * nzf, T):
            b = idx // nzf
            r = idx - b * nzf
            d_flux_z(Pn, alpha, hz, idz, idzc, FL, FR, bpar, rst, mu, nreed, 1, b, r // ny, r % ny)
        grid.sync()
        for idx in range(gt, nb * ncz, T):
            b = idx // ncz
            r = idx - b * ncz
            d_update(Pn, P, U1, U, U, 0.5, 0.5, 0.5, dt, FL, FR, alpha, yf, hy, idz, idy, idyc,
                     sig, sigm, is_up, bpar, rst, ist, mu, nreed, h, jpb, nslot, qslot, 1,
                     b, r // ny, r % ny)
        grid.sync()
        if blk < nb:
            d_reed(Pn, P, U1, U, alpha, dz, dy, yf, bpar, rst, ist, mu, nreed, h, dt, 1,
                   rec, False, 0, n * dt, mic_i, mic_j, mic_w, qslot, nslot, blk, tid, nt)
        grid.sync()


# ----------------------------------------------------------------------------
# host driver
# ----------------------------------------------------------------------------
class GpuBatch:
    TPB = (32, 4, 1)
    TPB_P = 128

    def __init__(self, cfgs):
        self.cfgs = cfgs
        self.sims = [R2.ReedSim(c) for c in cfgs]
        s0 = self.sims[0]
        for s in self.sims[1:]:
            assert (s.nz, s.ny, s.nreed, s.ngap) == (s0.nz, s0.ny, s0.nreed, s0.ngap), "grids differ"
            assert np.allclose(s.zf, s0.zf) and np.allclose(s.yf, s0.yf)
            assert s.cfg.mu == s0.cfg.mu and s.cfg.reed_h == s0.cfg.reed_h
        self.dt = min(s.dt for s in self.sims)
        self.t_end = max(c.t_end for c in cfgs)
        self.nsteps = int(math.ceil(self.t_end / self.dt))
        self.rec_every = s0.cfg.rec_every
        nrec = self.nsteps // self.rec_every + 2
        for s in self.sims:
            s.dt = self.dt
            s.nsteps = self.nsteps
            s.rec = np.zeros((nrec, s.rec.shape[1]))
        self.nb = len(self.sims)
        nz, ny = s0.nz, s0.ny
        st = lambda name: cuda.to_device(np.ascontiguousarray(np.stack([getattr(s, name) for s in self.sims])))
        self.U = st("U")
        self.U1 = cuda.to_device(np.zeros((self.nb, 3, nz, ny)))
        self.P = st("P")
        self.Pn = cuda.to_device(np.zeros((self.nb, 6, nz, ny)))
        self.FL = cuda.to_device(np.zeros((self.nb, 3, nz + 1, ny)))
        self.FR = cuda.to_device(np.zeros((self.nb, 3, nz + 1, ny)))
        self.alpha = st("alpha")
        self.qslot = cuda.to_device(np.zeros((self.nb, nz)))
        g = lambda a: cuda.to_device(np.ascontiguousarray(a))
        self.hz, self.idz, self.idzc = g(s0.hz), g(s0.idz), g(s0.idzc)
        self.yf, self.dy, self.hy, self.idy, self.idyc, self.dz = g(s0.yf), g(s0.dy), g(s0.hy), g(s0.idy), g(s0.idyc), g(s0.dz)
        self.sig, self.sigm, self.is_up = g(s0.sig), g(s0.sigm), g(s0.is_up.astype(np.uint8))
        self.bpar_h = np.array([[s.rho0, s.cfg.p_amb, s.c0sq, s.g1h, s.rho_up, s.Y_rest, s.m_r, s.k_r, s.c_r, 0.5]
                                for s in self.sims])
        self.bpar = g(self.bpar_h)
        self.rst = g(np.array([[s.rs[0], s.rs[1], s.rs[0], s.rs[1]] for s in self.sims]))
        self.ist = g(np.array([[s.ipar[3], s.ipar[4], s.ipar[3], s.ipar[4], 0] for s in self.sims], np.int32))
        self.rec = g(np.zeros((self.nb, nrec, self.sims[0].rec.shape[1])))
        self.mic_i, self.mic_j, self.mic_w = g(s0.mic_i), g(s0.mic_j), g(s0.mic_w)
        self.mu, self.nreed, self.h = s0.cfg.mu, s0.nreed, s0.cfg.reed_h
        self.jpb, self.nslot = s0.jpb, s0.nreed + s0.ngap
        t = self.TPB
        self.grid_z = ((ny + t[0] - 1) // t[0], (nz + 1 + t[1] - 1) // t[1], self.nb)
        self.grid_c = ((ny + t[0] - 1) // t[0], (nz + t[1] - 1) // t[1], self.nb)

    def step(self, n):
        dt, mu, nreed, h = self.dt, self.mu, self.nreed, self.h
        rec_on = (n % self.rec_every == 0)
        irec = n // self.rec_every
        P, Pn = self.P, self.Pn
        # stage 1
        g_flux_z[self.grid_z, self.TPB](P, self.alpha, self.hz, self.idz, self.idzc, self.FL, self.FR,
                                        self.bpar, self.rst, mu, nreed, 0)
        g_update[self.grid_c, self.TPB](P, Pn, self.U, self.U, self.U1, 0.0, 1.0, 1.0, dt, self.FL, self.FR,
                                        self.alpha, self.yf, self.hy, self.idz, self.idy, self.idyc,
                                        self.sig, self.sigm, self.is_up, self.bpar, self.rst, self.ist,
                                        mu, nreed, h, self.jpb, self.nslot, self.qslot, 0)
        g_reed[self.nb, 64](P, Pn, self.U, self.U1, self.alpha, self.dz, self.dy, self.yf, self.bpar,
                            self.rst, self.ist, mu, nreed, h, dt, 0, self.rec, rec_on, irec, n * dt,
                            self.mic_i, self.mic_j, self.mic_w, self.qslot, self.nslot)
        P, Pn = Pn, P
        # stage 2
        g_flux_z[self.grid_z, self.TPB](P, self.alpha, self.hz, self.idz, self.idzc, self.FL, self.FR,
                                        self.bpar, self.rst, mu, nreed, 1)
        g_update[self.grid_c, self.TPB](P, Pn, self.U1, self.U, self.U, 0.5, 0.5, 0.5, dt, self.FL, self.FR,
                                        self.alpha, self.yf, self.hy, self.idz, self.idy, self.idyc,
                                        self.sig, self.sigm, self.is_up, self.bpar, self.rst, self.ist,
                                        mu, nreed, h, self.jpb, self.nslot, self.qslot, 1)
        g_reed[self.nb, 64](P, Pn, self.U1, self.U, self.alpha, self.dz, self.dy, self.yf, self.bpar,
                            self.rst, self.ist, mu, nreed, h, dt, 1, self.rec, False, 0, n * dt,
                            self.mic_i, self.mic_j, self.mic_w, self.qslot, self.nslot)
        # after two swaps P is the current state again

    use_coop = False  # persistent cooperative kernel: correct, but slower on a 4090 (FP64-bound)

    def steps(self, n0, count):
        """Advance `count` steps (per-kernel launches, or one cooperative launch)."""
        if not self.use_coop:
            for n in range(n0, n0 + count):
                self.step(n)
            return
        if not hasattr(self, "_coop"):
            k = g_persistent
            args = self._pargs(0, 1)
            kd = k.specialize(*args)
            nblk = kd.overloads[next(iter(kd.overloads))].max_cooperative_grid_blocks(self.TPB_P)
            need = (self.nb * (self.sims[0].nz + 1) * self.sims[0].ny + self.TPB_P - 1) // self.TPB_P
            self._coop = (kd, max(self.nb, min(nblk, need)))
        kd, nblk = self._coop
        kd[nblk, self.TPB_P](*self._pargs(n0, count))

    def _pargs(self, n0, count):
        return (self.P, self.Pn, self.U, self.U1, self.alpha, self.hz, self.idz, self.idzc, self.FL, self.FR,
                self.yf, self.hy, self.idy, self.idyc, self.dz, self.dy, self.sig, self.sigm, self.is_up,
                self.bpar, self.rst, self.ist, self.mu, self.nreed, self.h, self.jpb, self.nslot, self.qslot,
                self.dt, self.rec, self.rec_every, n0, count, self.mic_i, self.mic_j, self.mic_w)

    def run(self, out_root, verbose=True, check_every=2000):
        T0 = 1.0 / self.sims[0].f_vac
        c0 = self.cfgs[0]
        snap_dt = T0 / c0.snap_per_cycle
        next_snap = self.t_end - c0.snap_cycles * T0
        snaps = [[] for _ in range(self.nb)]
        wall0 = time.time()
        last = wall0
        n = 0
        while n < self.nsteps:
            if c0.t_ramp > 0 and n % 200 == 0:
                for b, s in enumerate(self.sims):
                    sfac = min(1.0, n * self.dt / s.cfg.t_ramp)
                    sfac = sfac * sfac * (3 - 2 * sfac)
                    self.bpar_h[b, 4] = s._rho_of_p(s.cfg.p_amb + s.cfg.dp * sfac)
                self.bpar.copy_to_device(self.bpar_h)
            cnt = min(check_every - n % check_every, self.nsteps - n)
            if next_snap <= self.t_end:
                cnt = max(1, min(cnt, int(math.ceil((next_snap - n * self.dt) / self.dt))))
            if c0.t_ramp > 0:
                cnt = min(cnt, 200 - n % 200)
            self.steps(n, cnt)
            n += cnt
            t = n * self.dt
            if t >= next_snap and next_snap <= self.t_end:
                Ph = self.P.copy_to_host()
                ah = self.alpha.copy_to_host()
                rs = self.rst.copy_to_host()
                for b, s in enumerate(self.sims):
                    snaps[b].append(dict(t=t, Y=rs[b, 0], p=(Ph[b, 3] - s.cfg.p_amb).astype(np.float32),
                                         u=Ph[b, 1].astype(np.float32), v=Ph[b, 2].astype(np.float32),
                                         alpha=ah[b].astype(np.float32)))
                next_snap += snap_dt
            if n % check_every == 0 or n == self.nsteps:
                ist = self.ist.copy_to_host()
                if ist[:, 4].any():
                    bad = [self.cfgs[b].name for b in np.nonzero(ist[:, 4])[0]]
                    raise RuntimeError(f"blow-up at step {n} in {bad}")
                now = time.time()
                if verbose and (now - last > 15 or n == self.nsteps):
                    last = now
                    rs = self.rst.copy_to_host()
                    rate = n / (now - wall0)
                    ys = " ".join(f"{1e3 * (rs[b, 0] - s.Y_rest):+.3f}" for b, s in enumerate(self.sims))
                    print(f"t={t * 1e3:8.3f} ms step {n}/{self.nsteps} {rate:.0f} steps/s "
                          f"eta {(self.nsteps - n) / rate:.0f}s  Y[mm]: {ys}", flush=True)
        cuda.synchronize()
        self.wall_time = time.time() - wall0
        rec = self.rec.copy_to_host()
        Uh = self.U.copy_to_host()
        Ph = self.P.copy_to_host()
        ah = self.alpha.copy_to_host()
        rs = self.rst.copy_to_host()
        ist = self.ist.copy_to_host()
        out_dirs = []
        for b, s in enumerate(self.sims):
            nrec = min(self.nsteps // self.rec_every + 1, rec.shape[1])
            s.rec = rec[b]
            s.ipar[5] = nrec
            s.U, s.P, s.alpha = Uh[b], Ph[b], ah[b]
            s.rs[:] = rs[b, :2]
            s.ipar[3], s.ipar[4] = ist[b, 0], ist[b, 1]
            s.wall_time = self.wall_time
            od = os.path.join(out_root, s.cfg.name)
            os.makedirs(od, exist_ok=True)
            s.save(od, snaps[b])
            out_dirs.append(od)
        return out_dirs


def check_against_cpu(nsteps=400):
    """Run CPU and GPU solvers side by side and compare the state."""
    cfg = R2.Config(name="chk", dp=1000.0, t_end=1.0)
    cpu = R2.ReedSim(cfg)
    gpu = GpuBatch([replace(cfg), replace(cfg, name="chk2", p_amb=80e3)])
    cpu.dt = gpu.dt
    st = R2.advance(nsteps, 0, cpu.dt, cpu.U, cpu.U1, cpu.alpha, cpu.P, cpu.Pn, cpu.FL, cpu.FR, cpu.qslot,
                    cpu.hz, cpu.idz, cpu.idzc, cpu.yf, cpu.dy, cpu.hy, cpu.idy, cpu.idyc, cpu.dz,
                    cpu.sig, cpu.sigm, cpu.is_up, cpu.par, cpu.ipar, cpu.rs, cpu.mic_i, cpu.mic_j,
                    cpu.mic_w, cpu.rec, cpu.cfg.rec_every)
    assert st == 0
    gpu2 = GpuBatch([replace(cfg), replace(cfg, name="chk2", p_amb=80e3)])
    for n in range(nsteps):
        gpu.step(n)
    gpu2.steps(0, nsteps // 2)
    gpu2.steps(nsteps // 2, nsteps - nsteps // 2)
    cuda.synchronize()
    for lab, g in (("per-kernel", gpu), ("persistent", gpu2)):
        Ug = g.U.copy_to_host()[0]
        rs = g.rst.copy_to_host()[0]
        rel = np.abs(Ug - cpu.U).max() / np.abs(cpu.U).max()
        rc = g.rec.copy_to_host()[0]
        k = nsteps // cpu.cfg.rec_every
        print(f"{lab}: after {nsteps} steps max|dU|/max|U| = {rel:.3e}; Y cpu {cpu.rs[0]:.12e} "
              f"gpu {rs[0]:.12e}; force cpu {cpu.rec[k - 1, 3]:.10f} gpu {rc[k - 1, 3]:.10f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    multi = {"p_amb", "dp", "kick", "Q_mech", "set_gap", "t_end"}
    for k, v in asdict(R2.Config()).items():
        if k in ("mics", "name"):
            continue
        flag = "--" + k.replace("_", "-")
        if k in multi:
            ap.add_argument(flag, type=float, nargs="+", default=[v])
        elif isinstance(v, bool):
            ap.add_argument(flag, type=lambda s: s.lower() in ("1", "true", "yes"), default=v)
        else:
            ap.add_argument(flag, type=type(v), default=v)
    ap.add_argument("--prefix", default="gpu")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--grid", choices=["product", "zip"], default="product",
                    help="combine multi-valued options as a cartesian product or element-wise")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--no-analysis", action="store_true")
    a = ap.parse_args()
    if a.check:
        check_against_cpu()
        return
    base = {k: getattr(a, k) for k in asdict(R2.Config()) if k not in ("mics", "name") and k not in multi}
    lists = {k: getattr(a, k) for k in multi}
    keys = sorted(lists)
    if a.grid == "zip":
        n = max(len(v) for v in lists.values())
        combos = [{k: (lists[k][i] if len(lists[k]) > 1 else lists[k][0]) for k in keys} for i in range(n)]
    else:
        import itertools
        combos = [dict(zip(keys, vals)) for vals in itertools.product(*[lists[k] for k in keys])]
    cfgs = []
    for c in combos:
        tag = "_".join(f"{k}{c[k]:g}" for k in keys if len(lists[k]) > 1) or "run"
        cfgs.append(R2.Config(name=f"{a.prefix}_{tag}", **base, **c))
    for c in cfgs:
        print(c.name)
    batch = GpuBatch(cfgs)
    print(f"batch of {batch.nb}, grid {batch.sims[0].nz}x{batch.sims[0].ny}, dt={batch.dt:.3e}, "
          f"{batch.nsteps} steps", flush=True)
    dirs = batch.run(a.out)
    print(f"wall time {batch.wall_time:.0f}s")
    if not a.no_analysis:
        from analysis import analyze_run
        for d in dirs:
            analyze_run(d)


if __name__ == "__main__":
    main()
