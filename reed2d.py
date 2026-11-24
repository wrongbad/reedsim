#!/usr/bin/env python3
"""
reed2d.py -- 2-D compressible Navier-Stokes model of an accordion free reed.

What is simulated
-----------------
A cross-section taken *across* the reed tongue (perpendicular to its length),
near the tip where the valve action happens:

        y ^          bellows / reservoir  (p_amb + dp)
          |
          |        [=====reed=====]          <- steel strip, moves in y only
     -----+--------+  slot  +--------------  <- reed plate (aluminium), thickness t_p
          |        |        |
          |   +----+ chamber+----+           <- reed chamber (2-D proxy)
          |   |                  |
          |   +------+    +------+           <- outlet ("pallet hole")
          |            room (p_amb)
          +-----------------------------> z

* Air: compressible, barotropic (isentropic) Navier-Stokes. Ambient pressure
  p_amb sets the density rho0 = p_amb/(R T); the speed of sound and the dynamic
  viscosity do NOT depend on pressure at fixed temperature, so the only way
  ambient pressure enters is through rho0 (fluid inertia / added mass, jet
  speed sqrt(2 dp/rho0), Reynolds number, acoustic impedance rho0*c).
* Reed: strip theory. The 2-D section is a rigid steel strip of width b and
  thickness h on a spring tuned to the first cantilever mode of a uniform
  tongue of length L (Euler-Bernoulli). Mass/stiffness are per unit length, so
  the section's air-loading ratio equals the modal one for the cantilever.
* Numerics: finite volume on a stretched Cartesian grid; MUSCL (van Leer) +
  Rusanov flux with Thornber low-Mach velocity correction; SSP-RK2 in time.
  Static walls are blocked cell faces; the moving reed is a conservative 1-D
  cut-cell (its top/bottom faces cut grid rows) with small-cell merging.
  Mirror symmetry about the reed centre-line (z = 0) halves the cost.
* Boundaries: sponge layers relax density to the reservoir / ambient value.

Run `python reed2d.py -h` for options. Output goes to runs/<name>/.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass, field

import numpy as np
from numba import njit, prange

BETA1 = 1.8751040687  # first cantilever eigenvalue (beta*L)


# ----------------------------------------------------------------------------
# configuration
# ----------------------------------------------------------------------------
@dataclass
class Config:
    name: str = "run"
    # --- air ---
    p_amb: float = 101325.0   # ambient static pressure [Pa]
    T: float = 293.15         # air temperature [K]
    gamma: float = 1.4
    R_air: float = 287.05     # specific gas constant [J/(kg K)]
    mu: float = 1.81e-5       # dynamic viscosity [Pa s]  (pressure independent)
    dp: float = 1000.0        # bellows over-pressure [Pa]; 0 => plucked reed in still air
    t_ramp: float = 0.0       # ramp time for dp [s] (0 = start pressurised)
    # --- reed tongue (Swedish spring steel), roughly an A4 treble reed ---
    reed_L: float = 23.75e-3  # free length [m]
    reed_b: float = 2.5e-3    # width [m]
    reed_h: float = 0.30e-3   # (equivalent uniform) thickness [m]
    E: float = 206e9          # Young's modulus [Pa]
    rho_s: float = 7850.0     # steel density [kg/m^3]
    Q_mech: float = 800.0     # structural quality factor (internal + clamp losses)
    set_gap: float = 0.10e-3  # rest height of reed underside above the plate ("set")
    kick: float = 0.6e-3      # initial downward displacement (pluck) [m]
    # --- reed plate / slot ---
    plate_t: float = 1.5e-3   # plate thickness [m]
    gap: float = 0.05e-3      # side clearance between tongue edge and slot wall [m]
    # --- reed chamber (downstream cavity) ---
    chamber: bool = True
    chamber_w: float = 8e-3
    chamber_h: float = 10e-3
    outlet_w: float = 4e-3
    wall_t: float = 2e-3
    # --- domain / grid ---
    z_max: float = 18e-3      # half-width of domain
    y_top: float = 15e-3      # top of reservoir region
    room_below: float = 22e-3 # free air below the chamber
    dz_fine: float = 0.025e-3 # across-gap resolution (gap/2 by default)
    dy_fine: float = 0.04e-3
    d_coarse: float = 0.6e-3
    stretch: float = 1.08
    core_up: float = 1.6e-3   # fine region above the plate
    core_dn: float = 1.0e-3   # fine region below the plate
    core_side: float = 0.5e-3 # fine region beyond the slot wall
    sponge: float = 5e-3
    sponge_sigma: float = 1.0e6     # strong enough to hold reservoir pressures under throughflow
    # --- time ---
    t_end: float = 0.02
    cfl: float = 0.8
    rec_every: int = 8        # record probes every N steps
    snap_cycles: float = 1.0  # save field snapshots over the last N reed periods
    snap_per_cycle: int = 12
    mics: list = field(default_factory=list)  # [[name, z, y], ...]; default set if empty

    # derived helpers -------------------------------------------------------
    def rho0(self):
        return self.p_amb / (self.R_air * self.T)

    def c0(self):
        return math.sqrt(self.gamma * self.R_air * self.T)

    def f_vacuum(self, b=None):
        """First bending mode of the uniform steel cantilever (no air)."""
        return (BETA1 ** 2 / (2 * math.pi * self.reed_L ** 2)) * math.sqrt(
            self.E * self.reed_h ** 2 / (12 * self.rho_s))


# ----------------------------------------------------------------------------
# grid
# ----------------------------------------------------------------------------
def _stretch(start, end, d0, dmax, ratio):
    """Faces from `start` towards `end` with geometrically growing spacing."""
    sgn = 1.0 if end > start else -1.0
    faces = []
    x, d = start, d0
    while True:
        d = min(d * ratio, dmax)
        if sgn * (end - (x + sgn * d)) < 0.5 * d:
            faces.append(end)
            break
        x += sgn * d
        faces.append(x)
    return faces


def make_axis(lo, hi, core_lo, core_hi, d0, dmax, ratio):
    n = int(round((core_hi - core_lo) / d0))
    core = list(core_lo + d0 * np.arange(n + 1))
    up = _stretch(core[-1], hi, d0, dmax, ratio) if hi > core[-1] + 1e-12 else []
    dn = _stretch(core[0], lo, d0, dmax, ratio)[::-1] if lo < core[0] - 1e-12 else []
    f = np.array(dn + core + up)
    assert np.all(np.diff(f) > 0)
    return f


# ----------------------------------------------------------------------------
# numba kernels
# ----------------------------------------------------------------------------
FM = {"nsz", "arcp", "contract", "afn", "reassoc"}  # fast math, but keep NaN semantics


@njit(inline="always", fastmath=FM)
def _pres(r, rho0, p0, c0sq, g1h):
    # 2nd-order expansion of the isentrope p = p0 (rho/rho0)^gamma (error ~ (drho/rho)^3)
    d = r - rho0
    return p0 + c0sq * d * (1.0 + g1h * d)


@njit(inline="always", fastmath=FM)
def _mc(a, b):
    """Monotonised-central slope limiter."""
    # branchless:  0.5*(sgn a + sgn b) * min(2|a|, 2|b|, |a+b|/2)
    m = min(2.0 * abs(a), 2.0 * abs(b), 0.5 * abs(a + b))
    return 0.5 * (math.copysign(1.0, a) + math.copysign(1.0, b)) * m


@njit(inline="always", fastmath=FM)
def _rusanov(rL, nL, tL, cL, icL, rR, nR, tR, cR, icR, rho0, p0, c0sq, g1h):
    pL = _pres(rL, rho0, p0, c0sq, g1h)
    pR = _pres(rR, rho0, p0, c0sq, g1h)
    # Thornber et al. (2008) low-Mach correction of the reconstructed velocities
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


@njit(inline="always")
def _prim(U, alpha, P, i, j, rho0, p0, c0sq, g1h, nreed, Vr):
    """Primitive variables (rho, u, v, p, c, 1/c) of one cell; False if unphysical."""
    a = alpha[i, j]
    ok = True
    if a > 0.0:
        m = U[0, i, j]
        r = m / a
        P[0, i, j] = r
        P[1, i, j] = U[1, i, j] / m
        P[2, i, j] = U[2, i, j] / m
        ok = (r > 0.3 * rho0) and (r < 3.0 * rho0)
        if not ok:
            r = rho0
    else:
        r = rho0
        P[0, i, j] = r
        P[1, i, j] = 0.0
        P[2, i, j] = Vr if i < nreed else 0.0
    d = r - rho0
    P[3, i, j] = p0 + c0sq * d * (1.0 + g1h * d)
    c = math.sqrt(c0sq * (1.0 + 2.0 * g1h * d))
    P[4, i, j] = c
    P[5, i, j] = 1.0 / c
    return ok


@njit(parallel=True)
def k_prims(U, alpha, P, rho0, p0, c0sq, g1h, nreed, Vr):
    nz, ny = alpha.shape
    bad = 0
    for i in prange(nz):
        for j in range(ny):
            if not _prim(U, alpha, P, i, j, rho0, p0, c0sq, g1h, nreed, Vr):
                bad += 1
    return bad


@njit(cache=True)
def k_prims_rows(U, alpha, P, j0, j1, rho0, p0, c0sq, g1h, nreed, Vr):
    bad = 0
    for i in range(nreed):
        for j in range(j0, j1):
            if not _prim(U, alpha, P, i, j, rho0, p0, c0sq, g1h, nreed, Vr):
                bad += 1
    return bad


@njit(parallel=True, fastmath=FM)
def k_flux_z(P, alpha, hz, idz, idzc, FL, FR, rho0, p0, c0sq, g1h, mu, nreed, Vr):
    """Fluxes through faces normal to z (MUSCL slopes computed on the fly).
    FL = flux as seen by the left cell, FR = by the right cell (they differ only
    where part of the face is a wall)."""
    nz, ny = alpha.shape
    for i in prange(nz + 1):
        for j in range(ny):
            aL = alpha[i - 1, j] if i > 0 else 0.0
            aR = alpha[i, j] if i < nz else 0.0
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
                g0 = (P[0, Rr, j] - P[0, L, j]) * ic
                g1 = (P[1, Rr, j] - P[1, L, j]) * ic
                g2 = (P[2, Rr, j] - P[2, L, j]) * ic
                rL = P[0, L, j]
                nL = P[1, L, j]
                tL = P[2, L, j]
                if L >= 1 and aL == 1.0 and aR == 1.0 and alpha[L - 1, j] == 1.0:
                    il = idzc[L - 1]
                    x = hz[L]
                    rL += x * _mc((P[0, L, j] - P[0, L - 1, j]) * il, g0)
                    nL += x * _mc((P[1, L, j] - P[1, L - 1, j]) * il, g1)
                    tL += x * _mc((P[2, L, j] - P[2, L - 1, j]) * il, g2)
                rR = P[0, Rr, j]
                nR = P[1, Rr, j]
                tR = P[2, Rr, j]
                if Rr <= nz - 2 and aL == 1.0 and aR == 1.0 and alpha[Rr + 1, j] == 1.0:
                    ir = idzc[Rr]
                    x = hz[Rr]
                    rR -= x * _mc(g0, (P[0, Rr + 1, j] - P[0, Rr, j]) * ir)
                    nR -= x * _mc(g1, (P[1, Rr + 1, j] - P[1, Rr, j]) * ir)
                    tR -= x * _mc(g2, (P[2, Rr + 1, j] - P[2, Rr, j]) * ir)
                f0, f1, f2 = _rusanov(rL, nL, tL, P[4, L, j], P[5, L, j],
                                      rR, nR, tR, P[4, Rr, j], P[5, Rr, j], rho0, p0, c0sq, g1h)
                f1 -= mu * g1
                f2 -= mu * g2
                l0 = beta * f0
                l1 = beta * f1
                l2 = beta * f2
                r0 = l0
                r1 = l1
                r2 = l2
            wl = aL - beta
            if wl > 0.0:  # wall on the +z side of the left cell
                ii = i - 1
                un = P[1, ii, j]
                pw = P[3, ii, j] + P[0, ii, j] * P[4, ii, j] * un
                if i == nz:  # outer slip wall
                    l1 += wl * pw
                else:
                    k = 2.0 * mu * idz[ii]
                    l1 += wl * (pw + k * un)
                    l2 += wl * (k * P[2, ii, j])
            wr = aR - beta
            if wr > 0.0:  # wall on the -z side of the right cell
                un = P[1, i, j]
                pw = P[3, i, j] - P[0, i, j] * P[4, i, j] * un
                if i == 0:  # symmetry plane (slip)
                    r1 += wr * pw
                else:
                    k = 2.0 * mu * idz[i]
                    wt = Vr if i == nreed else 0.0  # the reed edge moves in y
                    r1 += wr * (pw - k * un)
                    r2 -= wr * k * (P[2, i, j] - wt)
            FL[0, i, j] = l0
            FL[1, i, j] = l1
            FL[2, i, j] = l2
            FR[0, i, j] = r0
            FR[1, i, j] = r1
            FR[2, i, j] = r2


@njit(inline="always", fastmath=FM)
def _yflux(P, alpha, yf, hy, idy, idyc, i, j, ny, nreed, Yb, Yt, rho0, p0, c0sq, g1h, mu):
    """Flux through the y-face j of column i. Returns (as seen by bottom cell,
    as seen by top cell), components (mass, z-mom, y-mom)."""
    b0 = 0.0
    b1 = 0.0
    b2 = 0.0
    t0 = 0.0
    t1 = 0.0
    t2 = 0.0
    closed = (i < nreed) and (yf[j] >= Yb) and (yf[j] <= Yt)
    if not closed:
        aB = alpha[i, j - 1] if j > 0 else 0.0
        aT = alpha[i, j] if j < ny else 0.0
        if aB > 0.0 and aT > 0.0:
            B = j - 1
            T = j
            ic = idyc[B]
            g0 = (P[0, i, T] - P[0, i, B]) * ic
            g1 = (P[1, i, T] - P[1, i, B]) * ic
            g2 = (P[2, i, T] - P[2, i, B]) * ic
            rB = P[0, i, B]
            tB = P[1, i, B]
            nB = P[2, i, B]
            if B >= 1 and aB == 1.0 and aT == 1.0 and alpha[i, B - 1] == 1.0:
                il = idyc[B - 1]
                x = hy[B]
                rB += x * _mc((P[0, i, B] - P[0, i, B - 1]) * il, g0)
                tB += x * _mc((P[1, i, B] - P[1, i, B - 1]) * il, g1)
                nB += x * _mc((P[2, i, B] - P[2, i, B - 1]) * il, g2)
            rT = P[0, i, T]
            tT = P[1, i, T]
            nT = P[2, i, T]
            if T <= ny - 2 and aB == 1.0 and aT == 1.0 and alpha[i, T + 1] == 1.0:
                ir = idyc[T]
                x = hy[T]
                rT -= x * _mc(g0, (P[0, i, T + 1] - P[0, i, T]) * ir)
                tT -= x * _mc(g1, (P[1, i, T + 1] - P[1, i, T]) * ir)
                nT -= x * _mc(g2, (P[2, i, T + 1] - P[2, i, T]) * ir)
            f0, f1, f2 = _rusanov(rB, nB, tB, P[4, i, B], P[5, i, B],
                                  rT, nT, tT, P[4, i, T], P[5, i, T], rho0, p0, c0sq, g1h)
            b0 = f0
            b1 = f2 - mu * g1
            b2 = f1 - mu * g2
            t0 = b0
            t1 = b1
            t2 = b2
        elif aB > 0.0:  # wall on the +y side of the bottom cell
            jj = j - 1
            un = P[2, i, jj]
            pw = P[3, i, jj] + P[0, i, jj] * P[4, i, jj] * un
            if j == ny:
                b2 = pw
            else:
                k = 2.0 * mu * idy[jj]
                b2 = pw + k * un
                b1 = k * P[1, i, jj]
        elif aT > 0.0:  # wall on the -y side of the top cell
            un = P[2, i, j]
            pw = P[3, i, j] - P[0, i, j] * P[4, i, j] * un
            if j == 0:
                t2 = pw
            else:
                k = 2.0 * mu * idy[j]
                t2 = pw - k * un
                t1 = -k * P[1, i, j]
    return b0, b1, b2, t0, t1, t2


@njit(parallel=True, fastmath=FM)
def k_update(P, Pn, Ucur, Uold, Unew, ca, cb, cc, dt, FL, FR, alpha, yf, hy, idz, idy, idyc,
             sig, sigm, is_up, rho_up, rho_dn, rho0, p0, c0sq, g1h, mu, nreed, Yb, Yt, Vr,
             jt, jb, jpb, nslot, qslot):
    """y-fluxes + divergence + sponge + RK update + new primitives, column by column.
    Unew = ca*Uold + cb*Ucur + cc*dt*R(Ucur)."""
    nz, ny = alpha.shape
    bad = 0
    for i in prange(nz):
        _, _, _, ft0, ft1, ft2 = _yflux(P, alpha, yf, hy, idy, idyc, i, 0, ny, nreed, Yb, Yt,
                                        rho0, p0, c0sq, g1h, mu)
        cdt = cc * dt
        for j in range(ny):
            b0, b1, b2, t0, t1, t2 = _yflux(P, alpha, yf, hy, idy, idyc, i, j + 1, ny, nreed,
                                            Yb, Yt, rho0, p0, c0sq, g1h, mu)
            if j + 1 == jpb and i < nslot:
                qslot[i] = t0
            if alpha[i, j] > 0.0:
                iz = idz[i]
                iy = idy[j]
                q0 = -(FL[0, i + 1, j] - FR[0, i, j]) * iz - (b0 - ft0) * iy
                q1 = -(FL[1, i + 1, j] - FR[1, i, j]) * iz - (b1 - ft1) * iy
                q2 = -(FL[2, i + 1, j] - FR[2, i, j]) * iz - (b2 - ft2) * iy
                s = sig[i, j]
                if s > 0.0 and alpha[i, j] == 1.0:
                    rt = rho_up if is_up[i, j] else rho_dn
                    q0 += s * (rt - Ucur[0, i, j])
                    sm = sigm[i, j]
                    q1 -= sm * Ucur[1, i, j]
                    q2 -= sm * Ucur[2, i, j]
                Unew[0, i, j] = ca * Uold[0, i, j] + cb * Ucur[0, i, j] + cdt * q0
                Unew[1, i, j] = ca * Uold[1, i, j] + cb * Ucur[1, i, j] + cdt * q1
                Unew[2, i, j] = ca * Uold[2, i, j] + cb * Ucur[2, i, j] + cdt * q2
            else:
                Unew[0, i, j] = 0.0
                Unew[1, i, j] = 0.0
                Unew[2, i, j] = 0.0
            # cells next to the moving reed get their primitives after the geometry update
            if not (i < nreed and j >= jb - 2 and j <= jt + 2):
                if not _prim(Unew, alpha, Pn, i, j, rho0, p0, c0sq, g1h, nreed, Vr):
                    bad += 1
            ft0 = t0
            ft1 = t1
            ft2 = t2
    return bad


@njit(cache=True)
def k_reed_walls(P, Unew, cdt, dz, dy, alpha, mu, nreed, jt, jb, Vr):
    """Pressure/shear on the moving reed's faces. Adds the wall fluxes (times cdt)
    to the cut cells and returns the y-force per unit length on the HALF reed."""
    F = 0.0
    for i in range(nreed):
        # top face (fluid above, wall below)
        hd = 0.5 * dy[jt]
        vv = P[2, i, jt]
        pw = P[3, i, jt] + P[0, i, jt] * P[4, i, jt] * (Vr - vv)
        fw = pw - mu * (vv - Vr) / hd
        Unew[2, i, jt] += cdt * fw / dy[jt]
        Unew[1, i, jt] -= cdt * mu * P[1, i, jt] / hd / dy[jt]
        F -= fw * dz[i]
        # bottom face (fluid below, wall above)
        hd = 0.5 * dy[jb]
        vv = P[2, i, jb]
        pw = P[3, i, jb] + P[0, i, jb] * P[4, i, jb] * (vv - Vr)
        fw = pw + mu * (vv - Vr) / hd
        Unew[2, i, jb] -= cdt * fw / dy[jb]
        Unew[1, i, jb] -= cdt * mu * P[1, i, jb] / hd / dy[jb]
        F += fw * dz[i]
    # viscous drag of the gap flow on the reed's side edge
    ig = nreed
    hd = 0.5 * dz[ig]
    for j in range(jb, jt + 1):
        w = 1.0 - alpha[0, j]
        if w > 0.0 and alpha[ig, j] > 0.0:
            F += mu * (P[2, ig, j] - Vr) / hd * w * dy[j]
    return F


@njit(inline="always")
def _mix(U, alpha, dy, i, a, b):
    for k in range(3):
        tot = U[k, i, a] * dy[a] + U[k, i, b] * dy[b]
        vol = alpha[i, a] * dy[a] + alpha[i, b] * dy[b]
        q = tot / vol
        U[k, i, a] = alpha[i, a] * q
        U[k, i, b] = alpha[i, b] * q


@njit(cache=True)
def k_geometry(Y, h, yf, dy, nreed, alpha, U, amix):
    """Update fluid fractions of the reed columns for reed bottom at y=Y, move the
    content of newly covered cells to the adjacent cut cell, merge small cut cells."""
    ny = dy.shape[0]
    Yb = Y
    Yt = Y + h
    jb = np.searchsorted(yf, Yb, side="left") - 1   # yf[jb] <  Yb <= yf[jb+1]
    jt = np.searchsorted(yf, Yt, side="right") - 1  # yf[jt] <= Yt <  yf[jt+1]
    j0 = max(jb - 3, 0)
    j1 = min(jt + 4, ny)
    for j in range(j0, j1):
        if j > jb and j < jt:
            a = 0.0
        else:
            ov = min(yf[j + 1], Yt) - max(yf[j], Yb)
            a = 1.0 - ov / dy[j] if ov > 0.0 else 1.0
            if a < 0.0:
                a = 0.0
        for i in range(nreed):
            alpha[i, j] = a
    ymid = Yb + 0.5 * h
    for i in range(nreed):
        for j in range(jb + 1, jt):
            if U[0, i, j] != 0.0 or U[1, i, j] != 0.0 or U[2, i, j] != 0.0:
                tgt = jt if 0.5 * (yf[j] + yf[j + 1]) > ymid else jb
                f = dy[j] / dy[tgt]
                for k in range(3):
                    U[k, i, tgt] += U[k, i, j] * f
                    U[k, i, j] = 0.0
        if alpha[i, jt] < amix:
            _mix(U, alpha, dy, i, jt, jt + 1)
        if alpha[i, jb] < amix:
            _mix(U, alpha, dy, i, jb, jb - 1)
    return jt, jb


@njit
def advance(nsteps, step0, dt, U, U1, alpha, P, Pn, FL, FR, qslot,
            hz, idz, idzc, yf, dy, hy, idy, idyc, dz, sig, sigm, is_up, par, ipar, rs,
            mic_i, mic_j, mic_w, rec, rec_every):
    rho0 = par[0]; p0 = par[1]; c0sq = par[2]; g1h = par[3]; mu = par[4]
    h = par[5]; mr = par[6]; kr = par[7]; cr = par[8]; Yr = par[9]
    amix = par[10]; rho_up = par[11]; t0 = par[12]
    nreed = ipar[0]; nslot = ipar[1]; jpb = ipar[2]; jt = ipar[3]; jb = ipar[4]
    irec = ipar[5]
    ny = dy.shape[0]
    Y = rs[0]; V = rs[1]
    nmic = mic_w.shape[0]
    rho_dn = rho0
    status = 0
    for n in range(nsteps):
        step = step0 + n
        # ---------------- stage 1:  U1 = U + dt L(U) ----------------
        if step % rec_every == 0 and irec < rec.shape[0]:
            rec[irec, 0] = t0 + n * dt
            rec[irec, 1] = Y
            rec[irec, 2] = V
            for m in range(nmic):
                s = 0.0
                for q in range(4):
                    s += mic_w[m, q] * P[3, mic_i[m, q], mic_j[m, q]]
                rec[irec, 5 + m] = s
        k_flux_z(P, alpha, hz, idz, idzc, FL, FR, rho0, p0, c0sq, g1h, mu, nreed, V)
        bad = k_update(P, Pn, U, U, U1, 0.0, 1.0, 1.0, dt, FL, FR, alpha, yf, hy, idz, idy, idyc,
                       sig, sigm, is_up, rho_up, rho_dn, rho0, p0, c0sq, g1h, mu, nreed,
                       Y, Y + h, V, jt, jb, jpb, nslot, qslot)
        F = k_reed_walls(P, U1, dt, dz, dy, alpha, mu, nreed, jt, jb, V)
        if step % rec_every == 0 and irec < rec.shape[0]:
            rec[irec, 3] = 2.0 * F
            qs = 0.0
            for i in range(nslot):
                qs += qslot[i] * dz[i]
            rec[irec, 4] = 2.0 * qs / rho0
            irec += 1
        A = (2.0 * F - cr * V - kr * (Y - Yr)) / mr
        Y1 = Y + dt * V
        V1 = V + dt * A
        jt1, jb1 = k_geometry(Y1, h, yf, dy, nreed, alpha, U1, amix)
        bad += k_prims_rows(U1, alpha, Pn, max(min(jb, jb1) - 4, 0), min(max(jt, jt1) + 5, ny),
                            rho0, p0, c0sq, g1h, nreed, V1)
        if bad > 0:
            status = -(n + 1)
            break
        tmp = P
        P = Pn
        Pn = tmp
        # ---------------- stage 2:  U = (U + U1 + dt L(U1)) / 2 ----------------
        k_flux_z(P, alpha, hz, idz, idzc, FL, FR, rho0, p0, c0sq, g1h, mu, nreed, V1)
        bad = k_update(P, Pn, U1, U, U, 0.5, 0.5, 0.5, dt, FL, FR, alpha, yf, hy, idz, idy, idyc,
                       sig, sigm, is_up, rho_up, rho_dn, rho0, p0, c0sq, g1h, mu, nreed,
                       Y1, Y1 + h, V1, jt1, jb1, jpb, nslot, qslot)
        F1 = k_reed_walls(P, U, 0.5 * dt, dz, dy, alpha, mu, nreed, jt1, jb1, V1)
        A1 = (2.0 * F1 - cr * V1 - kr * (Y1 - Yr)) / mr
        Y = 0.5 * (Y + Y1 + dt * V1)
        V = 0.5 * (V + V1 + dt * A1)
        jt, jb = k_geometry(Y, h, yf, dy, nreed, alpha, U, amix)
        bad += k_prims_rows(U, alpha, Pn, max(min(jb, jb1) - 4, 0), min(max(jt, jt1) + 5, ny),
                            rho0, p0, c0sq, g1h, nreed, V)
        tmp = P
        P = Pn
        Pn = tmp
        if bad > 0:
            status = -(n + 1)
            break
    rs[0] = Y
    rs[1] = V
    ipar[3] = jt
    ipar[4] = jb
    ipar[5] = irec
    return status


# ----------------------------------------------------------------------------
# simulation driver
# ----------------------------------------------------------------------------
class ReedSim:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        C = cfg
        self.rho0 = C.rho0()
        self.c0 = C.c0()
        self.g1 = C.gamma - 1.0
        self.g1h = 0.5 * self.g1 / self.rho0
        self.c0sq = self.c0 ** 2

        # ---- snapped geometry -------------------------------------------
        dz0, dy0 = C.dz_fine, C.dy_fine
        self.nreed = int(round(0.5 * C.reed_b / dz0))
        self.ngap = max(1, int(round(C.gap / dz0)))
        self.b = 2 * self.nreed * dz0
        self.g = self.ngap * dz0
        n_tp = int(round(C.plate_t / dy0))
        self.tp = n_tp * dy0
        z_core = (self.nreed + self.ngap + int(round(C.core_side / dz0))) * dz0
        y_core_lo = -(n_tp + int(round(C.core_dn / dy0))) * dy0
        y_core_hi = int(round(C.core_up / dy0)) * dy0
        self.y_ch_bot = -self.tp - C.chamber_h          # chamber floor (inner)
        y_bot = (self.y_ch_bot - C.wall_t if C.chamber else -self.tp) - C.room_below
        self.zf = make_axis(0.0, C.z_max, 0.0, z_core, dz0, C.d_coarse, C.stretch)
        self.yf = make_axis(y_bot, C.y_top, y_core_lo, y_core_hi, dy0, C.d_coarse, C.stretch)
        self.zc = 0.5 * (self.zf[1:] + self.zf[:-1])
        self.yc = 0.5 * (self.yf[1:] + self.yf[:-1])
        self.dz = np.diff(self.zf)
        self.dy = np.diff(self.yf)
        self.hz, self.hy = 0.5 * self.dz, 0.5 * self.dy
        self.idz, self.idy = 1.0 / self.dz, 1.0 / self.dy
        self.idzc, self.idyc = 1.0 / np.diff(self.zc), 1.0 / np.diff(self.yc)
        nz, ny = len(self.zc), len(self.yc)
        self.nz, self.ny = nz, ny
        self.jpb = int(np.argmin(abs(self.yf + self.tp)))  # face index of plate underside
        self.jpt = int(np.argmin(abs(self.yf)))
        assert abs(self.yf[self.jpb] + self.tp) < 1e-12 and abs(self.yf[self.jpt]) < 1e-12

        Z, Yc = np.meshgrid(self.zc, self.yc, indexing="ij")
        solid = np.zeros((nz, ny), bool)
        slot_edge = (self.nreed + self.ngap) * dz0
        solid |= (Z > slot_edge) & (Yc > -self.tp) & (Yc < 0.0)          # reed plate
        if C.chamber:
            w2, wt, ch_bot = 0.5 * C.chamber_w, C.wall_t, self.y_ch_bot
            solid |= (Z > w2) & (Z < w2 + wt) & (Yc > ch_bot - wt) & (Yc < -self.tp)   # side wall
            solid |= (Z > 0.5 * C.outlet_w) & (Z < w2 + wt) & (Yc > ch_bot - wt) & (Yc < ch_bot)  # floor
        self.solid = solid

        # ---- reed ---------------------------------------------------------
        self.f_vac = C.f_vacuum()
        w0 = 2 * math.pi * self.f_vac
        self.m_r = C.rho_s * self.b * C.reed_h            # kg/m
        self.k_r = self.m_r * w0 ** 2                      # N/m^2
        self.c_r = self.m_r * w0 / C.Q_mech                # N s/m^2
        self.Y_rest = C.set_gap
        Y_init = C.set_gap - C.kick
        # make sure the reed stays inside the fine core
        assert Y_init > y_core_lo + 0.3e-3 and Y_init + C.reed_h < y_core_hi - 0.3e-3

        # ---- pressures / densities ---------------------------------------
        self.rho_up = self._rho_of_p(C.p_amb + C.dp)

        # ---- state ----------------------------------------------------------
        self.alpha = np.where(solid, 0.0, 1.0)
        self.U = np.zeros((3, nz, ny))
        jt, jb = k_geometry(Y_init, C.reed_h, self.yf, self.dy, self.nreed, self.alpha,
                            np.zeros((3, nz, ny)), 0.5)
        rho_init = self.rho0 + (self.rho_up - self.rho0) * (Yc > 0.0) * (C.t_ramp <= 0.0)
        self.U[0] = self.alpha * rho_init
        jt, jb = k_geometry(Y_init, C.reed_h, self.yf, self.dy, self.nreed, self.alpha, self.U, 0.5)
        self.U1 = np.zeros_like(self.U)
        self.P = np.zeros((6, nz, ny))    # primitives rho, u, v, p, c, 1/c (current)
        self.Pn = np.zeros((6, nz, ny))   # scratch buffer
        self.FL = np.zeros((3, nz + 1, ny))
        self.FR = np.zeros((3, nz + 1, ny))
        self.qslot = np.zeros(nz)
        self.rs = np.array([Y_init, 0.0])
        k_prims(self.U, self.alpha, self.P, self.rho0, C.p_amb, self.c0sq, self.g1h, self.nreed, 0.0)

        # ---- sponge ---------------------------------------------------------
        def ramp(d):
            return np.clip((C.sponge - d) / C.sponge, 0, 1) ** 2
        sig = np.maximum.reduce([ramp(C.y_top - Yc), ramp(Yc - self.yf[0]), ramp(C.z_max - Z)])
        self.sig = C.sponge_sigma * sig
        self.is_up = (Yc > 0.0)
        self.sigm = np.where(self.is_up, 0.0, 0.3 * self.sig)

        # ---- mics -------------------------------------------------------------
        if not C.mics:
            if C.chamber:
                yb = self.y_ch_bot - C.wall_t
                C.mics = [["chamber", 2.5e-3, -self.tp - 0.6 * C.chamber_h],
                          ["outlet", 0.0, yb - 3e-3],
                          ["room", 8e-3, yb - 12e-3],
                          ["bellows", 6e-3, 6e-3]]
            else:
                C.mics = [["below", 0.0, -self.tp - 3e-3], ["room", 8e-3, -12e-3],
                          ["bellows", 6e-3, 6e-3]]
        self.mic_names = [m[0] for m in C.mics]
        nm = len(C.mics)
        self.mic_i = np.zeros((nm, 4), np.int64)
        self.mic_j = np.zeros((nm, 4), np.int64)
        self.mic_w = np.zeros((nm, 4))
        for k, (_, zm, ym) in enumerate(C.mics):
            zm = abs(zm)
            i1 = int(np.clip(np.searchsorted(self.zc, zm), 1, nz - 1))
            j1 = int(np.clip(np.searchsorted(self.yc, ym), 1, ny - 1))
            i0, j0 = i1 - 1, j1 - 1
            wz = np.clip((zm - self.zc[i0]) / (self.zc[i1] - self.zc[i0]), 0, 1)
            wy = np.clip((ym - self.yc[j0]) / (self.yc[j1] - self.yc[j0]), 0, 1)
            ii = [i0, i1, i0, i1]
            jj = [j0, j0, j1, j1]
            ww = [(1 - wz) * (1 - wy), wz * (1 - wy), (1 - wz) * wy, wz * wy]
            if any(solid[a, b] for a, b in zip(ii, jj)) or any(a < self.nreed for a in ii) and \
                    (Y_init - 1.5e-3 < ym < Y_init + 2e-3):
                raise ValueError(f"mic {C.mics[k][0]} sits in/next to a solid or the reed path")
            self.mic_i[k], self.mic_j[k], self.mic_w[k] = ii, jj, ww

        # ---- time step ----------------------------------------------------
        c_max = math.sqrt(self.c0sq * (1 + self.g1 * (self.rho_up - self.rho0) / self.rho0))
        # fastest expected air speed: Bernoulli jet, or air pumped through the side gaps
        # by the moving tongue (continuity: (b/2)/g times the reed velocity)
        u_jet = 1.4 * math.sqrt(2 * max(C.dp, 0.0) / self.rho0)
        u_pump = 0.5 * self.b / self.g * 2 * math.pi * self.f_vac * max(C.kick, 0.6e-3)
        u_est = max(u_jet, u_pump) + 15.0
        self.dt = C.cfl / ((c_max + u_est) * (1 / self.dz.min() + 1 / self.dy.min()))
        self.nsteps = int(math.ceil(C.t_end / self.dt))

        self.par = np.array([self.rho0, C.p_amb, self.c0sq, self.g1h, C.mu, C.reed_h,
                             self.m_r, self.k_r, self.c_r, self.Y_rest, 0.5, self.rho_up, 0.0])
        self.ipar = np.array([self.nreed, self.nreed + self.ngap, self.jpb, jt, jb, 0], np.int64)
        self.rec = np.zeros((self.nsteps // C.rec_every + 2, 5 + nm))

    def _rho_of_p(self, pres):
        return self.rho0 * (pres / self.cfg.p_amb) ** (1.0 / self.cfg.gamma)

    def info(self):
        C = self.cfg
        return dict(
            nz=self.nz, ny=self.ny, cells=self.nz * self.ny, dt=self.dt, nsteps=self.nsteps,
            rho0=self.rho0, c0=self.c0, nu=C.mu / self.rho0,
            b_eff=self.b, gap_eff=self.g, plate_t_eff=self.tp,
            f_vacuum=self.f_vac, m_reed_per_len=self.m_r, k_reed_per_len=self.k_r,
            jet_speed_bernoulli=math.sqrt(2 * C.dp / self.rho0) if C.dp > 0 else 0.0,
            added_mass_ratio_free_strip=math.pi * self.rho0 * self.b / (4 * C.rho_s * C.reed_h),
        )

    def run(self, out_dir, chunk=250, verbose=True):
        C = self.cfg
        os.makedirs(out_dir, exist_ok=True)
        T0 = 1.0 / self.f_vac
        t_snap0 = C.t_end - C.snap_cycles * T0
        snap_dt = T0 / C.snap_per_cycle
        next_snap = t_snap0
        snaps = []
        step = 0
        wall0 = time.time()
        last_print = wall0
        while step < self.nsteps:
            t = step * self.dt
            if C.t_ramp > 0:
                s = min(1.0, t / C.t_ramp)
                s = s * s * (3 - 2 * s)
                self.par[11] = self._rho_of_p(C.p_amb + C.dp * s)
            self.par[12] = t
            n = min(chunk, self.nsteps - step)
            status = advance(n, step, self.dt, self.U, self.U1, self.alpha, self.P, self.Pn,
                             self.FL, self.FR, self.qslot,
                             self.hz, self.idz, self.idzc, self.yf, self.dy, self.hy,
                             self.idy, self.idyc, self.dz, self.sig, self.sigm, self.is_up, self.par, self.ipar, self.rs,
                             self.mic_i, self.mic_j, self.mic_w, self.rec, C.rec_every)
            if status < 0:
                raise RuntimeError(f"solver blew up near t={t:.6e}s (step {step - status - 1})")
            step += n
            t = step * self.dt
            if t >= next_snap and next_snap <= C.t_end:
                snaps.append(dict(t=t, Y=self.rs[0], p=(self.P[3] - C.p_amb).astype(np.float32),
                                  u=self.P[1].astype(np.float32), v=self.P[2].astype(np.float32),
                                  alpha=self.alpha.astype(np.float32)))
                next_snap += snap_dt
            now = time.time()
            if verbose and (now - last_print > 10 or step >= self.nsteps):
                last_print = now
                _, u, v, _, c, _ = self.P
                cfl = self.dt * np.max(((np.abs(u) + c) / self.dz[:, None]
                                        + (np.abs(v) + c) / self.dy[None, :])[self.alpha > 0])
                rate = step / (now - wall0)
                print(f"[{C.name}] t={t * 1e3:7.3f} ms  step {step}/{self.nsteps}  "
                      f"Y={self.rs[0] * 1e3:+.3f} mm  cfl={cfl:.2f}  "
                      f"{rate:.0f} steps/s  eta {(self.nsteps - step) / rate:.0f}s", flush=True)
        self.wall_time = time.time() - wall0
        self.save(out_dir, snaps)

    def save(self, out_dir, snaps):
        C = self.cfg
        nrec = int(self.ipar[5])
        rec = self.rec[:nrec]
        info = self.info()
        info["wall_time_s"] = getattr(self, "wall_time", None)
        with open(os.path.join(out_dir, "config.json"), "w") as f:
            json.dump(dict(config=asdict(C), info=info), f, indent=2)
        np.savez_compressed(
            os.path.join(out_dir, "series.npz"),
            t=rec[:, 0], Y=rec[:, 1] - self.Y_rest, V=rec[:, 2], F=rec[:, 3], Qslot=rec[:, 4],
            mics=rec[:, 5:] - C.p_amb, mic_names=np.array(self.mic_names),
            mic_pos=np.array([[m[1], m[2]] for m in C.mics]),
            f_vacuum=self.f_vac, p_amb=C.p_amb, dp=C.dp, rho0=self.rho0, c0=self.c0,
            reed_L=C.reed_L)
        if snaps:
            np.savez_compressed(
                os.path.join(out_dir, "snapshots.npz"),
                zf=self.zf, yf=self.yf, t=np.array([s["t"] for s in snaps]),
                Y=np.array([s["Y"] for s in snaps]), h=C.reed_h, b=self.b, tp=self.tp,
                p=np.stack([s["p"] for s in snaps]), u=np.stack([s["u"] for s in snaps]),
                v=np.stack([s["v"] for s in snaps]), alpha=np.stack([s["alpha"] for s in snaps]),
                solid=self.solid)


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    d = Config()
    for k, v in asdict(d).items():
        if k == "mics":
            continue
        flag = "--" + k.replace("_", "-")
        if isinstance(v, bool):
            ap.add_argument(flag, type=lambda s: s.lower() in ("1", "true", "yes"), default=v)
        else:
            ap.add_argument(flag, type=type(v), default=v)
    ap.add_argument("--out", default="runs")
    ap.add_argument("--no-analysis", action="store_true")
    return ap


def main():
    args = build_parser().parse_args()
    kw = {k: getattr(args, k) for k in asdict(Config()) if k != "mics"}
    cfg = Config(**kw)
    sim = ReedSim(cfg)
    print(json.dumps(sim.info(), indent=1))
    out_dir = os.path.join(args.out, cfg.name)
    sim.run(out_dir)
    if not args.no_analysis:
        from analysis import analyze_run
        analyze_run(out_dir)


if __name__ == "__main__":
    main()
