# Model details

Technical notes behind the [results page](../README.md).

## 2-D accordion free-reed simulation

Compressible Navier–Stokes air coupled to a spring-steel reed tongue, built
to study how **ambient air pressure** changes the reed's pitch. There are two
2-D slicings:

* **`reed_side.py` (current): side view, along the tongue.** It has a bending
  cantilever (3 modes), the set profile `w0 = s_tip (x/L)^2`, a resolved tip
  clearance, a reed chamber along the tongue with an outlet, and a sub-grid
  "leak" through the tongue for the side openings. At every column a local
  gap-flow equation
  `rho l_eff dU/dt = dp - rho U|U|/(2 Cd^2) - 12 mu l_v U / a^2`, with
  `a = sqrt(g^2 + d^2)`, passes air from the cell above the tongue to the cell
  below. **This model self-oscillates from rest** (threshold 0.3–0.6 kPa at set
  0.3 mm, higher for a larger set).
* `reed2d.py` / `reed_gpu.py`: cross-section across the tongue (below; geometry in [cross_section_geometry.png](cross_section_geometry.png)). It
  never self-oscillates: the side-gap flow only damps the reed, and grid
  refinement doesn't change that. Kept for the plucked-reed air-loading
  numbers.

## Result: pitch vs ambient pressure (side-view model)

A4 reed, set 0.3 mm, Cd 0.7, leak jet on, 60 ms per run (~26 cycles), air at
20 °C. Pitch is compared at **matched tip amplitude** because the reed is
still growing slowly at 60 ms and pitch rises with amplitude
(`runs_gpu/sw2_dp_compare.png`). Cents are relative to the in-vacuo reed
(440.1 Hz).

| ambient | ≈ altitude | ρ0 kg/m³ | dp 1 kPa @ 1.28 mm | dp 2 kPa @ 1.70 mm |
|---|---|---|---|---|
| 70 kPa | 9,900 ft | 0.832 | −23.6 c | −29.0 c |
| 85 kPa | 4,800 ft | 1.010 | −26.4 c | — |
| 101.3 kPa | sea level | 1.204 | −29.7 c | −34.0 c |
| 115 kPa | — | 1.367 | −31.5 c | −37.0 c |
| 130 kPa | — | 1.545 | −34.3 c | −41.0 c |

* Slope: **−1.8 c per +10 kPa** at dp 1 kPa and −2.0 c at dp 2 kPa. Denser air
  flattens the reed (added mass wins over the chamber air spring). From sea
  level to 10,000 ft the reed goes about **+6 c sharp**.
* At fixed blowing pressure, thinner air gives a slightly larger amplitude
  (1.98 mm at 70 kPa vs 1.87 mm at 130 kPa after 60 ms). Because pitch rises
  with amplitude, comparing the last cycles alone gives a smaller slope of
  −1.5 c per 10 kPa.
* Extrapolated to zero density, the offset is about −11 c (dp 1 kPa) to −15 c
  (dp 2 kPa). That part comes from the blowing and flow, not from air mass.
* Grid check (101.3 kPa, dp 1 kPa): refining to 532×443 cells (min 16.7 µm)
  moves the pitch by 0.4–1.2 c and the amplitude growth by under 1%. Pressure
  differences between runs on the same grid are well above that noise.
* Measured mean pressures: bellows 996–998 Pa (1987 Pa at dp 2 kPa), chamber
  +100–300 Pa. The room mic sits in the outlet jet's recirculation and reads
  −75 to −330 Pa, which is local suction, not a boundary error.
* The absolute flattening (~−30 c at sea level) is probably too large: the
  2-D slab confines the air more than a real reed's 3-D surroundings. The
  slope is the more trustworthy number, but treat it as an upper bound too.

> Fixed: until the sponge strength was raised to 1e6 1/s, the sponges couldn't
> hold the bellows/room pressures under throughflow. Blown cross-section runs
> before that fix saw only ~0.3–0.6 kPa at a nominal 1 kPa. Plucked runs
> were unaffected.

## Model

```
        y ^          bellows (p_amb + dp)
          |        [=====reed=====]          <- steel strip, moves in y
     -----+--------+  slot  +--------------  <- reed plate, 1.5 mm
          |   +----+ chamber+----+           <- reed chamber (2-D proxy)
          |   +------+    +------+           <- outlet
          |            room (p_amb)
          +-----------------------------> z     (mirror-symmetric about z=0)
```

* **Air**: barotropic (isentropic) compressible N–S. At fixed temperature
  ambient pressure enters *only* through `rho0 = p_amb/(R T)`: c and μ do not
  depend on pressure. So it changes the fluid inertia (added mass), jet speed
  `sqrt(2 dp/rho0)`, Reynolds number, and the air's stiffness `rho0 c^2 = γ p_amb`.
* **Reed**: an A4 treble tongue (L 23.75 mm, b 2.5 mm, h 0.30 mm, steel,
  f_vacuum = 440.1 Hz). The section is a rigid strip on a spring tuned to the
  first cantilever mode, per unit length. With strip theory the air-loading
  ratio of the section equals the cantilever's modal one.
* **Numerics**: finite volume on a stretched Cartesian grid, MUSCL + Rusanov
  with a Thornber low-Mach fix, SSP-RK2. The moving reed is a conservative
  cut-cell with small-cell merging. Walls are no-slip, and the far field uses
  sponge layers. The time step (~27 ns) is set by the speed of sound in the
  25 µm cells needed to resolve the 50 µm side clearance.

## Files

| file | what |
|---|---|
| `reed_side.py` | side-view model (GPU): `python reed_side.py --prefix s --p-amb 70e3 101325 --dp 1000 --kick=-0.5e-3` |
| `regrid.py` | adapt the grid to a pilot run's resolution maps (`diag.npz`) → `--grid-file` |
| `reed2d.py` | config, grid, CPU solver (numba). `python reed2d.py -h` |
| `reed_gpu.py` | batched mixed-precision CUDA solver (numba-cuda, CUDA graphs) |
| `reed_gpu_fp64.py` | all-FP64 GPU reference (used by `reed_gpu.py --check`) |
| `analysis.py` | per-run pitch (cycle by cycle), WAVs, plots |
| `compare.py` | compare runs at *matched amplitude* (pitch vs ambient pressure) |
| `sweep.py` | CPU multi-process sweep driver |
| `poster_chart.py` | altitude chart from a `compare.py` JSON |
| `render_fields.py` | speed / vorticity / pressure stills, cycle movie and GIF |

Each run writes `runs/<name>/`: `series.npz` (reed Y, V, force, slot flow,
mic pressures), `summary.json`, `overview.png`, `fields.png`, and
`mic_<name>.wav` / `mic_<name>_loop.wav` (raw signal, and the last single cycle
tiled into a 2 s tone).

```bash
# GPU: one batch = all combinations, advanced together
python reed_gpu.py --prefix sw --t-end 0.04 --p-amb 70e3 85e3 101325 115e3 130e3 --dp 0 1000
python compare.py runs/sw_*            # pitch at matched amplitude vs p_amb
python reed_gpu.py --check             # mixed precision vs FP64
python reed_gpu.py --bench
```

Performance (A4 grid, 28.7k cells, 40 ms ≈ 17 cycles ≈ 1.5 M steps):
i7-9750H CPU 1.4 ms/step (≈35 min/run); RTX 4090 86 µs/step for one run,
17 µs/run-step for batches ≥ 16.

## Limits of the 2-D section

* It captures the side-clearance leak paths, confinement in the slot, the
  chamber's air spring, and the scaling of all of these with `rho0`.
* It misses flow around the tip, the opening that varies along the tongue, 3D
  jet spreading and relief, 3D radiation, and the real chamber geometry.
* In the tested range (dp 0.5–4 kPa, set 0.1–0.6 mm, kicks up to 2.2 mm) the
  blown 2-D reed does **not** self-oscillate: the air flow damps it. So pitch
  is measured on the ringing reed at matched amplitude. That gives the air
  loading at a given amplitude, but not the amplitude a real reed settles at.
