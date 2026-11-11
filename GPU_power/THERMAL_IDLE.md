# Idle power is a function of die temperature (and of the card): thermal/leakage runs (2026-11-09)

**Question (PI, 2026-11-09):** idle power is not a fixed number. It is a by-product of the
preceding dynamic power, and it feeds every breakdown and the power roof (P_s, P_0).

**Method:** `microbench/thermal_leakage.py`. No clock or power-limit change; one CUDA context
open throughout; NVML at 10 ms (POWER_INSTANT, averaged power, energy counter, clocks,
temperature).
- **Baseline:** idle until the temperature is steady.
- **Heat soak, then cool-down:** a DRAM stream to a temperature plateau, then idle for up to
  10 min. Repeated with an fp16 GEMM as the heater.
- **Occupancy ramp:** DRAM-stream occupancy up and back down, each level held until the
  temperature is steady.

Fits use 1 s bins, one clock state, and exclude the first 2/5/10 s after the load stops,
because NVML's averaging tail settles in about 1.3 s while the die takes minutes to cool.

**Runs:**
- **A100 80GB PCIe** (kai, PCI C1:00.0) and **RTX A6000** (kai): `microbench/thermal_{A100,A6000}/`, with `thermal.png`.
- **A5000 smoke test:** the debug A5000 is capped at 100 W, so every ramp level throttles to 210 MHz. Unusable for the ramp.
- **B200** (PARCC dgx-b200, node dgx016, PCI 43:00.0): `microbench/thermal_B200/`, job 10342748, 31 min.

## Results
**1. Idle power rises linearly with die temperature, and only temperature matters (A100).**

| cool-down after | W per °C (excluding 2 / 5 / 10 s) | R² | idle extrapolated to the 43 °C baseline | measured baseline |
|---|---|---|---|---|
| DRAM stream | 0.75 / 0.74 / 0.71 | 0.98 | 73.0 W | 73.6 W |
| fp16 GEMM | 0.78 / 0.77 / 0.75 | 0.98 | 72.9 W | 73.6 W |

- **The heater doesn't matter.** Memory-bound and compute-bound heating give the same line,
  so idle power depends on how hot the chip is, not on what made it hot.
- **The averaging tail is excluded.** The slope barely moves as more of the start is excluded,
  so this isn't NVML's power averaging.
- **One line covers both states.** It passes through the cold baseline, so P_idle(T) is a
  single function.
- **Size:** over a normal operating swing (43 → 72 °C) that's **+22 W, about +30% of idle**.
  This matches the retrospective binning of idle samples in the trace replays (97 → 119 W
  from 35–45 °C to 60–65 °C).

**2. Leakage biases the dynamic coefficients when load and temperature rise together (A100
ramp, uncapped levels).**

| fit | intercept above idle | J per DRAM byte | leakage term |
|---|---|---|---|
| P = a + e·BW | 12.5 W | 1.29e-10 | — |
| P = a + e·BW + k·(T − 43 °C) | 6.5 W | **1.21e-10 (−6%)** | **0.64 W/°C** |

- **Independent agreement.** The ramp's leakage coefficient (0.64 W/°C) agrees with the
  cool-down slope (0.71–0.78) without being fitted to it.
  **Revised 2026-11-09 after the B200 run (§5):** treat this agreement as a coincidence,
  not as confirmation. In a ramp, bandwidth and temperature rise together — on this A100
  they correlate at **r = 0.986** — so the temperature coefficient is barely identified
  separately from the per-byte term. On B200 the same method gives a ramp k of 2.59 W/°C
  against a cool-down slope of 1.03, a 2.5x disagreement. The cool-down fit is the
  trustworthy estimate on all three cards, because there bandwidth is zero and temperature
  varies alone.
- **About half of the "activity" intercept above idle is temperature.** The per-byte energy
  is about 6% too high when temperature isn't modelled. (Revised 2026-11-11, §5: with k fixed
  at the cool-down slope instead of fitted, the A100 numbers barely move: intercept
  12.5 → 5.6 W, e_b −6.9%. The intercept drop is because this ramp started hot, 53 °C at the
  first level against the 43 °C baseline: 10 °C × 0.75 ≈ 7.5 W of leakage.)
- **The up/down hysteresis is uninformative.** Each level was held until the temperature was
  steady, so up and down landed within 1–2 °C of each other. The regression is the
  cross-check.

**3. Idle power differs between two cards of the same model in the same state.**
- **This run's A100** (PCI C1:00.0), CUDA context open, 1410 MHz, 43 °C: **73.6 W**. The
  instantaneous reading, the averaged reading and the energy counter all agree.
- **Our earlier A100** (PCI 41:00.0), same state (`idle_clk` in `gpus_measured.json`):
  **99.6 W**. The trace replays, also on 41:00.0, show 97 W at 35–45 °C.
- **Temperature can't explain it.** A 26 W gap at about 0.75 W/°C would need a 35 °C
  difference, so this looks like card-to-card leakage variation. That's only two cards so
  far.

**4. A6000:** the GPU drops to 210 MHz (31 W) within seconds of going idle, even with a
context open. So "idle" there is first a clock-state question:
- deep idle: 31 W;
- clocks-up idle: 107 W (`idle_clk`).

The deep-idle cool-down slope is 0.37–0.43 W/°C (R² 0.86–0.96). The fan is part of board
power and its speed tracks temperature, so leakage and fan power can't be separated here. Only
2 ramp levels per direction stay under the 300 W limit, so the A6000 temperature-corrected fit
is not reliable.

**5. B200: the effect reproduces, is bigger in watts and smaller in proportion, and it
breaks the ramp cross-check.**

Same script, no clock or power-limit change, one B200 (1000 W limit, 148 SMs). Baseline
idle with a context open at 1965 MHz and 32 °C: **240.2 W**.

| cool-down after | W per °C (excluding 2 / 5 / 10 s) | R² | idle extrapolated to the 32 °C baseline | measured baseline |
|---|---|---|---|---|
| DRAM stream | 1.08 / 1.06 / 1.05 | 0.99 | 241.0 W | 240.2 W |
| fp16 GEMM | 1.02 / 1.00 / 0.99 | 0.98 | 241.2 W | 240.2 W |

- **The A100's three qualitative claims all hold.** The heater doesn't matter (stream and
  GEMM agree to ~5%, and the GEMM heater ran at 1324 TF/s against the stream's 7326 GB/s);
  the slope barely moves as more of the start is excluded, so it isn't NVML's averaging
  tail; and both lines pass through the independently measured cold baseline, here to
  within **1.0 W (0.4%)**.
- **k grows with the card, but strongly sublinearly.** 1.03 W/°C against the A100's 0.75,
  for 3.3x the power limit (1000 vs 300 W) and a much larger die — a **1.4x** slope. As a
  fraction of its own idle the B200 is the *less* temperature-sensitive card:
  **0.43%/°C** vs the A100's 1.02%/°C.
- **The B200 runs cold, so the operating swing is smaller.** It peaked at **57 °C** under a
  full 1000 W GEMM, and returned to 31 °C within the 120 s cool-down floor every time. Our
  53 B200 trace-replay levels sit at 36–61 °C (median 47). Over the serving swing
  (32 → 61 °C) idle moves **+30 W, about +12% of idle**, against the A100's +30%.
- **It hits the power cap on a pure DRAM stream.** At ≥592 concurrent programs the card
  reached 1000 W and downclocked (1965 → 1530–1732 MHz), so those 4 of 14 ramp levels are
  excluded from the fit, as on the A100.

**The ramp cross-check does not survive.** The B200 ramp fit gives a leakage term of
**2.59 W/°C**, 2.5x its own cool-down slope:

| fit | intercept above idle | J per DRAM byte | leakage term |
|---|---|---|---|
| P = a + e·BW | 32.9 W | 1.308e-10 | — |
| P = a + e·BW + k·(T − 32 °C), k fitted | 32.9 W | 1.218e-10 (−6.9%) | 2.59 W/°C |
| P = a + e·BW + k·(T − 32 °C), k fixed at the cool-down 1.03 | 32.9 W | **1.272e-10 (−2.8%)** | 1.03 W/°C |

(Absolute intercept 273.0 W; the 240.2 W baseline is subtracted to match §2. Unlike the A100,
the intercept doesn't move with the temperature term, because the B200 ramp started at the
baseline temperature: 32 °C at the first level.)

The cause is collinearity: a ramp raises bandwidth and temperature together, so the two
regressors are nearly the same variable. Across the three cards the correlation between
bandwidth and temperature over the fitted levels is **0.956 (B200), 0.986 (A100), 0.924
(A6000)** — the A100, whose ramp and cool-down happened to agree, is the *most* collinear
of the three. So that agreement was luck. **Use the cool-down slope**, where bandwidth is
zero and temperature varies on its own; §2's ramp k should not be quoted as an independent
confirmation.

**The per-byte correction is card-dependent, not a uniform 6–7%** (revised 2026-11-11). The
free-k fits give −5.9% (A100), −6.8% (A6000), −6.9% (B200), but those carry the same
collinearity: on the B200 the inflated k (2.59) is what makes the correction look as large as
the A100's. Holding k at each card's cool-down slope:

| card | k fixed at | e_b correction | °C per TB/s during the ramp | predicted k·dT/dBW ÷ e_b |
|---|---|---|---|---|
| A100 | 0.75 (0.71–0.78) | **−6.9%** (−6.6 to −7.2) | 13.5 | 7.9% |
| A6000 | 0.40 (0.37–0.43), fan-confounded, n=4 | **−4.2%** (−3.9 to −4.5) | 22 | 4.2% |
| B200 | 1.03 (0.99–1.08) | **−2.8%** (−2.6 to −2.9) | 3.1 | 2.5% |

The size follows from how fast each card heats per unit of bandwidth. The A100 heats
13.5 °C per TB/s, so a byte carries almost 8% of extra leakage. The B200 heats only 3 °C per
TB/s (large heatsink, cold intake), so leakage adds under 3%.

**Up/down pairs at the same bandwidth give an in-load k on the B200, and it matches the
cool-down.** On the A100 the up and down levels landed within 1–2 °C, so the pairs are
uninformative there. The B200 heats fast enough that the down leg runs 2–4 °C hotter than
the up leg at the same occupancy. Bandwidth matches to within 0.4%, so the pair difference
isolates temperature with no collinearity:

| programs | DRAM GB/s | ΔT down − up | ΔP (bandwidth-corrected) | W/°C |
|---|---|---|---|---|
| 37 | 548 | +2 °C | +1.9 W | 0.97 |
| 74 | 1084 | +2 °C | +2.2 W | 1.09 |
| 148 | 2147 | +4 °C | +4.2 W | 1.05 |
| 296 | 4078 | +4 °C | +10.5 W | 2.6 (outlier, see below) |

Three of the four pairs agree with the idle cool-down slope (1.03), so on this card leakage
per °C is the same under load as at idle. Each pair is coarse, because NVML reports
temperature in whole degrees. At 296 programs (about 800 W) the ramp fit's residual is about
8 W, which is the same size as the pair difference.

**6. Quantified on our own B200 breakdowns.** Applying `P_s(T) = 240.2 + 1.03·(T − 32)`
with each trace level's own median die temperature, against the cold idle those runs
actually used (234–244 W):

| level | median T | idle used | idle at T | static share | corrected |
|---|---|---|---|---|---|
| 7B reasoning x0.5 | 57 °C | 241.5 W | 265.9 W | 27.6% | 30.3% |
| MoE reasoning x0.5 | 61 °C | 243.7 W | 270.1 W | 25.0% | 27.7% |
| MoE azure23_conv x12 | 51 °C | 243.7 W | 259.8 W | 27.2% | 28.9% |
| 7B azure23_code x2 | 36 °C | 240.0 W | 244.3 W | 66.3% | 67.5% |

Across all 53 levels the static slice is understated by **+0.6 to +5.9 points, median
+2.1**, and the dynamic slices are overstated by the same amount. The error is largest
exactly where it matters most — the hottest, highest-load levels. This does not change any
J/token number, only the attribution between static and dynamic.

## What this changes
- **Total J/token predictions: little change.** The dynamic coefficients were measured on hot,
  loaded chips, so they already absorb operating-temperature leakage.
- **The breakdown's static slice is understated.**
  - It uses cool, clocks-up idle. At serving temperature the A100's static draw is about
    15–22 W higher, roughly 5–7 points of a 300 W budget.
  - The dynamic slices are overstated by the same amount. Part of P_0, and 3–7% of the
    per-byte energy (A100 about 7%, B200 about 3%), is leakage.
- **The power roof moves with temperature and with the card.** The A100's dynamic headroom is
  about 300 − P_s − P_0 ≈ 175–200 W, depending on which P_s.
  - The temperature swing (about 22 W) moves the roof by about 11–12%.
  - The card-to-card spread (about 26 W) moves it by about 13–15%.
  - The roof isn't a property of the SKU alone.
- **Model fix:**
  - Replace the constant P_s with P_s(T) = P_s0 + k·(T − T0), where k ≈ 0.7 W/°C on the A100.
  - Measure P_s0 per card.
  - Refit e_b and P_0 with the temperature term.

  The B200 run answers this: **k does scale with die size and power, but sublinearly** —
  1.03 W/°C at 1000 W against 0.75 W/°C at 300 W. Per card:

  | card | idle (cold, clocks up) | k (cool-down) | k as % of idle | swing over its own operating range |
  |---|---|---|---|---|
  | A100 80GB PCIe | 73.6 W @ 43 °C | 0.75 W/°C | 1.02%/°C | +22 W (43→72 °C), +30% |
  | RTX A6000 | 107 W clocks-up / 31 W deep | 0.37–0.43 W/°C (fan-confounded) | — | not reliable |
  | B200 | 240.2 W @ 32 °C | 1.03 W/°C | 0.43%/°C | +30 W (32→61 °C), +12% |

  So a single k is not portable, but neither is it wild: within 1.4x across a 3.3x range of
  power limits. Measure it per card; the cool-down is a 4-minute measurement.

  One practical caveat for the refit: **fit k from cool-downs, not from a load ramp.** The
  ramp makes bandwidth and temperature collinear (r = 0.92–0.99 on all three cards) and the
  B200 shows it can inflate k by 2.5x. Use up/down pairs at matched bandwidth as the in-load
  check (§5). Then refit e_b with k held fixed.
