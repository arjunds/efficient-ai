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
- **B200:** `microbench/b200_thermal.sbatch`, pending.

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
- **About half of the "activity" intercept above idle is temperature.** The per-byte energy
  is about 6% too high when temperature isn't modelled.
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

## What this changes
- **Total J/token predictions: little change.** The dynamic coefficients were measured on hot,
  loaded chips, so they already absorb operating-temperature leakage.
- **The breakdown's static slice is understated.**
  - It uses cool, clocks-up idle. At serving temperature the A100's static draw is about
    15–22 W higher, roughly 5–7 points of a 300 W budget.
  - The dynamic slices are overstated by the same amount. Part of P_0, and about 6% of the
    per-byte energy, is leakage.
- **The power roof moves with temperature and with the card.** The A100's dynamic headroom is
  about 300 − P_s − P_0 ≈ 175–200 W, depending on which P_s.
  - The temperature swing (about 22 W) moves the roof by about 11–12%.
  - The card-to-card spread (about 26 W) moves it by about 13–15%.
  - The roof isn't a property of the SKU alone.
- **Model fix:**
  - Replace the constant P_s with P_s(T) = P_s0 + k·(T − T0), where k ≈ 0.7 W/°C on the A100.
  - Measure P_s0 per card.
  - Refit e_b and P_0 with the temperature term.

  The B200 run will show whether k scales with die size and power.
