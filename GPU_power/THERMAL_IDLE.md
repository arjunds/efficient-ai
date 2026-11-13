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
- **B200 card A** (PARCC dgx-b200, node dgx016, PCI 43:00.0): `microbench/thermal_B200/`,
  job 10342748, 31 min.
- **B200 card B** (node dgx007, PCI DF:00.0): `microbench/thermal_B200_pairs/`, job 10516414,
  17 min. Dense ramp (12 levels) with **fixed 20 s holds** so the die never equilibrates, to
  make the up/down pair check informative (§7).
- **A100 card 3** (kai, PCI 61:00.0) and **A6000 card 2** (kai, PCI 81:00.0):
  `microbench/thermal_{A100,A6000}_run2/`, 2026-11-12. Same 20 s fixed holds, with finer low-load
  levels (`--ramp_fracs`) so the A6000 gets 8 uncapped levels instead of 2 (§8).

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

**Superseded 2026-11-13 by the dedicated pair run (§7).** With 11 pairs instead of 4 and
ΔT of 8–9 °C instead of 2–4, the in-load k is *not* a single number: it rises with
bandwidth, and only its zero-bandwidth limit matches the idle cool-down. These four pairs
all sit in the small-ΔT regime where a ±1 W error moves the ratio by ±0.5 W/°C, so their
tight agreement with 1.03 was not as informative as it looked.

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

**7. The in-load k is not constant: it rises with bandwidth, and only its zero-bandwidth
limit matches the idle cool-down (B200 card B, 2026-11-13).**

Built to settle §5's in-load check. Twelve ramp levels instead of seven, and **fixed 20 s
holds** (`--level_min_s 20 --level_max_s 20`) so the die never reaches steady state. The
lag is the point: the down leg then runs much hotter than the up leg at the same
occupancy. It worked — ΔT at matched bandwidth is **8–9 °C at low occupancy**, against
2–4 °C with steady-state holds. (20 s, not shorter: each level's mean is taken from the
last 10 s, so a shorter hold would fold in the load transient. A moving temperature also
makes the 10 s average fractional rather than snapping to NVML's whole degrees.)

Bandwidth matches across each pair to within 0.2%, so the difference isolates temperature.

| programs | DRAM GB/s | ΔT | ΔP | W/°C |
|---|---|---|---|---|
| 18 | 265 | 9.0 | 11.1 | **1.23** |
| 37 | 542 | 9.0 | 11.8 | **1.31** |
| 56 | 826 | 8.7 | 13.5 | 1.55 |
| 74 | 1077 | 8.2 | 14.0 | 1.70 |
| 111 | 1617 | 7.9 | 14.0 | 1.76 |
| 148 | 2131 | 6.0 | 11.8 | 1.96 |
| 185 | 2635 | 4.0 | 6.5 | 1.64 |
| 222–444 | 3140–5652 | 1.3–2.0 | 3.4–8.1 | 1.70–6.27 (noise) |

**The tail is noise, the rise is not.** ΔT shrinks as occupancy grows, and a ±1 W error in
ΔP moves the ratio by ±0.11 W/°C at ΔT = 9 but ±0.77 W/°C at ΔT = 1.3 — so the last four
pairs carry no information. But over the seven well-resolved pairs, where the noise floor
is ±0.11–0.17, k still climbs from 1.23 to ~1.9. Regressing those on bandwidth:

> **k(BW) ≈ 1.30 + 0.22 W/°C per TB/s** (R² 0.57, n = 7)

- **At zero bandwidth this extrapolates to 1.30 W/°C, against this card's own idle
  cool-down of 1.21–1.26.** That is the real confirmation that leakage per degree is the
  same under load as at idle — but it holds *in the limit*, not across the board.
- **At 4 TB/s the apparent k is about 2.2 W/°C, nearly double the idle value.**

Two mechanisms predict this sign and **this run cannot separate them**: (a) genuinely more
leakage, because more of the die is powered and switching, so more devices leak per degree;
(b) a sensor artifact, because NVML reports one temperature while the true mean die and
HBM temperature run further above it the harder the card is driven, so ΔT is understated
and the ratio inflated. Distinguishing them needs per-sensor or hot-spot telemetry.

**A second B200, and the card-to-card picture differs from the A100's.** This run landed on
a different card from §5 (dgx007 PCI DF:00.0 vs dgx016 PCI 43:00.0), so the two can be
compared at a matched temperature using each card's own fit:

| | idle at 42 °C | cool-down k |
|---|---|---|
| card A (dgx016) | 251.5 W | 1.00–1.08 |
| card B (dgx007) | 246.8 W | 1.21–1.26 |

**4.8 W apart, 1.9%** — nothing like the A100 pair's 26 W (35%) in §3. But their *slopes*
differ by **18%**, so what varies between cards is not only the offset. Two cards each, on
both models, so this is a hint and not a distribution.

**8. The rising in-load k replicates on the A100 and A6000, more cleanly than on the B200
(2026-11-12 runs; `microbench/fig_inload_k.png`, `thermal_pairs_fig.py`).**

Same pair method as §7: 20 s fixed holds, pairs with ΔT ≥ 4 °C and no capping.

| card | pairs | k(BW) at matched bandwidth | R² | idle cool-down k | relative ∂e_b/∂T |
|---|---|---|---|---|---|
| A100 61:00.0 | 7 (0.11–1.35 TB/s) | **0.52 + 0.78·BW** | 0.97 | 0.77 | 0.6%/°C |
| A6000 81:00.0 | 7 (0.07–0.54 TB/s) | **0.30 + 2.76·BW** | 0.88 | 0.25 | 1.3%/°C (fan-confounded) |
| B200 DF:00.0 (§7) | 7 (0.27–2.64 TB/s) | **1.30 + 0.22·BW** | 0.57 | 1.26 | 0.2%/°C |

(BW in TB/s; relative ∂e_b/∂T = the BW coefficient divided by each card's fixed-k e_b.)

- **The rise is real, not a B200 quirk.** It's monotone on all three cards, and on the A100
  it's the cleanest (R² 0.97). At 1.35 TB/s the A100's apparent k is about 1.6 W/°C, twice
  its idle value.
- **A simpler reading than §7's two mechanisms:** dP/dT at fixed bandwidth = k_idle +
  BW·∂e_b/∂T. So the slope in BW is the temperature sensitivity of the energy per byte
  itself. On the A100, e_b rises about 0.6% per °C. Over a 25 °C serving swing that's about
  15% of e_b, too large to ignore in the dynamic term. This is a restatement, not a
  separation: "more of the chip leaking when busy" and "memory or I/O energy per byte rising
  with temperature" both look like this. The sensor-lag explanation from §7 is still open.
- **The zero-bandwidth limit matches idle on two of three cards.** It does on the B200
  (1.30 vs 1.26) and the A6000 (0.30 vs 0.25). It doesn't on the A100 (0.52 vs 0.77), which
  isn't explained yet.
- **Short holds also break the ramp collinearity.** corr(BW, T) falls from 0.92–0.99 with
  steady holds to 0.73 with 20 s holds. The free ramp k is then 1.02 (A100) and 1.11
  (A6000), between the idle and loaded pair values, as expected for a load-dependent k. With
  k held at the idle cool-down slope, e_b drops by 6.3% (A100 card 3) and 3.0% (A6000
  card 2). That agrees with §5's −6.9% and −4.2%.

**Card-to-card, now 3 A100s and 2 of each other card:**

| card | idle, cold, clocks as noted | idle cool-down k |
|---|---|---|
| A100 C1:00.0 | 73.6 W @ 43 °C, 1410 MHz | 0.75–0.78 |
| A100 41:00.0 | 99.6 W (`idle_clk`), 1410 MHz | — |
| A100 61:00.0 | 98.6 W @ 38 °C (about 102 W at 43 °C), 1410 MHz | 0.77 |
| A6000 61:00.0 | 31.0 W @ 32 °C, 210 MHz | 0.37–0.43 (R² 0.86–0.96) |
| A6000 81:00.0 | 27.9 W @ 32 °C, 210 MHz | 0.25 (R² 0.98) |
| B200 dgx016 43:00.0 | 251.5 W @ 42 °C | 1.00–1.08 |
| B200 dgx007 DF:00.0 | 246.8 W @ 42 °C | 1.21–1.26 |

- **A100: C1:00.0 is the outlier.** Two of three cards idle at about 100 W. The slope is the
  same on both cards measured (0.77). So on the A100 what varies is the offset.
- **B200: the reverse.** The offsets are within 2%, but the slopes differ by 18%.
- **A6000:** both differ (10% offset, about 1.6x slope). The first card's fit is noisier,
  and the fan is in board power.

**Next measurement:** log NVML's memory (HBM) temperature alongside the GPU temperature, on
the A100 and B200 where it's exposed. If the HBM–die temperature gap at matched bandwidth
grows with load, the sensor-lag reading (§7 b) gains weight. If it doesn't, e_b(T) is the
better description.

**9. How repeatable is "an A100 at T °C idles at X W", and does history matter? (2026-11-13)**

PI questions after the idle-vs-temperature figure:
- How accurate is a statement like "an A100 at 40 °C idles at X W", and does it vary between runs?
- What does the picture look like over time?

**Runs:** five kai jobs `a100-hist1..5` (`microbench/thermal_A100_hist*/`), about 12 min each.
- **Phases:** warm-up baseline, heavy stream to a plateau, then cool; light stream (1/8 occupancy) to a plateau, then cool.
- **New in the script:** memory (HBM) temperature every 100 ms (`mem_temp.csv`), plus PCI bus id, UUID and serial.
- **Cards:** four distinct cards. **PCI bus ids repeat across nodes:** hist3 and hist4 were both "E1:00.0" but on node8 and node3. Cards are therefore identified by node + bus from `kai pods`.
- **One crash:** hist1's first attempt errored on node3 and was rescheduled to node8 01:00.0, the same card as hist2.

Figures: `talk_figs/fig_idle_cards.png`, `fig_idle_history.png`, `fig_thermal_time.png` (`fig_thermal_detail.py`).
Cool-down samples start 10 s after the load stops.

| question | answer |
|---|---|
| Spread within a run at a fixed die temperature | 10–90% of 1 s means is **0.5 W** wide |
| Same card, two runs (node8 01:00.0), per degree | median **0.2 W** apart, max 1.2 W |
| Across cards at 50 °C, 6 A100s | **78–110 W**. Five cards sit within 104–111 W (±3%); node3 C1:00.0 is about 30 W lower. |
| Thermal history at a matched die temperature: warming at idle, after heavy load, after light load | within about **0.5 W** on 3 of 4 cards. On node3 81:00.0 the light-load cool-down runs 1–2 W low at 37–38 °C. |

- **So "this card at T idles at X W" is good to about ±1 W. "An A100 at T idles at X W" is
  good to about ±3%, with one card in six 30 W off.** Card-to-card variation dominates every
  other source.
- **The relationship is convex, not linear.** On every card the slope rises with temperature:
  0.6–0.9 W/°C at 38–46 °C, 0.7–1.0 at 45–53, 0.8–1.3 at 52–62. That's the expected shape of
  leakage. It also explains the earlier "history effect": the low warm-up slope (about
  0.5 W/°C at 31–43 °C) is this curvature, not path dependence. Fit P_s(T) locally or with a
  quadratic; a single k is an average over the range fitted (0.77 is the 45–68 °C average on
  the first card).
- **Memory temperature.** Under heavy load the HBM runs about 14 °C hotter than the die
  (75 vs 61 °C). Once idle it falls back to its usual offset within about 30 s. At a matched
  die temperature the memory–die gap is the same for all three histories, so at idle one
  sensor is enough. Two node3 cards read memory about 13 °C above the die even at idle,
  against about 2 °C on node8. Their idle power per die temperature is mid-pack, so this
  looks like a sensor offset or local airflow, not extra power.
- **Over time** (`fig_thermal_time.png`):
  - The A100 holds 1410 MHz whenever a context is open, so its idle power just follows
    temperature.
  - Under the 300 W cap the A100 sheds clock (1410 → about 1260 MHz).
  - The A6000's governor drops it to 210 MHz within about 10–20 s of going idle, a step of
    about 75 W unrelated to temperature. That's the one place where "power management
    kicking in" dominates idle power.

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
  check, with short fixed holds so ΔT is large enough to resolve (§7). Then refit e_b with k
  held fixed.

  **And treat P_s(T) with a single k as a low-occupancy approximation.** §7 finds the
  effective k rising about 0.22 W/°C per TB/s, roughly doubling by 4 TB/s. For our serving
  breakdowns that is second-order — the §6 correction uses idle-calibrated k and shifts the
  static slice by about +2 points, and a higher in-load k would only push the same way — but
  a model claiming to track power *across* occupancy should carry k as a function of load,
  or state the approximation.
