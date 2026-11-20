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

  **Revised 2026-11-19 after the third B200 (§10):** the 18% is a range artifact. Those two
  fits cover different temperature ranges on a curve that is convex (§9), so they are not
  comparable. On the common 37–43 °C window the three B200 slopes agree within 3%.

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

  **Separated 2026-11-20 by the pulse runs (§11, §12):** it is e_b(T). With the activity paused for
  1 s the static slope sits at the idle value at every bandwidth, and the gap to the loaded
  slope is proportional to bandwidth with a zero intercept. "More of the chip leaking when
  busy" is excluded as a mechanism.
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
- **B200: offsets and slopes are both tight** — within 2.5% and 3% across three cards, on a
  common temperature window (§10). The 18% slope difference first reported here was a range
  artifact; the k values in this table are each card's own-range fit and are not comparable
  between rows.
- **A6000:** both differ (10% offset, about 1.6x slope). The first card's fit is noisier,
  and the fan is in board power.

**Next measurement:** log NVML's memory (HBM) temperature alongside the GPU temperature, on
the A100 and B200 where it's exposed. If the HBM–die temperature gap at matched bandwidth
grows with load, the sensor-lag reading (§7 b) gains weight. If it doesn't, e_b(T) is the
better description.

**Answered in §9 (A100) and §10 (B200): the gap does grow with load** — about 3 °C per TB/s on
the B200, reaching +19 °C at 7.4 TB/s, with the memory heating 2.11x as fast as the die. That
is enough to explain the rise if about 40% of the leakage sits in the HBM, but it predicts a
step rather than the linear rise in BW that §8 measured, so it is not yet decided. §10 names
the run that would settle it.

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

**10. B200, third card: memory temperature answers §8's next measurement, and the "18% slope
difference" was a range artifact (2026-11-19)**

The B200 counterpart of §9, same protocol (`--phases baseline,heat:stream,heat:light`), on a
**third** B200: `microbench/thermal_B200_hist1/`, job 10835525, node dgx025 PCI 43:00.0. The
bus id repeats card A's but the UUID and serial differ, so these are distinct cards — §9's
warning about bus ids repeating holds on SLURM too. Baseline idle **235.4 W @ 33 °C**, 10 min.

**The HBM–die gap grows with bandwidth, which is what §8 asked for.**

| | bandwidth | die | memory (HBM) | gap |
|---|---|---|---|---|
| idle (baseline, and after either load) | 0 | 33.0 °C | 30.0 °C | **−3.0** |
| light stream, 1/8 occupancy | 0.26 TB/s | 34.0 °C | 33.0 °C | −1.0 |
| full stream | 7.38 TB/s | 52.9 °C | 72.0 °C | **+19.2** |

- The gap grows about **3.0 °C per TB/s**. At full bandwidth the memory sits 19 °C above the
  die, against the A100's 14 °C at its own peak (§9) — larger on the card with more bandwidth.
- Along the load direction the memory heats **2.11x** as fast as the die (ΔT_mem 42.0 vs
  ΔT_die 19.9 °C). The single die sensor does understate what the HBM sees, so §7's
  sensor-lag reading gains weight.
- **At idle the two track 1:1** (dT_mem/dT_die = 1.02 over the cool-down), so fitting the
  cool-down against memory temperature instead of die temperature gives the same slope
  (1.04 vs 1.08 W/°C). §9's "at idle one sensor is enough" holds on the B200 too.
- At idle this B200 reads memory **3 °C below** the die, where node8's A100s read about 2 °C
  above and node3's about 13 °C above. One more sensor offset, not extra power.

**How much of the in-load k rise this explains.** If a fraction φ of the leakage sits in the
HBM and we attribute all of it to the die sensor, the apparent k inflates by 1 + φ·(R − 1)
with R = 2.11. Reproducing §8's inflation on card B at 2.64 TB/s (1.88/1.30 = 1.45x) needs
**φ ≈ 0.4**. That is a plausible HBM share for an 8-stack HBM3e part, but it is one parameter
fitted to one number, not a measurement of φ.

It also does not close the question, because it predicts a *different shape* than §8 measured:
if the gap is proportional to bandwidth then R is bandwidth-independent, so the inflation
should appear as a step at the first non-trivial load and then flatten — not rise linearly
from the idle value, which is what §8's fits show. Our 0.26 TB/s point cannot referee it: die
and memory move 1 and 3 °C, at 1 °C sensor resolution.

**The discriminating run is now one command:** the §7 pairs protocol with memory temperature
logged, which the script has done since `14db4f8`. If the apparent k collapses onto a single
value when regressed against *memory* temperature, it is the sensor. If it still rises with
bandwidth against memory temperature, e_b(T) is real.

**Superseded 2026-11-20 by §11 and §12.** The pulse protocol answered it better and the φ ≈ 0.4 estimate
above is moot: measured directly with the activity off, the static k does not rise with
bandwidth at all, so there is no inflation for a sensor offset to explain. The HBM gap
replicates (2.8 °C per TB/s on a fourth card) but belongs to the per-byte term, not the static
one.

**§9's convexity replicates on the B200 — and it invalidates §8's slope comparison.** Local
slope from `cool:stream`, 1 s bins past 2 s, at the modal clock:

| card | 33–37 °C | 37–41 °C | 41–45 °C | 45–50 °C |
|---|---|---|---|---|
| A dgx016 43:00.0 | 1.04 | 1.13 | 1.40 | — |
| B dgx007 DF:00.0 | — | 1.12 | 1.35 | 1.32 |
| C dgx025 43:00.0 | 0.98 | 1.16 | 1.29 | — |

Three cards, same shape as the A100: the slope rises about 30% from 33–37 to 41–45 °C, and the
three agree band by band to within 6%.

So **§8's "the slopes differ by 18%" was an artifact of fitting different temperature ranges
on a convex curve.** Card B's cool-down started at 50 °C and card A's at 43 °C. Fit over each
card's own range and you get 1.29 (B) vs 1.10 (A); fit over the common 37–43 °C window and the
three cards agree:

| card | idle at 40 °C | k over 37–43 °C |
|---|---|---|
| A dgx016 43:00.0 | 249.8 W | 1.16 |
| B dgx007 DF:00.0 | 244.1 W | 1.13 |
| C dgx025 43:00.0 | 243.8 W | 1.16 |

**Spread 6.0 W (2.5%) at a matched 40 °C, slopes within 3%**, with B and C within 0.3 W. On the
B200 both the offset and the slope are tight across three cards — unlike the A100, where one
card in six is 30 W off (§9). Comparing cool-down slopes is only meaningful on a common
temperature window; the per-card ranges in §5/§7/§8 are not interchangeable.

**One caveat on this run:** the light-load heater barely heats a B200. At 1/8 occupancy it
draws 0.26 TB/s and 294 W, lifting the die 1 °C, so the light-vs-heavy thermal-history
comparison that §9 could make on the A100 has no temperature range here. This run says
nothing about history on the B200; that would need a heavier light load (1/2 occupancy).

**11. Static power at load temperature, read in 1 s pauses inside the load (2026-11-20)**

The professors' objection (static and dynamic power cannot be separated under load) left §1–§10
measuring idle, not load. This run reads the zero-activity power *at load temperature*: stop the
load for 1 s every 15 s. Activity power falls within ~0.1 s (the transition traces of §9), the
die needs seconds to cool, and NVML samples every 10 ms. `thermal_leakage.py` pulse phases
(`768a2d5`): `--phases baseline,pulse:stream,pulse:gemm,pulse_ramp`, 4 min per op, then a
7-level stream ramp up and down at 45 s per level. Per pause the analysis keeps the **trough**
(lowest 50 ms running median: clock and voltage not yet back up) and the **end** (last 0.2 s:
clock recovered).

Runs: `microbench/thermal_A100_pulse1/` (kai, A100 80GB PCIe, PCI 81:00.0, UUID GPU-b30129ef,
baseline 99.7 W @ 39 °C, 58 pauses) and `microbench/thermal_A6000_pulse1/` (PCI A1:00.0, UUID
GPU-59207268, 55 pauses). Figure: `talk_figs/fig_pause_static.png`.

| A100 pauses | n | loaded clock | end − trough | end − own cool-down idle at the same T |
|---|---|---|---|---|
| load below the cap | 16 | 1410 MHz | **0.1 W** | **−0.15 W** |
| capped stream | 27 | ~1290 MHz | **4.4 W** (IQR 3.4–6.8) | −0.75 W |
| capped GEMM | 15 | ~1035 MHz | **16.6 W** (IQR 14.3–18.3) | +0.39 W |

- **Below the cap, static power at load temperature is the idle curve.** Trough and end agree to
  0.1 W, and both land on the card's own cool-down idle at the same die temperature (−0.15 W). So
  for uncapped serving, P_s(T) from cool-downs is not an idle-only quantity: it is what the card
  draws with zero activity at its loaded temperature. This card reads 117.5 W at 60 °C against
  99.7 W cold, the same +18 W as §6's correction.
- **Under the cap, static power is lower than the idle curve, because voltage is lower.** The
  trough deficit grows with how far the load pulled the clock down: 0.1 W at 1410 MHz, 4.4 W at
  ~1290, 16.6 W at ~1035. This is a lower bound on the loaded-voltage effect: NVML already reports
  1335–1410 MHz at the trough, so the voltage has partly recovered.
- **What remains a convention:** the always-on clock power is inside both readings; NVML's power
  reading averages over ~25–100 ms; and any static component that disappears within ~0.1 s of a
  stop (e.g. the hottest spots cooling) is invisible. The die reads 2–5 °C cooler at the end of
  the pause than under load.
- **A6000:** in a 1 s pause it stays at ~1900 MHz, so the pauses give the clocks-up idle curve
  that the 210 MHz cool-downs of §4 could not: **94.6 W at 40 °C to 108–109 W at 66 °C**, slope
  ~0.8–1.0 W/°C above 55 °C, flatter below (fan speed is in board power on this blower card). The
  model's P_s of 106.7 W is this card's clocks-up idle at about 62 °C. Capped GEMM pauses show a
  smaller trough (median 0.7 W, IQR 0.2–4.7) than the A100's.
- **Nothing about the load matters except die temperature and clock** (`talk_figs/fig_pause.png`,
  bottom row; numbers in `fig_pause.json`). With the clock recovered, pause power sits on the idle
  line at the same die temperature (A100: median +0.14 W, 10–90% −0.9..+0.8 W, n = 58), and the
  residual does not depend on:
  - **the bandwidth the card was pulling just before:** A100 −0.04 W per TB/s (r = −0.04, 0–1.84
    TB/s); A6000 +0.6 W per TB/s (r = 0.19, 0–0.71 TB/s);
  - **how hot the HBM was:** on the A100 the HBM read 1–17 °C above the die when the pause
    started, slope −0.01 W/°C (r = −0.09). The HBM takes ~30 s to settle (§9), so during a 1 s
    pause it is still at its loaded temperature.

  So on the A100 a hot HBM adds no measurable static power. That argues against §10's reading
  of the rising in-load k (§7, §8) as HBM leakage, and points to the dynamic terms instead
  (energy per byte rising with temperature). The B200, where §10 proposed it, is the test that
  matters; it has now run (`thermal_B200_pulse`, **§12**) and agrees: no bandwidth dependence in
  the static reading, while the loaded slope has plenty. One A6000 pause is dropped: the first
  GEMM pause ended at 1800 MHz, clock not yet recovered.
- **For the model:** below the cap, P_s(T) from cool-downs is the static term at load
  temperature, within ~1 W. At the cap it is lower, by an amount set by how far the clock drops.
  - **A100 serving is at the cap for 80–93% of busy time** (`logs/A100_prov` power traces,
    SW power-cap flag 93%), but the cap only trims the clock: **median 1365 MHz** (p10 1185,
    p90 1395). Interpolating the pause deficits (0.1 W at 1410, 4.4 W at ~1290 MHz) gives
    **~2 W at the median, ~10 W at the p10 clock**. So §6's +14 W static correction at the
    median serving temperature is overstated by roughly 2 W typically, up to ~10 W in the
    heaviest moments, and its conclusion stands.
  - **B200 serving is at the cap for only 4.5% of busy time,** so §6 stands there as measured.

**12. The same pauses on a B200: the rising in-load k is real, and it is entirely in the
energy per byte (2026-11-20)**

§11 shows that below the power cap the pause reading *is* the idle curve, on the A100 and A6000,
and that the residual depends on neither bandwidth nor HBM temperature. This is the B200 run it
names as the one that matters — the card where §7, §8 and §10 located the problem. It reaches
the same conclusion from the other direction: instead of showing that static power has no
bandwidth dependence, it measures the bandwidth dependence that *is* there and shows it is not
static.

`microbench/thermal_B200_pulse/`, job 10836211, node dgx023 PCI 52:00.0 — a **fourth** B200,
and a hot one: baseline **249.5 W @ 44 °C**, levels running 46–63 °C, peak 66 °C. Protocol:
baseline, heavy stream and cool-down (the same-card idle reference), then `pulse_ramp` over six
bandwidth levels up and back down with a 1 s pause after every 10 s of work — **48 pauses**,
8 per level, 4 on each leg.

The pause does read zero-activity power at the loaded die temperature: power crosses halfway
**72 ms** (median; 31–140 ms) into the pause, and the trough, the 0.05–0.30 s window and the
last 0.2 s agree to a median of 2.4 W. On the 40 uncapped pauses the clock is still at 1965 MHz
through the whole pause, so the reading is at the loaded clock and voltage.

**Both of these are true at once, on one card, from the same 48 pauses:**

| | measured |
|---|---|
| total loaded dP/dT at matched bandwidth (§7 pairs method) | **1.83 + 0.26·BW** |
| static dP/dT, same pauses, activity off | **1.71 mean, no comparable rise** |
| this card's idle cool-down k | **1.64** (46–55 °C) |

Per level (BW in TB/s; k in W/°C):

| BW | T band | k_total (pairs) | k_static (pause) | difference |
|---|---|---|---|---|
| 0.27 | 46–48 °C | 2.21 | 1.40 ± 0.37 | +0.82 |
| 0.55 | 47–50 °C | 1.59 | 1.80 ± 0.32 | −0.22 |
| 1.08 | 49–53 °C | 2.08 | 1.50 ± 0.17 | +0.58 |
| 2.14 | 52–57 °C | 2.55 | 1.88 ± 0.08 | +0.67 |
| 4.08 | 55–62 °C | 2.85 | 1.96 ± 0.18 | +0.90 |
| 6.48 | 60–63 °C | *capped* | 1.66 ± 0.44 | — |

- **The rising in-load k replicates.** §7's own method, applied to the loaded power from these
  pauses, gives **1.83 + 0.26·BW** against **1.30 + 0.22·BW** on card B. Fourth card, different
  protocol, nearly the same bandwidth coefficient — the effect is real and reproducible.
- **It is not static leakage.** Turn the activity off for one second and the slope sits at the
  card's idle value at every bandwidth, 1.4–2.0 against an idle cool-down of 1.64, and never
  approaches the 2.6–2.9 the pairs method gives at the same bandwidths. At 6.5 TB/s §7's model
  predicts 2.73 W/°C; the direct measurement is 1.66 ± 0.44, a 2.4σ rejection.
- **The difference is proportional to bandwidth with a zero intercept.**
  Dropping the 0.27 TB/s level, whose 1 °C up/down swing makes its pair estimate meaningless,
  `k_total − k_static = −0.01 + 0.249·BW`. A fixed cost of being busy rising with temperature
  would show up as an intercept; it doesn't. So the term is per byte:
  **∂e_b/∂T ≈ +0.25 pJ/byte/°C**, about **+0.17% of e_b per °C**. Measured independently as
  (loaded − static)/BW at each pause it is +0.23, +0.27 and +0.10 pJ/byte/°C at 1.08, 2.14 and
  4.08 TB/s, which agrees.

**This settles the question §8 left open.** §8 wrote that "more of the chip leaking when busy"
and "memory or I/O energy per byte rising with temperature" both look identical from a ramp,
and called the distinction a restatement rather than a separation. Pausing the load separates
them, and it is the second: **leakage is a function of die temperature alone, bandwidth plays
no part in it, and the per-byte energy carries a small positive temperature coefficient.**

**What it changes in the model.** Keep `P_s(T) = P_s0 + k·(T − T0)` with a single
idle-calibrated k — the §6 correction stands exactly as written, and the §7 warning that a
model tracking power across occupancy "should carry k as a function of load" is now withdrawn.
Put the temperature term on e_b instead, where on this B200 it is small: 0.17% per °C, about
5% of e_b over a 30 °C serving swing, against §8's 0.6%/°C estimate for the A100. That is a
correction to the dynamic coefficient, not to the static slice.

**It also removes the need for §10's HBM-share estimate.** §10 asked what fraction of the
leakage would have to sit in the HBM for sensor misattribution to inflate the apparent static
k, and fitted φ ≈ 0.4. There is no inflated static k to explain. The gap itself is real and
replicates here — **−2.2 °C at idle to +18.3 °C at 7.3 TB/s, 2.8 °C per TB/s** against §10's
3.0 — but it is not what drives the apparent rise. It may still matter for the *per-byte*
coefficient: the bytes move through memory that is 18 °C hotter than the sensor we regress
against, so the +0.25 pJ/byte/°C above is per °C of **die** temperature and would be roughly
half that per °C of actual HBM temperature.

Read directly, as §11 does on the A100 (`talk_figs/fig_pause.png`, bottom middle;
`fig_pause.json`): pause end minus this card's cool-down idle line at the same die temperature
is median **−0.7 W** (10–90% −1.8..+0.5, n = 48), against an HBM−die gap of −1 to +16 °C when
the pause starts. Its slope on the gap is **−0.08 W/°C**; φ = 0.4 would need **+0.66 W/°C**,
i.e. **+12 W** at the top level, where every pause sits within 2.2 W of the line. The small
negative drift (−0.2 W per TB/s) is within what the extrapolation of the idle line past 55 °C
can produce (a linear instead of quadratic extension moves it by ≤0.4 W) and points the wrong way
for any HBM term. Uncapped troughs match the pause end (median −0.6 W); the capped top level
(1695–1717 MHz, 0.87 of idle clock) reads a median **2.7 W** below it, between the A100's
uncapped and capped-stream cases.

**Convexity, fourth card.** This card's cool-down gives k = 1.64 over 46–55 °C, extending the
§10 table upward: about 1.0 at 33–37 °C, 1.13 at 37–41, 1.3–1.4 at 41–45, **1.6 at 46–55**.
The curve keeps steepening, and this is why card D's k must not be compared with cards A–C's
without matching the window.

**Caveats.**
- **All 8 pauses at the top level were power-capped while loaded**, so that level has no usable
  `k_total`: under the cap the GPU sheds clock (1965 → 1695–1717 MHz) instead of drawing more
  power, and the loaded dP/dT collapses toward zero. §11 shows capped pauses also read *below*
  the idle curve, by an amount set by how far the clock fell, so that level's pause reading is
  a lower bound too. It is excluded from every pairs and e_b fit. The other 40 pauses hold
  1965 MHz for the whole pause — §11's "below the cap" case, where the pause reading is the
  idle curve to within about 1 W.
- **The static-k-versus-bandwidth slope cannot be separated from convexity in this run.** Across
  levels corr(T, BW) = 0.945 — the fast levels are the hot ones — and within a level the up/down
  swing is only 1–4 °C. The small positive slope the pause fits show (+0.075 ± 0.058, 1.3σ) is
  what convexity alone would produce. The conclusion above does not rest on that slope: it rests
  on k_static staying near the idle value while k_total runs away from it.
- Longer holds per level would give each pair a bigger ΔT and tighten `∂e_b/∂T`, which is the
  cheapest improvement if this number matters.

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

  **A single idle-calibrated k is right, and the load-dependent k is withdrawn (§12).** §7
  found the effective k rising about 0.22 W/°C per TB/s and advised carrying k as a function of
  load. The pulse run shows that rise is not in the static term: with the activity paused the
  slope stays at the idle value at every bandwidth up to 6.5 TB/s. So keep one k per card,
  calibrated from a cool-down and fitted over the temperature range you care about (the curve
  is convex), and put the load's temperature sensitivity on e_b instead — about +0.17% per °C
  on the B200, +0.6%/°C on the A100 (§8). The §6 correction is unaffected.
