# Fleet power control under a shared budget — simulation results

Controller-design study (2026-09-24). Question: does economic MPC, using our validated
energy + roofline model as its internal model, beat simpler controllers at allocating a
shared, time-varying power budget across a **heterogeneous** H200 + B200 fleet under
latency SLOs? Everything here is simulation. The plants are `controls/plant_fp.py` (our
first-principles stand-in) and `controls/plant.py` (the calibrated plant another agent
built). Full tables are in `controls/results/tables.md` and raw rows in
`controls/results/*.json`.

## Verdict

1. **Most of the saving comes from SLO-aware power capping, which any feedback controller
   gets.** Our energy model says energy = work-proportional dynamic energy plus
   P_static × time. At realistic load an uncapped GPU runs nearly empty iterations back to
   back, sweeping all the weights each time. Capping the GPU to the lowest power that still
   meets the TPOT SLO makes batches grow (Little's law) and amortizes those weight reads.
   Measured against Uncapped, this saves 48–52% (PI-cap) and 56–60% (MPC).
2. **MPC beats the best feedback baseline (PI-cap) by 12% on the mixed 7B fleet, 17% on
   4×H200 and 22% on 32B.** Violations are equal or lower (the 32B case needs the
   disturbance estimator, point 5). On the calibrated plant the margins are 9%, 14% and 18%.
   The source of the gain is **model-based instantaneous allocation, not prediction**:
   - Consolidating work onto fewer GPUs with MILP on/off decisions is worth about 6%
     (MPC-LP vs MPC).
   - Heterogeneity-aware routing is worth 4.1–7.7% (MPC-blind vs MPC).
   - Horizon N=1 (myopic) vs N=10 changes energy by only 1.5%, inside the CI. A perfect
     forecast saves 0.8%. An announced budget preview saves 0.0%.
3. **The notes predicted MPC wins only when the budget fluctuates and load is
   forecastable. That is not what we see.** The MPC advantage is the same on flat and
   demand-response budgets (−12.3% vs −11.8%). What matters is **load** variability, not
   budget variability:
   - With a flat budget and steady Poisson load, a static setting tuned offline is within
     2.5% of MPC. MPC is overkill in that case, as the notes said.
   - With bursty load, MPC uses 22–38% less energy than the tuned static setting.
4. **Heterogeneity is where the model earns its place, but its decisions hinge on one
   unmeasured number: the B200's minimum settable power limit.** With the assumed 300 W
   floor, the MPC puts 94–97% of 7B tokens on the B200s and leaves the H200s idle. With a
   400 W floor it moves 82% to the H200s; with 500 W, 96%. MPC energy stays flat
   (1073–1100 kJ) across these assumptions while an even split degrades 1223→1682 kJ.
   With parking allowed, idle-power-aware parking keeps one H200 and parks the B200s: 51%
   below MPC without parking. That choice is a slow-loop enumeration, not MPC.
5. **Robustness.** ±20% errors in e_wbyte, e_kvbyte, e_gemm and MBU change MPC energy by
   <2% and violations by <0.5 pp, **but only with online adaptation**. Without the offset-free
   TPOT/energy estimator, the certainty-equivalent MPC violates the TPOT SLO on **25–30%**
   of 32B tokens even with *zero* coefficient mismatch, because its steady-state model does
   not match the fluid plant's dynamics. A static setting tuned on `plant_fp` fails on
   `plant.py` (8–96% queue-SLO violations). Feedback is not optional.
6. **Solve time.** HiGHS MILP, N=10:
   - mean 8 / 18 / 42 ms and max 98 / 170 / 320 ms for 2 / 4 / 8 GPUs;
   - the LP variant runs at 5–12 ms mean;
   - N=30 takes about 125 ms mean and 365 ms max.

   All of these are well under the Δ = 1 s control interval.

## Setup

**Plant (`plant_fp.py`, fluid, 20 sub-steps per second).** For each vLLM iteration:

- Energy: `E_dyn = e_wbyte·W + e_kvbyte·KV + e_gemm·F`, using the reconciled
  coefficients from `gpu_coefficients.json`.
- Time: `t = max(t_roof, E_dyn/(P_cap − P_static))`, where the roofline uses the realized
  MBU(n) from `realized_utilization.csv` (H200 7B 0.494, B200 7B 0.325, H200 32B 0.670,
  B200 32B 0.505, all falling about 15% by n=64).
- State: queue, prefill backlog, and a remaining-output-token histogram. The histogram
  makes batches drain realistically.
- Validation against 9 measured Qwen2-7B runs: power within −4.2% to +2.7%; throughput
  −11% to +9% (the model runs slightly slow at c=4–16).

**Fleet, workload and SLOs.**
- Fleet: 2×H200 + 2×B200 by default. Also 4×H200, 2 GPUs, 8 GPUs, and a 32B variant.
- Workload: alpaca-like (prompt 50, output 153 tokens).
- Load: Poisson at 40 req/s ("steady"), or diurnal (period 600 s, ±50%) × MMPP bursts at
  2.5× ("bursty"). 32B uses 12 req/s.
- SLOs: queue delay (TTFT proxy) ≤ 2 s, measured per request from per-GPU FIFOs;
  TPOT ≤ 100 ms, counted per token.

**Budgets.**
- flat: 47% of ΣTDP (1598 W for the mixed fleet).
- dr: announced demand-response dips to 60%, 75% and 55%. The 60% and 55% dips sit
  *below the sum of the settable floors*, so some GPU must stop serving.
- oversub: an unannounced shared feed minus other tenants' draw (OU process with jumps).

**Settable power floors are assumptions:** H200 200 W (the H100/H200 SXM figure), B200
300 W. Sensitivity results are below.

**Controllers.** All budget-aware baselines get a floor-aware shed-and-project step, so
they can stop routing to GPUs when floors exceed the budget.

| name | what it is |
|---|---|
| Uncapped | TDP caps, JSQ routing (reference) |
| POLCA | reactive two-threshold uniform capping on last interval's measured fleet power |
| PI-adm | budget-proportional caps + PI on `max_running` holding queue delay (setpoint 0.4 s) |
| PI-cap / PI-cap-safe | per-GPU PI on the cap holding TPOT at 80% / 50% of SLO + queue term; JSQ routing |
| StaticOpt | fixed route split + per-type cap fraction; 405-point grid search on a *training* seed |
| MPC | economic MPC, MILP; see below |
| MPC-LP | same, with on/off relaxed (all GPUs always serving) |
| MPC-blind | same, but every GPU is modelled as the fleet-average GPU |
| MPC-persist / -oracle / -holt / -nopreview | forecast and preview ablations |

**MPC formulation (`EconMPC`).** Horizon N=10, Δ=1 s.

- Variables per GPU and step: routed demand A, service D, dynamic power π, waiting
  backlog B, tiered SLO slacks; one on/off binary per GPU; move-suppression and budget
  slack variables.
- Service envelope: our model reduces **exactly** (in steady state) to
  `π ≥ max(floor, a/T + b·D, (a/n_max + b)·D)`, where a = e_wbyte·W is the per-iteration
  weight cost, b is the per-token KV + GEMM + prefill cost, and T is the target iteration
  time. That makes it a MILP.
- Coupling constraints: `Σ_i (P_static,i + π_i) ≤ budget(k)`, and GPUs switched off while
  holding work keep drawing their floor until drained. Little's-law SLO: `B ≤ W·D`.
- Adaptation: online gain θ_i (measured / modelled dynamic energy) and an offset-free
  TPOT correction κ_i.

## Main results (plant_fp, 5 seeds, mean ± 95% CI, 1200 s episodes)

Energy is in kJ. Violations are: queue-SLO (% of requests) / TPOT (% of tokens) /
budget (% of intervals).

| scenario | Uncapped | POLCA | PI-cap | StaticOpt | MPC-LP | **MPC** | MPC-blind |
|---|---|---|---|---|---|---|---|
| flat_steady | 2372 | 1823 | 1202 | 1028 | 1120 | **1002** | 1023 |
| flat_bursty | 2442 | 1823 | 1223 ± 9 | 1373 (1.1% q) | 1145 | **1073 ± 17** | 1125 |
| dr_bursty | 2442 | 1598 (10.3% bud) | 1218 (1.3/0.7/2.1) | 1740 (6.4/0/2.3) | 1146 (6.7% bud) | **1074** (1.2/0.0/1.5) | 1123 |
| dr_steady | 2372 | 1598 (10.3% bud) | 1194 (1.8% bud) | 1048 (7.7% q) | 1117 (4.4% bud) | **1002** (0/0/0) | 1023 |
| oversub_bursty | 2442 | 1631 | 1222 | 1364 | 1146 | **1074** | 1123 |
| flat_bursty 4×H200 | 2019 | 1502 | 972 | 1104 | 880 | **807** | 807 |
| dr_bursty 4×H200 | 2019 | 1304 (10.5% bud) | 967 | 1172 | 880 (4.8% bud) | **806** (0.6/0/0.6) | 806 |
| flat_bursty 32B | 2952 | 2133 | 1534 (13% tpot) | 1516 | 1284 | **1192** (1.9% tpot) | 1226 (12.5% tpot) |
| dr_bursty 32B | 2952 | 1833 (9.4% tpot) | 1527 (21% tpot) | 1830 (9.0% tpot) | 1332 (9.8% tpot) | **1191** (1.0/2.1/0.1) | 1228 (13.6% tpot) |

Notes on the table:
- **PI-cap-safe** trades energy for SLO. It is 2–4% above PI-cap on 7B, and on 32B it
  reaches 0% TPOT violations (flat budget) at **1770 kJ, 48% above MPC**.
- The ~1–2% violations common to every budget-respecting controller in the DR scenarios
  come from the physically infeasible dips below the floors.
- `plots_controls/main_energy_violations.png` plots both panels.
- `plots_controls/dr_trace.png` shows power against budget. POLCA overshoots every dip
  (1-interval lag); MPC typically sits 400–800 W under the budget, because the budget rarely binds
  once caps are SLO-driven.
- `plots_controls/pareto.png` shows that across every setpoint sweep MPC (1073–1106 kJ,
  2.3–4.0% total violations) dominates PI-cap (1214–1365 kJ, 3.4–4.4%).

**Is prediction doing anything?** Measured on dr_bursty:
- N = 1 / 3 / 10 / 30 → 1090 / 1080 / 1074 / 1076 kJ.
- Perfect forecast (oracle) at N=10 → 1065 kJ.
- Persistence 1071, Holt 1076, oracle with 30% noise 1067, no budget preview 1074.

Prediction is worth ≤1.5%, within the CIs (`plots_controls/horizon.png`). The one place it
shows up is 32B TPOT: persistence forecasting gives 3.7–5.9% TPOT violations against
1.9–2.6% with EWMA.

## Heterogeneity (H200 + B200)

`plots_controls/hetero_share.png` and `mpc_allocation.png`; tables are in
`results/tables.md` under hetero / sens / park / park2.

- **7B, all GPUs powered: the MPC routes 94–97% of tokens to the B200s** and leaves the
  H200s idle about 90% of the time. P_static is sunk while a GPU is on. At big batches the
  B200 has *lower* marginal energy per token: e_gemm 0.54 vs 0.68 pJ and e_kvbyte 3.04 vs
  3.38 outweigh its 16% higher e_wbyte. Its assumed floor headroom is also smaller (62 vs
  84 W). The notes guessed "keep B200 for big-batch work", and that is what emerges, but:
- **It flips with the unknown B200 power floor.** At an assumed 400 W the B200 share is
  18%; at 500 W it is 4%. MPC energy is robust (1073 / 1100 / 1099 kJ). PI-cap's
  heterogeneity-blind even split pays for the floor: 1223 / 1443 / 1682 kJ, so MPC is 12%
  / 24% / 35% better. MPC-blind is 4.8% / 4.1% / 7.7% worse than MPC.
- **32B:** the split is balanced (46% B200). A heterogeneity-blind model is dangerous here,
  not just wasteful: MPC-blind violates TPOT on 12–17% of tokens, because the H200 is about
  2× slower per token at 32B.
- **Parking (slow loop, every 30 s; P_park = 0.3·P_static, 30 s wake — both assumed),**
  low-load diurnal:
  - Idle-aware parking keeps a single H200 awake and parks both B200s (238 W vs 116 W
    idle): **463 kJ, against 941 kJ for MPC without parking and 1201 kJ for PI-cap**.
  - A blind parker is only as good as its arbitrary GPU order. B200-first ordering costs
    +18% (545 kJ).
  - This decision is the same under PI-cap or MPC as the fast loop, so it is a
    static / slow-loop result, not MPC.

## Robustness

- **Coefficient mismatch (±10% and ±20% on e_wbyte, e_kvbyte, e_gemm, MBU per GPU; 7B
  dr_bursty):** no controller moves more than 1.5% in energy. The 7B fleet sits at the cap
  floors most of the time, so coefficient errors barely touch its decisions.
- **32B, where the model binds:**
  - MPC with adaptation: 1191 → 1170 kJ at ±20%, TPOT violations 2.1 → 2.5%.
  - **MPC-noadapt: 25–30% TPOT violations at 0% and at 20% mismatch.** The residual
    model/plant structure error needs the κ/θ disturbance estimator.
- **Calibrated plant (`plant.py`, 3 seeds):** MPC stays best. It is −9% vs PI-cap on
  flat_bursty (1132 vs 1242), −8.6% on dr_bursty, −14% on 4×H200, and −18% / −29% vs
  PI-cap / PI-cap-safe on 32B. The online θ converges to ≈1.0, so our energy model is
  consistent with the calibrated plant's energy. StaticOpt, tuned on plant_fp, collapses
  on plant.py (8–96% queue-SLO violations): open-loop settings are brittle to model error.
- **Adapter note for the plant.py owner:** under a power cap, `plant.py` throttles by doing
  fewer iterations but reports the *unthrottled* `t_iter_s`/`tpot_s`. The simulator uses
  `max(t_iter_s, dt/iterations)` whenever `throttled` is set. It also clamps caps only at
  P_static + 1 W (no settable floor).

## Actuator feasibility (A5000, read-only query, job on node-d1)

- `nvidia-smi` on the allocated RTX A5000: min 100 W, max 230 W, default 230 W. The
  **current/enforced limit is 100 W**: this GPU is already capped at its minimum by an
  admin or another tenant. Idle draw is 22 W and persistence mode is on.
- Our uid 74064 is non-root and `sudo -n` requires a password.
- `nvmlDeviceSetPowerManagementLimit` requires root, so **we cannot set power limits (or
  clocks)**. Nothing was changed.
- Only the one allocated GPU was queried. The other three A5000s may differ.
- Consequences:
  1. A live closed-loop demo must actuate **admission**, not power caps.
  2. At a 100 W cap the A5000 is in the capped regime, where our linear model is known to
     break (the A100-PCIe lesson). Power will sit near the cap whenever n ≥ 1, so admission
     acts as duty-cycling. Check `power.limit` in any energy job on this node.

## Proposed live closed-loop experiment (one A5000, about 1 GPU-hour)

1. **Harness (additive):** add `--admission_controller {static,pi,polca,mpc}` to the
   open-loop mode of `energy_profile_load.py`:
   - an asyncio gate (counter + Condition) in front of `drain_generate`;
   - a 1 Hz controller coroutine reading NVML power, the gate queue, and
     `n_running`/`kv_tokens` from iter_logger, then setting max-in-flight;
   - **persist per-request TTFT/TPOT**, which the harness does not record today.
2. **Sysid (15 min):** a `--concurrency_schedule` staircase to fit the A5000
   plant (P(n), t(n) under its 100 W cap) with `sysid.py`.
3. **Run:** Qwen2.5-3B (KV headroom on 24 GB); bursty MMPP arrivals; a *virtual* announced
   budget B(t) stepping 95 → 60 → 95 W, since the hardware cap cannot move. Controllers:
   static concurrency, PI on queue delay, POLCA-like reactive gate, admission-MPC with the
   identified plant and budget preview. 3 reps × 5 min each.
4. **Metrics:** NVML-measured budget violations (10 ms), TTFT/TPOT p50 and p99, J/token.
5. **Prediction from this study:** PI ≈ MPC except right after announced budget steps.
   The live test's real value is validating the throttle/saturation branch and the queue
   dynamics.

## Caveats

- Cap response assumes energy per unit work is cap-invariant (no DVFS efficiency gain).
  This is the least validated piece of either plant. The savings from capping could be
  larger in reality, and slow-down estimates pessimistic.
- Floors (H200 200 W, B200 300 W), P_park and wake time are assumptions, and the
  heterogeneous routing direction depends on the B200 floor.
- No per-request latency data exists to validate the TTFT/TPOT models. SLO metrics are
  fluid approximations.
- Episodes end with MPC holding more in-flight work (≤0.8% of tokens), which slightly
  favours it on total energy. J/token (in the tables) shows the same ranking.
- Not done: disaggregated prefill/decode pools (the 2609.11133 setting), and a replay of
  logged arrivals (all our logged arrivals are synthetic Poisson or closed-loop, so a
  replay adds nothing).

## Reproduce

Run inside the container on `debug` (numpy/scipy); `figs` needs pydeps on PYTHONPATH.

```
python3 -m controls.experiments --exp validate,main,traces,robust,pareto,hetero,sens,park,park2,sizes,horizon,robust32,plantcal --procs 16 --reps 5
PYTHONPATH=/shared_data0/adsampat/pydeps:$PYTHONPATH python3 -m controls.experiments --exp figs
```

Note: `plantcal` was run with `--reps 3`; the calibrated-plant numbers above use 3 seeds.

Static tuning is cached in `results/static_tuned.json`. The full run takes about 40 min
on 16 cores.

---

# Round 2 (2026-09-24): live A5000 experiment built; disaggregated pools partially run

**Status:** the live experiment is ready to run but **not run**. Submitting the GPU job
(`sbatch controls/live_a5000.sbatch`) was denied by the session's permission system, and
later CPU-only `srun` jobs were denied too. So part B's dynamic study and the re-runs with
the corrected e_gemm are still pending. Everything below comes from CPU runs completed
before the denials.

## A. Live closed-loop admission control on a 100-W-capped A5000 (ready to run)

**Files (new; the shared harness is untouched):**
- `controls/live_controllers.py`: admission-only controllers (static, PI on power,
  POLCA-style, policy-search AdmissionMPC) and a request-level CPU simulator of the
  capped GPU.
- `controls/live_load.py`: MMPP open-loop arrivals, an asyncio admission gate, and a
  1 Hz controller reading NVML *instant* power (field 186). It persists per-request
  arrival / admit / first-token / finish times and every controller tick, plus
  `gpu_query.csv` (index, pci.bus_id, power.limit, enforced.power.limit). It has a
  `--mock` engine for CPU dry runs and `--analyze` for summaries (energy integrated
  from the power trace; the NVML energy counter is wrong on this part).
- `controls/a5000_sysid.py`: builds the staircase command (via the harness's
  `--concurrency_schedule`) and fits the n→tok/s / power table. It also runs
  `controls/sysid.py`'s fit with monkeypatched globs and output path, so no shared
  file is written.
- `controls/live_a5000.sbatch`: gpu query → idle → staircase → table + fit → 4
  controllers × 3 reps (shuffled order, one engine load) → analysis. About 1.5 h. Output
  goes to `logs/A5000_live/`.

**Dry runs completed:**
- Mock-engine end-to-end run of all 4 controllers.
- Table and sysid fit on the physics agent's `logs/A5000` sweep. The fit runs, but the
  time model is only 13–24% MAPE in the capped regime; the table is the primary MPC
  model.

**Why admission can work on this plant.** At the enforced 100 W limit the GPU draws about
100 W whenever ≥1 request runs, and idles at about 61 W with a model loaded.
Per-request speed is roughly batch-independent (~20–35 tok/s). So:
- energy ≈ P_busy × busy time;
- the only lever is **batch-synchronous gating**: hold arrivals while idle, release them
  as one large batch, let it drain, and idle.

That is the non-work-conserving batching that gives nothing on uncapped H200/B200.

**CPU predictions for the live run** (request-level sim; Qwen2.5-3B table from
`logs/A5000`; MMPP 0.6/2.0 req/s; 300 s; 3 seeds):

| policy | energy kJ | J/token | TTFT p50 / p99 (s) | budget viol (20 s window) |
|---|---|---|---|---|
| static64 (work-conserving) | 29.8 | 0.488 | 0.03 / 0.03 | 40% (always busy) |
| fixed batch-sync gate, θ=16 | 23.9 | 0.404 | 7.4 / 28.9 | — |
| fixed batch-sync gate, θ=32 | 22.0 | 0.374 | 14.1 / 48.9 | — |
| MPC, flat budget, TTFT SLO 15 s | 25.3 | 0.422 | 6.0 / 14.2 | 0% |
| MPC, flat budget, TTFT SLO 30 s | 23.2 | 0.384 | 10.1 / 25.6 | 0% |
| MPC, budget 100→88→78→100 W, SLO 30 s | 23.3 | 0.401 | 14.2 / 40.5 | 14% |
| PI on power, same budget | 28.5 | 0.541 | 61 / 121 | 39% |

- Predicted: MPC saves about 20% energy per token against a work-conserving server if the
  user accepts roughly 10–30 s TTFT.
- The 78 W budget segment is only partly feasible at this load (MPC still violates 14% of
  windows).
- The live run tests this prediction, the throughput table at small n, and whether
  plant_fp and plant.py predict the real queue/latency dynamics under gating. No logged
  run has ever exercised gating.

## B. Disaggregated prefill/decode pools, heterogeneous fleet (partially run)

**Code:** `controls/disagg.py`.
- Per-GPU prefill model: `e_p = e_gemm·2P + e_kvbyte·1.5·kvb + e_wbyte·W/C`; roofline
  rate at MFU 0.5; work-proportional power.
- Decode model: the fleet-study envelope with the prefill FLOPs removed. Colocated GPUs
  time-share.
- Steady-state MILP over every pool assignment (min power at a given load, and max load
  under a Σcaps budget).
- Dynamic fluid sim: prefill token FIFOs, plus `plant_fp` decode GPUs in the new
  `prefilled` mode.
- Controllers: static TDP-proportional split, per-pool PI, a pool-MPC LP that splits the
  budget between pools with preview, and a blind MPC.
- Coefficients are read from `gpu_coefficients.json` at run time, so the study re-runs
  as-is with the corrected e_gemm. `--egemm_scale` and `--b200_floor` drive the
  sensitivity sweep (`exp_sens`: e_gemm ×0.7–1.3 × B200 floor 300/400/500 W).

**Preliminary steady-state result.** One deterministic LP pass on 2×H200 + 2×B200, 30%
of capacity, **with the corrected coefficients** (H200 e_gemm 0.794 pJ, B200 0.642 pJ).
The sensitivity sweep has not been run. Columns: J/request at the given load, and max
req/s under a Σcaps budget of 45% ΣTDP.

| workload | best disaggregated split | colocated | worst split (P on 2×H200, D on 2×B200) | max req/s @45% budget: colocated / best split |
|---|---|---|---|---|
| 7B chat (512 in / 256 out) | P on B200 (≥1) + D on H200s: 22.7 J/req | 22.7 | 23.8 (+5%) | 69.7 / 59.2 |
| 7B RAG (3000 / 150) | same: 82.4 | 82.4 | 88.8 (+8%) | 17.4 / 15.7 |
| 32B chat | same: 101.9 | 101.9 | 106.9 (+5%) | 11.1 / 12.8 (P:1B+2H, D:1B) |
| 32B RAG | same: 394.8 | 394.8 | 423.8 (+7%) | 3.0 / 3.1 |

- Under our linear additive energy model, a well-chosen disaggregation **ties**
  colocation on energy; it cannot beat it, because colocation with optimal routing is a
  superset.
- **Prefill belongs on the B200** (e_p 9.3 vs 11.4 mJ/token for 7B, from lower e_gemm).
  Decode is split by the floor and idle terms.
- Putting prefill on the H200s instead costs 5–8%.
- Under a budget, colocation gives the most throughput for 7B (69.7 vs 59.2 req/s);
  for 32B a split is slightly better.
- The honest reading: in our model disaggregation's value is SLO isolation (not
  modelled) rather than energy. The pool-MPC budget-splitting dynamic study is written
  but not run.

**Pending (needs compute):**
- `python3 -m controls.disagg --exp steady,sens,dynamic`
- rerun of `experiments.py --exp hetero,sens` with the corrected e_gemm, plus a +26–30%
  e_gemm robustness case
- the live A5000 job
