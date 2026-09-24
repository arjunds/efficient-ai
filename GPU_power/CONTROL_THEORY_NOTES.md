# Control-theoretic architecture for energy/GPU allocation — viability notes

Exploration (2026-09-24) of whether optimal-control concepts (MPC, state-space,
LQR) apply to the energy model / GPU-splitting problem. Verdict up front: **partly
right, and the honest split is clean** — the GPU *recommender* is static
optimization (MPC is overkill), but **dynamic allocation of a shared, fluctuating
power budget across prefill/decode is a genuine constrained-feedback problem where
MPC is the right tool**, and our energy+roofline model is a near-ideal internal
model for it.

## Control mapping (the key distinction: state has *memory*)
A variable is **state** only if its future depends on its own past (integrator/lag).
Our energy model computes *algebraic outputs*; the real dynamics live in the queue
and KV cache, which we have **not** modeled yet.

- **State x**: queue length per phase `Q_pf,Q_dec` (integrator: arrivals−completions);
  KV occupancy `K` (integrator with HBM ceiling); GPU temp (slow lag, optional).
- **Control u**: per-GPU power cap `P_cap` (fast, ~10s ms actuation); admission `a`;
  chunked-prefill budget; prefill/decode split; replica counts (slow, minutes).
- **Dynamics f**: `Q⁺=Q+Δ(λa−μ)`, `K⁺=K+tok_in−tok_evict`; service rate `μ` from
  roofline+throttle. **This f is the missing piece.**
- **Outputs h**: power `P=E_iter/t_iter`, latency, throughput — our model *is* h.
- **Cost J**: Σ energy s.t. SLO + power-budget constraints.

## Formalism per sub-problem
- **GPU recommendation** → static argmin. No dynamics. MPC is decoration; don't.
- **Hold SLO by admission throttle** → PI/PID. The baseline a controller must beat.
- **Shared fluctuating power budget across phases/GPUs** → **economic MPC**. The
  real target: dynamic, constrained (budget couples GPUs), predictive (forecast
  helps). This is where a *model* pays off and we have one.
- **Autoscaling** → slow MPC / queue-driven, separate loop (cold start = minutes).
- **DVFS setpoint** → already a hardware loop (RAPL/nvml); model as actuator
  (lag+saturation), command the setpoint not the clock.
- **LQR is largely wrong here**: it's unconstrained quadratic-tracking; our problem
  is economic + hard-constrained (budget, SLO, HBM). Constraints are what MPC exists
  for and LQR can't express.
- **RL is dominated by MPC *here*** precisely because we built a good plant model
  (7% cross-size MAPE). RL earns its keep when the plant is unknown.

## Prior art (the idea largely exists)
- **arXiv 2609.11133 (this month)** — "Phase-Decoupled, Model-Calibrated Power
  Control for Disaggregated LLM Serving": MPC with a calibrated power model setting
  *separate prefill/decode power caps* under a total budget. This is the intuition,
  implemented. Explicitly punts on: online adaptation, DVFS, **heterogeneous GPU
  clusters** — our (would-be) edge.
- **POLCA (2308.12908)** — power oversubscription for LLM inference; reactive
  threshold throttling, not predictive. Shows large inference power headroom.
- **Cooperative Distributed GPU Power Capping** — actual economic-MPC + RLS online
  ID under a power budget (training).
- **LPV control for HPC power capping (2608.03367)**; **Hellerstein, *Feedback
  Control of Computing Systems*** (canonical; RAPL as first-order actuation).
- **Approximate queueing model for LLM SLO autoscaling (2609.20957)** — the
  *dynamics* half we lack: state-dependent birth-death chain with a 3-param
  (α overhead, β compute, γ KV) service model — a coarser cousin of our
  (e_wbyte, e_kvbyte, e_gemm).
- **DistServe (2401.09670), Splitwise (ISCA'24)** — established phase
  disaggregation (decode on cheaper/lower-power GPUs), but **static/heuristic**
  placement, not receding-horizon control. Corroborates our L40S-prefill /
  A100-decode finding.
- **Offline Energy-Optimal LLM Serving (2407.04014)** — workload-based energy model
  for heterogeneous serving; static cousin of our recommender.

## Minimal simulatable MPC (per-GPU power caps under a shared budget)
Δ=1s control interval (≈5 of our 200ms bins; > power-cap actuation), horizon N=10.
- x=[Q,K]; u=[P_cap_1..G, a].
- service from our code: `t_iter=max(t_roof, E_dyn/(P_cap−P_static))` (the
  saturation nonlinearity), `μ=n_run/t_iter`, `P=min(E_dyn/t_iter+P_static, P_cap)`.
- f: `Q⁺=Q+Δ(λa−Σμ)`, `K⁺=K+Δ(tok_in−tok_evict)`; λ from a short forecast.
- J=Σ Δ Σ P; constraints: `ΣP_cap≤P_budget(k)` (the coupling), `P_static≤P_cap≤TDP`,
  `Q≤Q_max ⇔ W=Q/μ≤SLO` (Little's law), `K≤K_max`.
- Linearize the throttle → QP, ms solve. `predict()` in recommend_gpu.py is already
  the one-step predictor; add the two accumulators + a solver loop.

MPC only earns its place when (a) budget is shared & time-varying and (b) the queue
stores work across intervals so a forecast lets you pre-throttle. With one GPU +
constant load it collapses to "run at the lowest cap that meets the SLO" (static).

## Viability verdict + first experiment + risks
- **Worth pursuing, narrowly.** Not a blanket "control architecture"; one real
  control problem: shared, fluctuating power budget across phases/GPUs under SLO.
- **First experiment (sim):** EMPC vs (1) POLCA-style threshold throttle, (2) PI on
  admission, on a bursty trace under a *time-varying* budget. MPC should win *only*
  when the budget fluctuates and load is forecastable — if it doesn't beat PI on a
  flat budget, that's the honest "overkill" finding. Then the differentiator: run
  the same controller on a 2nd GPU with datasheet-scaled coeffs, zero-shot.
- **Risks (priority):** (1) **missing state dynamics** — we have h, not f; build &
  validate the queue/KV accumulators against vLLM `waiting`/resident-KV first;
  (2) transient/ramp mismatch at phase boundaries; (3) multi-rate actuation (fast
  power loop vs minute-scale autoscaling — separate loops); (4) saturation
  nonlinearity (→ why not LQR); (5) tail-latency is a lagging noisy signal —
  constrain on Q (instantaneous) via Little's law; (6) certainty-equivalence gets
  optimistic near saturation under bursts.

## ⚠ Tension with the B200 result
The agent's proposed differentiator was a *transferable datasheet-scaled* internal
model. The B200 measurements just **falsified datasheet scaling** (e_byte is ~invariant,
not ∝1/BW). BOTH the empirical work and this control review independently conclude
the model must be driven by **realized/measured utilization, not datasheet peak**.
So the controller's internal model = our energy model with *measured* per-(model,GPU)
utilization, and the "zero-shot from datasheet" edge is weaker than hoped — the
edge is the *interpretable calibrated* model, plus that e_byte turns out to be a
near-universal constant (which is its own kind of transferability).

Full source list: arXiv 2609.11133, 2609.20957, 2308.12908, 2608.03367, 2602.08191,
2212.12180, 2401.09670, 2407.04014; Splitwise (ISCA'24); Hellerstein et al.
