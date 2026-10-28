# Real workloads on the energy roofline: decode-only trace replay (2026-10-28)

**Goal:** move from characterization to provisioning and research direction. Where do
*real* decode workloads sit on each GPU's energy roofline? What does serving them at a
latency SLO cost per GPU? And which loss terms are worth attacking?

## Workloads
These are public production traces, normalized by `traces/prepare_traces.py` to
`{"t", "in", "out"[, "h"]}`. Stats are for the whole trace (`traces/trace_stats.json`).

| trace | source | req/s | prompt p50 / mean | output p50 / mean | decode share of tokens |
|---|---|---|---|---|---|
| `azure23_conv` | Azure LLM inference 2023 (Splitwise), chat, 1 h | 5.5 | 1020 / 1155 | 129 / 211 | 15.5% |
| `azure23_code` | same, code completion, 1 h | 2.6 | 1469 / 2048 | 13 / 28 | 1.3% |
| `azure24_peak` | Azure 2024 week trace, hour 32 (Mon 08:00 UTC peak) | 57 | ~965 / 1715 | ~41 / 111 | ~6% |
| `burstgpt` | BurstGPT v2 (Azure OpenAI GPT use), hour 521 | 4.6 | — / 409 | — / 98 | ~19% |
| `mooncake_conv` | Mooncake (Kimi) FAST'25 conversation, 1 h, shared-prefix blocks | 3.4 | 6909 / 12035 | 350 / 343 | 2.8% |
| `mooncake_tool` | Mooncake tool-agent, 1 h | 6.7 | 6346 / 8596 | 30 / 182 | 2.1% |
| `reasoning` | Azure 2023 chat arrivals × measured reasoning lengths (Qwen3-30B-A3B thinking, MATH-500) | 5.5 | ~100–300 | see `traces/reasoning_lengths_*.json` | ~95% |

**Observation already:** in every non-reasoning trace, decode is 1–19% of tokens. But
decode is most of the GPU-seconds, since a decode step processes one token per sequence.
Reasoning flips the token mix.

The Azure 2024 head (first ~37 h, Sunday→Monday) has a per-minute summary in
`traces/azure24_conv_per_min.csv`; it's the input to the fleet/diurnal analysis. Load
runs from 21 req/s (Sunday 04:00) to 57 req/s (Monday 08:00), 2.7× peak/trough.

## Method
`energy_profile_load.py --trace_file … --trace_speedups a,b,c --prefix_warm`; arms
`<gpu>_trace` and `<gpu>_reasoning` in `run_provisioning_sweep.py`.

1. **Open-loop replay.**
   - Each request is submitted at its trace arrival time divided by the speedup.
   - It uses its trace prompt and output lengths: `ignore_eos`, exact output length.
   - Prompts are synthetic token ids: distinct per request, except that Mooncake
     requests share token blocks wherever their `hash_ids` match, so the real prefix
     reuse is reproduced.
2. **Decode-only via prefix warm-up.**
   - Before each level, every prompt is prefilled with `max_tokens=1`, in reverse
     arrival order (unmeasured).
   - In the window, the prefix cache serves each prompt. Smoke tests show a 98.9–99.9%
     hit rate and ≤0.1% computed prefill.
   - This emulates the decode tier of a prefill/decode-disaggregated deployment.
   - Prompt admission is not free: vLLM 0.10.2 hashes every prompt block with
     pickle+sha256 (its only option), about 1.6 µs per prompt token on the A5000 host.
     That's a real cost of a prefix-cached decode server, and it shows up in the
     residual / host-overhead term.
3. **Fitting the cache.** The request set is cut so warmed prompts plus *all* generated
   tokens fit in 80% of the KV cache. vLLM evicts the least recently freed block once
   the never-used blocks run out, so this guarantees nothing warmed is evicted.
   Consequence: windows on small-memory GPUs are short (A6000 Mooncake: tens of
   requests).
4. **Levels.** The same request set is replayed at several speedups in one engine, so
   the work is identical and only the arrival rate differs. Speedup is the fraction of
   the source cluster's traffic routed to this GPU (<1 for the cluster-scale traces).
5. **Reasoning** has one run per rate (10 min of arrivals each; a set sized for the
   slowest rate never reaches steady state at the fastest). It has no warm-up, since its
   prompts are tiny, and the post-arrival drain is capped at 90 s.
6. **Analysis** (`trace_analysis.py`):
   - The steady window per level is [t0 + median request latency, last arrival].
   - Measured: J/token, power vs cap, idle share, TPOT/TTFT p50/p99, mean batch and
     context.
   - Roofline placement: I = FLOP/byte of the steps actually run, achieved vs roofs.
   - LIMINAL-E breakdown on the measured timeline: static, activity, weights, KV, GEMM,
     residual, plus LIMINAL-E's predicted energy as a validation on real traffic.
   - SLO capacity: the highest level meeting TPOT p99 ≤ 50/100 ms and TTFT p99 ≤ 1/2 s.

## What it answers
| question | output |
|---|---|
| Where do real workloads sit on the roofline? | per-level I and achieved FLOP/s vs the bandwidth, compute and power roofs; % of power cap |
| Which losses dominate per workload? | energy shares: static/idle, weight re-streaming at small batch, KV, residual (host/launch/admission) |
| Which GPU serves a workload at an SLO for the least energy? | capacity at SLO, J/token there, per GPU |
| Fleet sizing over a day | per-minute Azure 2024 load × measured P(load) per GPU: peak-provisioned vs consolidated vs consolidated + power-gated |
| Which research lever saves the most? | what-ifs on the breakdown: KV compression (E_kv × (1 − 1/r)), energy-proportional idle (static during idle), host-overhead removal (residual), consolidation (J/token at SLO capacity vs at the trace's native share) |

## Caveats
- **Synthetic token ids.** Data-dependent switching power is not reproduced; lengths,
  arrivals and prefix reuse are.
- **BurstGPT timestamps** are whole seconds, so same-second requests arrive together.
- **Mooncake prompts** are clipped to 32k − output: Qwen2.5-7B's native context. That
  affects ~4% of requests, and long prompts are the most clipped.
- **The reasoning trace pairs** real arrivals with real reasoning lengths from a
  different model family; outputs are sampled with replacement.
- **Model:** Qwen2.5-7B-Instruct (Qwen2-7B architecture, 32k context) on all GPUs.
  On B200, also 32B and the Qwen3-30B-A3B MoE.
