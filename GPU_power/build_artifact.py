#!/usr/bin/env python3
"""Build the self-contained proposal.html artifact: inline the 4 figures as
data URIs into the designed page. Pure stdlib (runs on login python3)."""
import base64, os

FIGS = {
    "FIG1": "plots_proposal/fig1_size_independence.png",
    "FIG2": "plots_proposal/fig2_phase_decomposition.png",
    "FIG3": "plots_proposal/fig3_transfer.png",
    "FIG4": "plots_proposal/fig4_recommender.png",
}


def uri(p):
    with open(p, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


HTML = r"""<title>Energy-Optimal GPU Selection for LLM Serving — Proposal</title>
<style>
:root{
  --bg:#f5f7f9; --surface:#ffffff; --surface-2:#eceff3; --plate:#ffffff;
  --ink:#141920; --ink-2:#515a66; --ink-3:#7c8794; --line:#d7dde4;
  --mem:#1f5fbf; --compute:#d1701a; --boundary:#a33862;
  --good:#1f8f5f;
  --font-sans:system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  --font-mono:ui-monospace,"SF Mono","JetBrains Mono",Menlo,Consolas,monospace;
  --maxw:880px;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --bg:#0c0f13; --surface:#151a21; --surface-2:#1b222b; --plate:#f7f8fa;
  --ink:#e7ecf2; --ink-2:#a3aeba; --ink-3:#717d8a; --line:#28313b;
  --mem:#5f9bf0; --compute:#e79a52; --boundary:#d178a0; --good:#4fbc88;
}}
:root[data-theme="dark"]{
  --bg:#0c0f13; --surface:#151a21; --surface-2:#1b222b; --plate:#f7f8fa;
  --ink:#e7ecf2; --ink-2:#a3aeba; --ink-3:#717d8a; --line:#28313b;
  --mem:#5f9bf0; --compute:#e79a52; --boundary:#d178a0; --good:#4fbc88;
}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{
  margin:0; background:var(--bg); color:var(--ink);
  font-family:var(--font-sans); font-size:17px; line-height:1.62;
  -webkit-font-smoothing:antialiased;
}
.wrap{max-width:var(--maxw); margin:0 auto; padding:0 24px}
a{color:var(--mem); text-underline-offset:2px}

/* ---- header ---- */
header{padding:64px 0 30px; border-bottom:1px solid var(--line)}
.eyebrow{font-family:var(--font-mono); font-size:12px; letter-spacing:.14em;
  text-transform:uppercase; color:var(--ink-3); margin:0 0 14px}
h1{font-size:clamp(28px,4.4vw,42px); line-height:1.1; margin:0 0 16px;
  letter-spacing:-.02em; text-wrap:balance; font-weight:680}
.thesis{font-size:clamp(18px,2.3vw,21px); color:var(--ink-2); margin:0;
  max-width:62ch; text-wrap:pretty}
.meta{display:flex; flex-wrap:wrap; gap:8px; margin-top:24px}
.chip{font-family:var(--font-mono); font-size:12px; color:var(--ink-2);
  background:var(--surface-2); border:1px solid var(--line); border-radius:999px;
  padding:4px 11px}
.chip b{color:var(--ink); font-weight:600}

/* ---- readout panel ---- */
.readouts{display:grid; grid-template-columns:repeat(4,1fr); gap:14px;
  margin:34px 0 6px}
@media(max-width:720px){.readouts{grid-template-columns:repeat(2,1fr)}}
.tile{background:var(--surface); border:1px solid var(--line); border-radius:12px;
  padding:16px 16px 15px; position:relative; overflow:hidden}
.tile::before{content:""; position:absolute; left:0; top:0; bottom:0; width:3px;
  background:var(--accent,var(--mem))}
.tile .k{font-family:var(--font-mono); font-size:11px; letter-spacing:.06em;
  text-transform:uppercase; color:var(--ink-3); margin:0 0 8px}
.tile .v{font-family:var(--font-mono); font-size:23px; font-weight:600;
  font-variant-numeric:tabular-nums; letter-spacing:-.01em; line-height:1;
  color:var(--ink)}
.tile .u{font-family:var(--font-mono); font-size:12px; color:var(--ink-3)}
.tile .d{font-size:13px; color:var(--ink-2); margin:9px 0 0; line-height:1.4}

/* ---- sections ---- */
section{padding:44px 0; border-bottom:1px solid var(--line)}
.snum{font-family:var(--font-mono); font-size:12px; letter-spacing:.14em;
  color:var(--mem); text-transform:uppercase}
h2{font-size:clamp(21px,2.8vw,27px); margin:6px 0 18px; letter-spacing:-.015em;
  font-weight:660; text-wrap:balance}
h3{font-size:17px; margin:26px 0 6px; color:var(--ink); font-weight:640}
p{margin:0 0 15px; max-width:68ch}
strong{font-weight:640}
.lede{font-size:18.5px; color:var(--ink-2)}
ul{margin:0 0 15px; padding-left:20px} li{margin:5px 0; max-width:66ch}

/* equation / formula */
.formula{font-family:var(--font-mono); font-size:15px; background:var(--surface-2);
  border:1px solid var(--line); border-radius:10px; padding:16px 18px; margin:18px 0;
  overflow-x:auto; color:var(--ink); line-height:1.7}
.formula .mem{color:var(--mem); font-weight:600}
.formula .cmp{color:var(--compute); font-weight:600}
.formula .cmt{color:var(--ink-3)}

/* figure plate */
figure{margin:22px 0 8px}
.plate{background:var(--plate); border:1px solid var(--line); border-radius:12px;
  padding:14px; overflow-x:auto}
.plate img{display:block; width:100%; height:auto; border-radius:4px}
figcaption{font-size:14px; color:var(--ink-2); margin-top:12px; max-width:70ch}
figcaption b{color:var(--ink)}

/* callouts */
.note{border-left:3px solid var(--boundary); background:var(--surface);
  border-radius:0 10px 10px 0; padding:13px 16px; margin:18px 0; font-size:15px;
  color:var(--ink-2)}
.note b{color:var(--boundary)}
.key{border-left:3px solid var(--good); background:var(--surface);
  border-radius:0 10px 10px 0; padding:13px 16px; margin:18px 0; font-size:15px;
  color:var(--ink-2)}
.key b{color:var(--good)}

/* table */
.tblwrap{overflow-x:auto; margin:18px 0}
table{border-collapse:collapse; width:100%; font-size:14.5px}
th,td{text-align:left; padding:10px 12px; border-bottom:1px solid var(--line);
  vertical-align:top}
th{font-family:var(--font-mono); font-size:11.5px; letter-spacing:.05em;
  text-transform:uppercase; color:var(--ink-3); font-weight:600}
td b{color:var(--ink)}
.delta{color:var(--good); font-weight:600}
tbody tr:last-child td{border-bottom:none}

/* phase legend inline */
.dot{display:inline-block; width:10px; height:10px; border-radius:2px;
  vertical-align:baseline; margin-right:5px}
.dot.m{background:var(--mem)} .dot.c{background:var(--compute)}

footer{padding:34px 0 70px; color:var(--ink-3); font-size:13.5px}
footer code{font-family:var(--font-mono); font-size:12.5px; color:var(--ink-2)}
.reveal{opacity:0; transform:translateY(10px); animation:rise .6s ease forwards}
@keyframes rise{to{opacity:1; transform:none}}
@media(prefers-reduced-motion:reduce){.reveal{animation:none; opacity:1; transform:none}}
</style>

<div class="wrap">
<header class="reveal">
  <p class="eyebrow">Research proposal · preliminary results</p>
  <h1>A physically-grounded energy model for LLM serving — and energy-optimal GPU selection</h1>
  <p class="thesis">Fit two hardware coefficients from measured GPU power, show they hold across models, sizes, and a Mixture-of-Experts architecture, then use them to pick the energy-optimal GPU for each <em>phase</em> of a request.</p>
  <div class="meta">
    <span class="chip"><b>Hardware</b> NVIDIA H200 (+A100)</span>
    <span class="chip"><b>Serving</b> vLLM 0.10.2 · continuous batching</span>
    <span class="chip"><b>Workload</b> alpaca + sharegpt, ragged</span>
    <span class="chip"><b>Bins</b> 9,274 (dense) + ladder + MoE</span>
  </div>
</header>

<div class="readouts reveal">
  <div class="tile" style="--accent:var(--mem)">
    <p class="k">Memory coeff</p>
    <div class="v">1.06–1.14<span class="u"> e‑10</span></div>
    <p class="d">J/byte, <b>±7%</b> across 64× model size — a hardware constant</p>
  </div>
  <div class="tile" style="--accent:var(--mem)">
    <p class="k">Held-out (dense)</p>
    <div class="v">6.5–11<span class="u"> %</span></div>
    <p class="d">MAPE predicting an <b>unseen model's</b> energy</p>
  </div>
  <div class="tile" style="--accent:var(--compute)">
    <p class="k">MoE transfer</p>
    <div class="v">R²&nbsp;0.79</div>
    <p class="d">held-out <b>30B MoE</b>, a new architecture</p>
  </div>
  <div class="tile" style="--accent:var(--compute)">
    <p class="k">Phase shift</p>
    <div class="v">6×</div>
    <p class="d">compute-energy share, decode → prefill</p>
  </div>
</div>

<section class="reveal">
  <span class="snum">01 — Motivation</span>
  <h2>The gap: interpretable <em>and</em> measured <em>and</em> predictive</h2>
  <p class="lede">LLM inference energy is a first-order cost, but existing models pick two of three.</p>
  <p>Analytical/roofline models (LIMINAL, LLM-Viewer, “Tokens-to-Watt-hours”) estimate energy from datasheet constants but are <strong>never calibrated or validated against measured power</strong>. Learned models (WattGPU) predict power on unseen GPUs well but are <strong>black-box and exclude MoE</strong>. We target coefficients that are <em>measured</em> yet <em>physically meaningful</em>, so they answer a concrete deployment question:</p>
  <div class="key"><b>Which GPU — or which GPU for each phase of serving — minimizes energy for a given workload?</b> This matters because prefill is compute-bound and decode is memory-bound, so the optimal hardware can differ <em>within one request</em>.</div>
</section>

<section class="reveal">
  <span class="snum">02 — Model</span>
  <h2>Measured energy vs analytic work, per 200&nbsp;ms bin</h2>
  <p>We regress <strong>measured</strong> dynamic energy against <strong>analytically computed</strong> work (from model architecture + vLLM per-iteration logs), fixing static power from an idle baseline:</p>
  <div class="formula">
    E<sub>bin</sub> = <span class="mem">e_wbyte·weight_bytes</span> + <span class="mem">e_kvbyte·kv_bytes</span> + <span class="cmp">e_gemm·gemm_flops</span> + P_static·Δt<br>
    <span class="cmt"># Y = measured ∫P dt − P_static·Δt   ·   X = analytic bytes & FLOPs   ·   OLS, coeffs ≥ 0</span>
  </div>
  <h3>Training space (domain of validity)</h3>
  <ul>
    <li><strong>Hardware:</strong> H200, <em>uncapped</em> regime (the model breaks under power capping — §5).</li>
    <li><strong>Models:</strong> dense 7–8B (calibration) · size ladder 0.5–32B · one 30B MoE.</li>
    <li><strong>Load:</strong> concurrency 1–64 + Poisson; variable-length prompts → ragged batches.</li>
    <li><strong>Arithmetic intensity ≈ 1–2000× FLOP/byte</strong> — the identifiability condition that lets the coefficients separate. Extrapolation beyond (fp8, 100k-context, capped) is out of domain.</li>
  </ul>
</section>

<section class="reveal">
  <span class="snum">03 — Result</span>
  <h2>Coefficients are hardware constants — even across 64× model size</h2>
  <p>Across a single-architecture size ladder (Qwen2.5, 0.5B→32B), the <strong>memory</strong> coefficient is essentially constant, and splitting weight-bytes from KV-bytes (3-term) lifts fit quality at every size. The <strong>compute</strong> coefficient is the honest weak spot.</p>
  <figure>
    <div class="plate"><img alt="Left: e_wbyte flat 1.06-1.14e-10 across 0.5-32B; right: e_flop drifts and 3-term R2 exceeds 2-term at every size" src="%%FIG1%%"></div>
    <figcaption><b>e_wbyte = 1.06–1.14 ×10⁻¹⁰ J/byte for models ≥1.5B (±7% over 64× size)</b> — tighter than the lumped e_bit, and independent of model size. The compute coefficient drifts ~2× and the fit weakens for large memory-bound models (32B) — so <em>memory</em> energy is a clean constant while <em>compute</em> energy needs a utilization/overhead term at the extremes.</figcaption>
  </figure>
  <p class="key" style="margin-top:22px"><b>Not overfit to 7–8B.</b> The FLOP term is essential (bytes-only R² is <em>negative</em>); coefficients have tight bootstrap CIs (e_bit 1.120 [1.115, 1.126]); and fitting on 3 dense models predicts the 4th at 6.5–11% MAPE.</p>
</section>

<section class="reveal">
  <span class="snum">04 — Result</span>
  <h2>Energy splits by phase — the recommender's premise</h2>
  <p>Attributing each bin's dynamic energy to the memory vs compute term shows the split the recommender exploits: decode is almost pure memory; prefill shifts toward compute.</p>
  <figure>
    <div class="plate"><img alt="Stacked bars: decode-heavy 94% memory, prefill-heavy 65% memory / 35% compute" src="%%FIG2%%"></div>
    <figcaption><span class="dot m"></span><b>memory</b> &nbsp; <span class="dot c"></span><b>compute</b> &nbsp;— compute-energy share rises <b>6×</b> from decode-heavy (5.9%) to prefill-heavy (35.1%) bins.</figcaption>
  </figure>
</section>

<section class="reveal">
  <span class="snum">05 — Result</span>
  <h2>Transfer to a new architecture (MoE) — with a routing correction</h2>
  <p>Fitting on 4 dense models and predicting a held-out <strong>30B Mixture-of-Experts</strong> model fails with naive per-token accounting — until weight-byte traffic is modeled by <em>expert occupancy</em>: the expected distinct experts a batch touches, <span style="font-family:var(--font-mono)">E·(1−(1−k/E)<sup>t</sup>)</span>, not per-token active params.</p>
  <figure>
    <div class="plate"><img alt="Bars: dense held-out 8.5% (R2 0.69), MoE naive 62% (R2 -1.35), MoE +occupancy 21% (R2 0.79) vs 115% baseline" src="%%FIG3%%"></div>
    <figcaption>The occupancy fix is <b>necessary and sufficient</b>: R² −1.35 → <b>0.79</b> (MAPE 62% → 21%; naive baseline 115%). It also collapses the MoE's <em>own</em> fitted e_bit from 4.8e-10 onto the dense ~1.1e-10 — independent confirmation that e_bit is hardware, not model.</figcaption>
  </figure>
  <div class="note"><b>Honest:</b> 21% is rougher than dense-to-dense (6–11%); the occupancy formula assumes uniform routing (real routing has ~20–31% expert overlap), so it is an upper bound. The formula itself is standard combinatorics from MoE performance work — our contribution is applying it to <em>energy</em> and showing it enables cross-architecture transfer.</div>
</section>

<section class="reveal">
  <span class="snum">06 — Proposal</span>
  <h2>Energy-optimal GPU selection — per phase</h2>
  <p>The interpretable memory/compute split feeds a recommender: given a workload, predict per-phase energy on each candidate GPU (3-term model + roofline latency + power-cap throttling) and pick the winner. Because prefill is compute-bound and decode memory-bound, <strong>the optimal GPU can differ between phases</strong>.</p>
  <figure>
    <div class="plate"><img alt="Prefill favors high-FLOPS parts (L40S beats A100); decode favors high-bandwidth parts (A100 beats L40S 2x)" src="%%FIG4%%"></div>
    <figcaption><b>L40S and A100 swap rank between phases:</b> L40S wins prefill (more FLOPS), A100 wins decode by ~2× (more bandwidth) — the model recommends disaggregated prefill/decode placement. H200 coefficients are measured; other GPUs are datasheet-scaled (e_byte ∝ 1/BW, e_flop ∝ 1/FLOPS) as the proposed generalization.</figcaption>
  </figure>
  <div class="note"><b>Boundary result:</b> the same fit on a power-capped A100-80-PCIe gives <em>negative</em> R² — its 300 W cap is saturated at every operating point, so energy ≈ P_cap·t, not work. A constant-power model wins there. The two-term model is valid in the <em>uncapped</em> regime; serving-time use needs the cap term <span style="font-family:var(--font-mono)">P = min(e·rates + P_static, P_cap)</span>.</div>
</section>

<section class="reveal">
  <span class="snum">07 — Positioning</span>
  <h2>Honest placement vs prior work</h2>
  <div class="tblwrap"><table>
    <thead><tr><th>Component</th><th>Closest prior</th><th>Our delta</th></tr></thead>
    <tbody>
      <tr><td>Two-term memory+compute energy form</td><td>Choi 2013 (roofline energy); Horowitz 2014</td><td>form is <b>not</b> novel</td></tr>
      <tr><td>Analytical LLM-inference energy</td><td>“Tokens-to-Watt-hours” 2025</td><td>they use datasheet constants — <b>no measured-power calibration/validation</b></td></tr>
      <tr><td>Cross-GPU inference power prediction</td><td>WattGPU</td><td>they are black-box XGBoost & exclude MoE; ours is <b>interpretable + MoE</b></td></tr>
      <tr><td>MoE expert-occupancy byte count</td><td>MoE-CAP; MoE latency work</td><td>we apply it to <b>energy</b> + show it enables cross-arch transfer</td></tr>
      <tr><td>Power waveform from scheduler state</td><td>“Smoothing the Ramp” 2025</td><td>overlaps; we tie it to the fitted model</td></tr>
    </tbody>
  </table></div>
  <p><strong>What is ours (integration + empirical):</strong> measured-power-calibrated, interpretable coefficients shown to be transferable hardware constants — across models, sizes, and one MoE — packaged into a phase-aware energy-optimal GPU recommender. No single prior work does calibrated + interpretable + MoE + phase-level selection together. Framing: a <strong>measurement/systems</strong> paper (MLSys / workshop tier), not a new-model paper.</p>
</section>

<section class="reveal">
  <span class="snum">08 — Risks &amp; plan</span>
  <h2>What would make this a paper</h2>
  <ul>
    <li><strong>Byte/FLOP counts are analytic, not HW-measured</strong> (perf counters blocked by <code style="font-family:var(--font-mono)">ERR_NVGPUCTRPERM</code>). e_* are effective coefficients; a DCGM/NCU cross-check would ground them. <em>(needs admin)</em></li>
    <li><strong>Cross-GPU scaling rests on one clean GPU (H200)</strong> + datasheet priors; needs 2–3 measured uncapped GPUs to become a result, not a hypothesis. <em>(needs hardware access)</em></li>
    <li><strong>MoE transfer (21%) assumes uniform routing</strong> — measure real expert occupancy to tighten it.</li>
  </ul>
  <h3>Next three experiments, by leverage</h3>
  <ul>
    <li><b>1.</b> Size-ladder analysis — <span class="delta">done</span> (memory coeff size-independent).</li>
    <li><b>2.</b> Measure e_* on 2–3 more GPUs — turns the recommender from proposal into result.</li>
    <li><b>3.</b> Hardware byte validation via DCGM/NCU if perf counters get enabled.</li>
  </ul>
</section>

<footer class="reveal">
  <p>Reproducible from <code>binned_table.csv</code> per run via <code>diagnose_fit.py</code>, <code>fit_channels.py</code>, <code>predict_moe.py</code>, <code>recommend_gpu.py</code>, <code>plot_proposal.py</code>. Coefficient summaries in <code>two_term_summary*.csv</code>. Full narrative in <code>SESSION_LOG.md</code>. Branch <code>vllm_ragged</code>.</p>
</footer>
</div>
"""

for k, p in FIGS.items():
    HTML = HTML.replace("%%" + k + "%%", uri(p))
with open("proposal.html", "w") as f:
    f.write(HTML)
print("wrote proposal.html", os.path.getsize("proposal.html") // 1024, "KB")
