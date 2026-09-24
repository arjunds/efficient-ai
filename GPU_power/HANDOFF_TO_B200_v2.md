# Handoff v2 → B200 server: hardware byte counts + energy microbenchmarks

**Who this is for:** whoever runs the next B200 session (PARCC `dgx-b200`).
**What it answers:** is the ~invariant serving energy per byte (H200 1.076e-10,
B200 1.252e-10 J/byte, see `FINDINGS_B200.md`) the energy of **DRAM** traffic,
or does it lump in **on-chip** (L2/L1) traffic? And why is B200 **+16%** per byte
over H200 when both use HBM3e? Background on the A5000 (GDDR6) side of the test
is in `FINDINGS_A5000.md`.

Everything is in one job script with four stages:
`microbench/b200_microbench.sbatch` (about 1.5 h on 1 B200).

```
git pull            # needs b89a04e or later (canonical models.py) + this handoff's files
sbatch microbench/b200_microbench.sbatch                       # all 4 stages
STAGES=probe sbatch microbench/b200_microbench.sbatch          # permission probe only (~2 min)
```
Results go to `microbench/results_B200.json` and `microbench/ncu_B200/`, with the
job log in `slurm_out/b200-mbench-*.out`. Commit these files back. They are small, and
none of them contains dataset text.

---

## Priority 1: Nsight Compute hardware bytes (ncu IS available)

`ncu` 12.8.1 is inside the vLLM container at `/usr/local/cuda/bin/ncu`. The
earlier "not installed" result came from a probe that only checked the host PATH.
A colleague collected `dram__bytes.sum` on this cluster in July 2026, so the
counters are probably not admin-locked here. Check that before anything else.

### 1a. Permission probe (stage `probe`)
The probe runs ncu on a 1-kernel script. If the output contains
`ERR_NVGPUCTRPERM`, the counters are locked: skip to Priority 3 (dcgmi) and tell
the lead. If it prints a `dram__bytes_read.sum` value, continue.

### 1b. Per-op byte split (stage `ncu_micro`)
The stage runs every microbenchmark op from `energy_microbench.py` under ncu. It
records `dram__bytes_read.sum`, `dram__bytes_write.sum`, `lts__t_bytes.sum` (L2) and
`l1tex__t_bytes.sum` (L1), and repeats each op with `--cache-control none` (L2 stays
warm, which is realistic) and `--cache-control all` (L2 is flushed before each
kernel, which gives the cold bound). `ncu_microbench.py --parse` then prints
`DRAM/analytic`, `L2/analytic` and `L1/analytic` for each op.

What to expect if the ops do what their names say:
- `dram_stream`: DRAM/analytic ≈ 1.0. The working set is 16× L2, which is about
  2 GB on B200. B200's L2 is about 126 MB, split across the two dies.
- `l2_stream_half`: DRAM/analytic ≈ 0 and L2/analytic ≈ 1.
- `l1_stream_8KBperprog`: L2/analytic ≪ 1.
- `gemv_b1`: DRAM/analytic ≈ 1.

If any of these is off, the energy numbers for that op mean something different
from what the op name says. Report the ratios as measured.

### 1c. Serving decode bytes (stage `ncu_serve`)
This stage profiles a batch-1 Qwen2-7B decode: an 8-token prompt and 24 generated
tokens. `ncu_dram_check.py --profile_range` limits the profiler to the measured
`generate()`. That flag is new and backward-compatible. Without it, engine init
(weight load, vLLM's max-batch profiling forward, KV-cache memsets) swamps the
count. Stage 1c parses the result into hardware bytes per token vs the analytic
weight+KV bytes:
- **DRAM/analytic** tests the analytic byte model. It is the long-sought gate #1
  ("do our analytic bytes equal real DRAM bytes?").
- **L2/analytic** and **L1/analytic** give the on-chip traffic per analytic byte.
  This is explanation 2 of `FINDINGS_B200.md`, measured directly.

**ncu serializes and replays kernels.** Never read energy from a profiled run.
Energy comes from the un-profiled stage `energy` or from the serving sweeps. Join
the two by op name or by config.

---

## Priority 2: Energy microbenchmarks (stage `energy`, no profiler)

`microbench/energy_microbench.py` is portable to any NVIDIA GPU and needs no
root. Per test point it takes a 3 s settle and a ≥10 s steady window, and it
reports:

- Power from NVML field 186 (`POWER_INSTANT`), with the total-energy counter and
  the 1 s-averaged `GetPowerUsage` as cross-checks.
  **Check `energy_counter_over_inst` in the json.** On the A5000 (driver 565) the
  energy counter under-read the board power by 4–20× and was unusable. If B200
  shows ≈1.0, the counter is fine there.
- Two idle conventions: `idle_pre` (context alive, no work, the state vLLM's
  idle baseline sees) and `idle_p0_pre` (clocks held up by a no-op keepalive).
  **Report both.** On the A5000 they differ by 18 W for the same GPU (61 W at
  1695 MHz vs 79 W at 1905 MHz). If B200's 237 W idle was taken at lower clocks
  than serving runs at, part of the "+16%" is idle-subtraction, not per-byte energy.
- Continuous tests (`stream` from L1-resident to 4 GB, `vendor`, `gemv`, `gemm`,
  `grid`): these give J/byte and J/flop with repeat spread. B200 is uncapped
  (≤68% of 1000 W in every sweep), so these are valid there. They are **not**
  valid on the A5000, which is capped.
- Duty-cycle slopes (`duty`): 1.5 ms bursts at 0–7.5% duty, fitted as
  `P = floor + e·rate`. This gives the marginal J/byte at the operating clocks,
  with the floor fitted rather than assumed. It works under any power cap and
  isolates the idle-subtraction issue. **On B200 the `duty` e and the continuous
  `j_per_byte_dyn` should agree.** If they don't, the idle convention is the reason.
- `duty10`: the short-gap version of `duty`. The period is fixed at 10 ms and
  the floor is re-measured before each point. **Prefer this over `duty`** (see
  lesson 5).
- `burstscan`: this sweeps burst length at a fixed mean rate, with the floor
  re-measured before every point. It checks whether short-burst slopes carry a
  per-burst wake-up cost.

**What answers the questions:**

| B200 number | compare with | tells you |
|---|---|---|
| `gemv_b1` J/byte (continuous and duty) | serving direct c=1 1.252e-10 | Whether serving ≈ pure weight streaming. On A5000 the two agree within ~10% in the same DVFS state. |
| `dram_stream` − `l2_stream_half` J/byte | same difference on A5000 (1.7e-10) | Energy of the DRAM side alone, HBM3e vs GDDR6 (hypothesis 1). |
| `l2_stream_half` J/byte, and L2/analytic from 1c | A5000 L2 0.86e-10 J/B | Share of serving J/byte that is on-chip (hypothesis 2). |
| `l2_stream_half` vs `l2_stream_sixteenth` J/byte | each other | B200's two-die L2: far-die L2 hits cross the die-to-die link, a candidate for the +16%. |
| `gemm_fp16_4096` pJ/flop and achieved TFLOPS | 1/peak prediction 0.296 pJ; the old 8192³ test gave 1520 TF | Compute channel. |

---

## Priority 3 (fallback): DCGM `DRAM_ACTIVE` during serving

Use this only if ncu counters are locked. `/usr/bin/dcgmi` exists on the host
node but not in the container. The harness (`dram_counter.py`) already samples
field 1005 whenever `dcgmi` is on PATH. Steps:

1. On the **host** (in the job, before `apptainer exec`):
   ```
   ldd /usr/bin/dcgmi                      # note libdcgm*.so* paths (e.g. /usr/lib/x86_64-linux-gnu/)
   dcgmi discovery -l                      # needs nv-hostengine running on the node; lists GPU ids + PCI bus ids
   nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader   # the GPU SLURM gave you
   ```
   The DCGM GPU id is the **host** index whose PCI bus id matches. It is usually
   not 0 under SLURM. If `dcgmi discovery` cannot connect, no hostengine is
   running and this route is closed: ask the admins, don't start one yourself.
2. Bind the binary and its libraries in, and point the harness at them:
   ```
   export APPTAINERENV_DCGM_GPU_ID=<host id from step 1>        # new env override in dram_counter.py
   export APPTAINERENV_DCGMI_BIN=/opt/dcgm/dcgmi
   export APPTAINERENV_LD_LIBRARY_PATH=/opt/dcgm/lib
   apptainer exec --nv -B /usr/bin/dcgmi:/opt/dcgm/dcgmi \
       -B /usr/lib/x86_64-linux-gnu/libdcgm.so.4:/opt/dcgm/lib/libdcgm.so.4 \
       ... (plus any other libdcgm* ldd listed) ...
   ```
   The container shares the host network namespace, so `dcgmi` reaches the
   hostengine on localhost:5555. Test with `dcgmi dmon -e 1005 -c 5 -i $DCGM_GPU_ID`
   inside the container before running a sweep.
3. Rerun the c=1 Qwen2-7B points (`RAGGED_CONCURRENCY=1 RAGGED_TASKS=alpaca`).
   `results.json` then has `dram_backend: "dcgmi"` and `iter_log.dram_bytes` is
   filled in. `gate_dram.py` compares bytes per token with the analytic value.

**Caveats:**
- `DRAM_ACTIVE` is the fraction of cycles the DRAM interface was busy,
  multiplied by an *assumed* peak (8.0e12 B/s for B200 in `PEAK_HBM_BW`). It is
  an activity ratio, not a byte count. A match to analytic bytes validates the
  combination (analytic bytes, assumed peak, activity definition), and partly
  assumes what it is testing. It cannot split L2 from DRAM.
- DCGM profiling fields and ncu cannot run at the same time.
- The DCGM exporter may already be watching profiling fields. Watching them
  twice is fine, but a `dcgmi profile --pause` by the admins makes 1005 read 0.

---

## Lessons from the A5000 that apply here
1. **Read the enforced power limit, not the TDP.** `nvidia-smi -q -d POWER` and
   `Current Power Limit`. The A5000s turned out to be admin-capped at 100 W
   (TDP 230 W), and every serving run was pinned at the cap.
2. **Energy per byte depends on the V/f operating point.** SM-side energy per
   byte changes with SM voltage and clock, so it is not a pure memory constant.
   Record `sm_mhz` during every measurement. B200 serving and the microbenchmark
   should be compared at similar SM clocks.
3. **The NVML energy counter can be wrong.** Use `POWER_INSTANT` (field 186),
   and cross-check the counter before trusting it.
4. **Check whether the B200 power reading is smoothed.** The NVML docs say
   `nvmlDeviceGetPowerUsage` is a 1 s moving average on Ampere-non-GA100 and
   newer parts, which includes B200. `microbench/nvml_smoothing_check.py`
   shows that the existing H200/B200 per-bin fits are best explained with a
   ~0.4 s smear. Correcting for it leaves `e_wbyte` within −1 to −2%, lowers
   `e_kvbyte` by 17–40%, and raises **`e_gemm` by 26–30%**.
   The `square` test in stage `energy` settles the question on B200: compare
   `corr_on_vs_p_avg` with `corr_on_vs_p_inst` in `results_B200.json`
   (A5000: 0.39 vs 0.64). `nvml_logger.py` now also writes `power_inst_w`
   (additive; existing readers are unaffected), so re-run sweeps get the better
   signal for 200 ms bins.
5. **Use short gaps in any duty-cycled measurement.** With gaps ≥50 ms the A5000
   floor sinks between bursts and burst energy vanishes from floor-subtracted
   power (`burstscan`). Use the `duty10` design (10 ms period, floor re-measured
   before each point) if you duty-cycle anything.
