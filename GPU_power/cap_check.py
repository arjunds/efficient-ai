"""Model-free power-cap check (FINDINGS_TRACES.md section 1): for every trace level, in the steady window,
the share of NVML samples with the SW-power-cap throttle reason (0x4) set, the median SM clock vs its max,
and mean power. Thermal / HW slowdown bits are reported separately.
  python3 cap_check.py <GPU_power dir>
"""
import csv, glob, json, os, statistics as st, sys
ROOT = sys.argv[1]
for rd in sorted(glob.glob(ROOT + "/logs/*_prov/*/*/*/power_trace.csv")):
    d = os.path.dirname(rd)
    if "mooncake" in d or not os.path.exists(d + "/requests.csv"):
        continue
    P = [r for r in csv.DictReader(open(rd)) if r.get("t_wall") and r.get("power_inst_w")]
    try:
        P = [(float(r["t_wall"]), float(r["power_inst_w"]), float(r["sm_mhz"]), int(float(r["clk_reasons"] or 0))) for r in P]
    except ValueError:
        continue
    fmax = max(p[2] for p in P)
    R = list(csv.DictReader(open(d + "/requests.csv")))
    levels = sorted({r.get("trace_speedup") or "1" for r in R}, key=float)
    for x in levels:
        L = [r for r in R if (r.get("trace_speedup") or "1") == x]
        subs = sorted(float(r["t_submit"]) for r in L)
        ramp = min(st.median(float(r["e2e_s"]) for r in L), 0.4 * (subs[-1] - subs[0]))
        ws, we = subs[0] + ramp, subs[-1]
        S = [p for p in P if ws <= p[0] <= we]
        if len(S) < 5:
            continue
        pc = sum(1 for p in S if p[3] & 0x4) / len(S)          # SW power cap
        th = sum(1 for p in S if p[3] & (0x20 | 0x40 | 0x8 | 0x80)) / len(S)  # thermal / HW slowdown / power brake
        print("%-6s %-22s %-16s x%-6s n=%4d  P=%5.0f W (max %5.0f)  SMclk med %4.0f/%4.0f MHz  pwrcap %3.0f%%  thermal/HW %3.0f%%" % (
            d.split("/")[-4].split("_")[0], d.split("/")[-2][:22], d.split("/")[-1], x, len(S),
            st.mean(p[1] for p in S), max(p[1] for p in S), st.median(p[2] for p in S), fmax, 100 * pc, 100 * th))
