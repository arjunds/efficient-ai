#!/bin/bash
# CPU-only analysis of a completed GPU sweep. No GPU, no charge -- run on the
# login node. Produces the "Record for each GPU" deliverable from
# HANDOFF_CROSSGPU.md and tees everything to <logroot>/RECORD.txt.
#
#   ./b200_analyze.sh [logroot]        (default logs/B200)
set -u
ROOT="${1:-logs/B200}"
OUT="$ROOT/RECORD.txt"
[ -d "$ROOT" ] || { echo "no such log dir: $ROOT"; exit 1; }

{
echo "======================================================================"
echo " GPU ENERGY-COEFFICIENT RECORD   $(date)"
echo " log root: $ROOT"
echo "======================================================================"

echo
echo "### 1. Hardware / datasheet ###"
cat "$ROOT/specs.json" 2>/dev/null || echo "(no specs.json -- was the sweep run via b200_sweep.sbatch?)"

echo
echo "### 2. Measured idle P_static + power-cap check (gotcha #1) ###"
python3 - "$ROOT" <<'PY'
import glob, json, os, sys
root = sys.argv[1]
cap = None
try:
    cap = json.load(open(os.path.join(root, "specs.json")))["p_cap"]
except Exception:
    pass
print(f"{'run':<44}{'idle W':>8}{'avg W':>8}{'%cap':>7}{'tok/s':>10}")
worst = 0.0
for rd in sorted(glob.glob(os.path.join(root, "*", "*"))):
    mp, rp = os.path.join(rd, "run_meta.json"), os.path.join(rd, "results.json")
    if not (os.path.exists(mp) and os.path.exists(rp)):
        continue
    m, r = json.load(open(mp)), json.load(open(rp))
    avg = r.get("avg_power_window_w") or 0.0
    cp = (100.0 * avg / cap) if cap else 0.0
    worst = max(worst, cp)
    print(f"{os.path.relpath(rd, root):<44}{m.get('idle_power_w') or 0:>8.1f}"
          f"{avg:>8.1f}{cp:>6.0f}%{r.get('aggregate_tokens_per_sec') or 0:>10.1f}")
print()
if cap:
    print(f"power cap = {cap:.0f} W; worst observed = {worst:.0f}% of cap")
    print("=> CAP-SATURATED: linear model invalid (gotcha #1)" if worst > 97
          else "=> OK: below cap in every run; coefficients are meaningful")
PY

echo
echo "### 3. Channel fit: 2 / 3 / 4-term, CIs, held-out (fit_channels.py) ###"
python3 fit_channels.py "$ROOT" 2>&1

echo
echo "### 4. Identifiability + bootstrap + held-out transfer (diagnose_fit.py) ###"
echo "    (gotcha #2: need identifiable=True and a wide arithmetic-intensity span)"
python3 diagnose_fit.py "$ROOT" 2>&1

echo
echo "### 5. Cross-GPU scaling law vs published H200 coefficients ###"
LABEL=$(basename "$ROOT")
python3 scaling_check.py "${LABEL}:${ROOT}" 2>&1

echo
echo "### 6. Files to hand back ###"
find "$ROOT" -name binned_table.csv | sed "s|^|    |" | sort
echo
echo "  Once the H200 raw logs are on this server, the better pooled analysis is:"
echo "    python3 pool_gpus.py H200:logs/ragged ${LABEL}:${ROOT}"
echo "======================================================================"
} 2>&1 | tee "$OUT"

echo
echo "wrote $OUT"
