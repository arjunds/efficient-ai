#!/bin/bash
#SBATCH --job-name=b200-longctx
#SBATCH --partition=dgx-b200
#SBATCH --qos=dgx
#SBATCH --account=leebcc-systems-architecture-int
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=28
#SBATCH --time=04:00:00
#SBATCH --output=slurm_out/%x-%j.out
#
# Long-context sweep to separate KV-byte energy from compute energy, plus an NCU check
# of KV byte counts at long context, a logger-overhead A/B, and a probe of which other
# GPU partitions (A100/H100/H200) this account can use. HANDOFF_B200_LONGCTX.md.
#   sbatch b200_longctx.sbatch                         # all stages, ~2.5-3 h
#   STAGES=sweep,analyze sbatch b200_longctx.sbatch    # subset; resume-safe
set -u
cd "$SLURM_SUBMIT_DIR"
STAGES="${STAGES:-probe_gpus,sweep,ncu,analyze}"
BASE=/vast/projects/leebcc/systems-architecture-int/adsampat
CTR=$BASE/containers
SIF="${SIF:-}"
[ -z "$SIF" ] && [ -s "$CTR/vllm-0.10.2.sif" ] && SIF="$CTR/vllm-0.10.2.sif"
[ -z "$SIF" ] && { echo "FATAL: no vllm-0.10.2.sif in $CTR"; exit 1; }
export PROV_LOG_ROOT="${PROV_LOG_ROOT:-logs/B200_prov}"
export PROV_ARMS="${PROV_ARMS:-longctx,longctx32,loggerab}"
mkdir -p "$PROV_LOG_ROOT" slurm_out microbench/ncu_B200

echo "==== B200 long-context  $(date)  node=$(hostname)  stages=$STAGES ===="
nvidia-smi --query-gpu=name,power.limit,enforced.power.limit,memory.total,clocks.max.sm,driver_version --format=csv

case ",$STAGES," in *,probe_gpus,*)
  # Which other GPUs can THIS account use on PARCC? (The UPenn cluster's A100 node
  # rejects our account by partition ACL.) Read-only: sinfo + sbatch --test-only.
  echo "=== [probe_gpus] partitions with A100/H100/H200/L40S and whether we may submit ==="
  { sinfo -h -o "%P|%G|%a|%l" | grep -iE "a100|h100|h200|l40" || echo "(no A100/H100/H200/L40S GRES listed)"
    sacctmgr -nP show assoc user=$USER format=account,partition,qos
    for P in $(sinfo -h -o "%P %G" | grep -iE "a100|h100|h200|l40" | awk '{print $1}' | tr -d '*' | sort -u); do
      G=$(sinfo -h -p "$P" -o "%G" | grep -oiE "(a100|h100|h200|l40s?)[^:,(]*" | head -1)
      echo "--- $P ($G): sbatch --test-only"
      sbatch --test-only -p "$P" -A leebcc-systems-architecture-int --gres=gpu:1 -t 00:10:00 --wrap true 2>&1 | head -2
    done; } 2>&1 | tee "$PROV_LOG_ROOT/gpu_partitions_probe.txt"
;; esac

export APPTAINERENV_HF_HOME=$BASE/hf_cache
export APPTAINERENV_HF_HUB_CACHE=$BASE/hf_cache/hub
export APPTAINERENV_HF_XET_CACHE=$BASE/hf_cache/xet
export APPTAINERENV_PYTHONNOUSERSITE=1
export APPTAINERENV_VLLM_USE_TRTLLM_ATTENTION=0        # Blackwell long-sequence crash fix
export APPTAINERENV_TOKENIZERS_PARALLELISM=false
export APPTAINERENV_PROV_LOG_ROOT="$PROV_LOG_ROOT"
export APPTAINERENV_PROV_ARMS="$PROV_ARMS"
export APPTAINERENV_PROV_DURATION_S="${PROV_DURATION_S:-60}"
[ -n "${HF_TOKEN:-}" ] && export APPTAINERENV_HF_TOKEN="$HF_TOKEN"
source /vast/parcc/sw/etc/profile.d/Z98-lmod.sh
module load apptainer/1.4.1
RUN() { apptainer exec --nv -B "$SLURM_SUBMIT_DIR:/workspace" -B "$BASE/hf_cache:$BASE/hf_cache" "$SIF" bash -lc "cd /workspace && $1"; }

case ",$STAGES," in *,sweep,*)
  echo "=== [sweep] arms=$PROV_ARMS (60 s windows) ==="
  RUN "python3 -u run_provisioning_sweep.py 2>&1"
;; esac

case ",$STAGES," in *,ncu,*)
  # Does the analytic KV byte count hold at long context (paged attention)? Same
  # method as b200_microbench ncu_serve (application replay; the prompt is decoded
  # once before profiling, so the profiled pass hits the prefix cache and measures
  # decode, not prefill). At 30k context KV is ~11% of decode bytes: a 2x KV error
  # would move DRAM/analytic by ~10%.
  NCU=/usr/local/cuda/bin/ncu
  METRICS=dram__bytes_read.sum,dram__bytes_write.sum,lts__t_bytes.sum,l1tex__t_bytes.sum
  for P in 8192 30720; do
    RD=microbench/ncu_B200/serve_ctx$P; mkdir -p $RD
    echo "=== [ncu] Qwen2-7B batch-1 decode at context $P ==="
    RUN "$NCU --target-processes all --profile-from-start off --replay-mode application --cache-control none --clock-control none \
           --metrics $METRICS --csv --page raw --log-file $RD/prof.csv \
           python3 ncu_dram_check.py --mode generate --profile_range --model Qwen/Qwen2-7B-Instruct \
             --run_dir $RD --prompt_len $P --gen_tokens 16 --max_model_len 32768 \
             --gpu_memory_utilization 0.6 2>&1 | tail -3; \
         python3 microbench/ncu_microbench.py --serving_csv $RD/prof.csv \
             --model Qwen/Qwen2-7B-Instruct --n_tokens \$(cat $RD/n_tokens.txt) --ctx $((P + 8))" 2>&1 | tail -25
  done
;; esac

case ",$STAGES," in *,analyze,*)
  echo "=== [analyze] ==="
  RUN "python3 -W ignore longctx_analysis.py --root $PROV_LOG_ROOT 2>&1; python3 -W ignore provisioning_analysis.py --root $PROV_LOG_ROOT 2>&1 | sed -n '/== 1. runs/,/bin energy source/p'"
  for P in 8192 30720; do
    f=microbench/ncu_B200/serve_ctx$P/prof_serving_summary.json
    [ -s "$f" ] && python3 -c "import json;d=json.load(open('$f'));print('ncu ctx $P: DRAM/analytic',round(d['dram_over_analytic'],4),' L2/analytic',round(d['l2_over_analytic'],3))"
  done
;; esac
echo "==== done $(date) ===="
