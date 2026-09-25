#!/bin/bash
#SBATCH --job-name=b200-mbench
#SBATCH --partition=dgx-b200
#SBATCH --qos=dgx
#SBATCH --account=leebcc-systems-architecture-int
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=slurm_out/%x-%j.out
#
# B200 energy microbenchmarks + Nsight Compute byte counts. See
# HANDOFF_TO_B200_v2.md. Submit from the GPU_power/ checkout:
#   sbatch microbench/b200_microbench.sbatch                 # all stages
#   STAGES=probe,ncu_micro sbatch microbench/b200_microbench.sbatch
# Stages (comma list in STAGES):
#   probe       ncu counter-permission probe (ERR_NVGPUCTRPERM?)            ~1 min
#   energy      energy_microbench.py full suite (no profiler)              ~55 min
#   ncu_micro   same ops under ncu: DRAM / L2 / L1 bytes per op            ~10 min
#   ncu_serve   Qwen2-7B batch-1 decode under ncu (hardware bytes/token)    ~10 min
set -u
cd "$SLURM_SUBMIT_DIR"
STAGES="${STAGES:-probe,energy,ncu_micro,ncu_serve}"
BASE=/vast/projects/leebcc/systems-architecture-int/adsampat
CTR=$BASE/containers
SIF="${SIF:-}"
[ -z "$SIF" ] && [ -s "$CTR/vllm-0.10.2.sif" ] && SIF="$CTR/vllm-0.10.2.sif"
[ -z "$SIF" ] && { echo "FATAL: no .sif in $CTR"; exit 1; }
mkdir -p slurm_out microbench/ncu_B200
source /vast/parcc/sw/etc/profile.d/Z98-lmod.sh
module load apptainer/1.4.1

export APPTAINERENV_HF_HOME=$BASE/hf_cache
export APPTAINERENV_HF_HUB_CACHE=$BASE/hf_cache/hub
export APPTAINERENV_HF_XET_CACHE=$BASE/hf_cache/xet
export APPTAINERENV_PYTHONNOUSERSITE=1
export APPTAINERENV_VLLM_USE_TRTLLM_ATTENTION=0
export APPTAINERENV_TOKENIZERS_PARALLELISM=false
[ -n "${HF_TOKEN:-}" ] && export APPTAINERENV_HF_TOKEN="$HF_TOKEN"

echo "==== B200 microbench $(date) node=$(hostname) stages=$STAGES ===="
nvidia-smi --query-gpu=name,pci.bus_id,power.limit,power.default_limit,clocks.max.sm,clocks.max.mem,driver_version --format=csv

RUN() { apptainer exec --nv -B "$SLURM_SUBMIT_DIR:/workspace" -B "$BASE/hf_cache:$BASE/hf_cache" "$SIF" bash -lc "cd /workspace && $1"; }
NCU=/usr/local/cuda/bin/ncu
METRICS=dram__bytes_read.sum,dram__bytes_write.sum,lts__t_bytes.sum,l1tex__t_bytes.sum

case ",$STAGES," in *,probe,*)
  echo "=== [probe] ncu counter permission ==="
  RUN "$NCU --version | head -2; $NCU --metrics dram__bytes_read.sum python3 microbench/ncu_microbench.py --op probe 2>&1 | tail -15" \
    | tee microbench/ncu_B200/probe.txt
  grep -q ERR_NVGPUCTRPERM microbench/ncu_B200/probe.txt && echo ">>> COUNTERS LOCKED (ERR_NVGPUCTRPERM): skip ncu stages, use dcgmi fallback"
;; esac

case ",$STAGES," in *,energy,*)
  echo "=== [energy] energy_microbench.py (NO profiler attached) ==="
  RUN "python3 -W ignore -u microbench/energy_microbench.py --out microbench/results_B200.json --repeats 3"
;; esac

case ",$STAGES," in *,ncu_micro,*)
  echo "=== [ncu_micro] per-op hardware bytes ==="
  for op in dram_stream l2_stream_half l2_stream_sixteenth l1_stream_8KBperprog gemv_b1 gemm_fp16_4096; do
    for cc in none all; do
      d=microbench/ncu_B200/cache_$cc; mkdir -p $d
      RUN "$NCU --profile-from-start off --cache-control $cc --clock-control none --metrics $METRICS \
             --csv --page raw --log-file $d/$op.csv \
             python3 -W ignore microbench/ncu_microbench.py --op $op --launches 3 --meta $d/$op.json" \
        2>&1 | tail -3
    done
  done
  RUN "python3 microbench/ncu_microbench.py --parse microbench/ncu_B200/cache_none; \
       python3 microbench/ncu_microbench.py --parse microbench/ncu_B200/cache_all"
;; esac

case ",$STAGES," in *,ncu_serve,*)
  echo "=== [ncu_serve] Qwen2-7B batch-1 decode under ncu ==="
  for cc in none all; do
    RD=microbench/ncu_B200/serve_$cc; mkdir -p $RD
    RUN "$NCU --target-processes all --profile-from-start off --replay-mode kernel --cache-control $cc --clock-control none \
           --metrics $METRICS --csv --page raw --log-file $RD/prof.csv \
           python3 ncu_dram_check.py --mode generate --profile_range --model Qwen/Qwen2-7B-Instruct \
             --run_dir $RD --prompt_len 8 --gen_tokens 24 --max_model_len 2048 \
             --gpu_memory_utilization 0.6 2>&1 | tail -5; \
         python3 microbench/ncu_microbench.py --serving_csv $RD/prof.csv \
             --model Qwen/Qwen2-7B-Instruct --n_tokens \$(cat $RD/n_tokens.txt) --ctx 20"
  done
;; esac
echo "==== end $(date) ===="
