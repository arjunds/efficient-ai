#!/bin/bash
#SBATCH --job-name=b200-attnfix
#SBATCH --partition=dgx-b200
#SBATCH --qos=dgx
#SBATCH --account=leebcc-systems-architecture-int
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=28
#SBATCH --time=00:30:00
#SBATCH --output=slurm_out/%x-%j.out
#
# Find an attention config that survives long (sharegpt) sequences on B200.
#
# THE BUG: vllm/utils/flashinfer.py::use_trtllm_attention() auto-enables the
# TRTLLM kernel PER BATCH:
#     use_trtllm = (num_tokens <= 256 and max_seq_len < 131072
#                   and kv_cache_dtype == "auto")
# With alpaca (uniformly short prompts) the chosen path is stable and every run
# passed. With sharegpt (16..23k chars) num_tokens swings across the 256
# boundary, so vLLM flips between the TRTLLM and plain-FlashInfer kernels inside
# one run -> "CUDA error: an illegal memory access" in attention/layer.py:249.
# 5 of 8 sharegpt runs died; 0 of 8 alpaca runs did.
#
# Reproduce fast: Qwen2 sharegpt c16 crashed after only 9 bins (~1.8 s), so a
# 30 s run per config is plenty. ~5 min of GPU work total.
set -u
cd "$SLURM_SUBMIT_DIR"
BASE=/vast/projects/leebcc/systems-architecture-int/adsampat
SIF=$BASE/containers/vllm-0.10.2.sif
MODEL=Qwen/Qwen2-7B-Instruct
OUT=logs/B200_attnfix
rm -rf "$OUT"; mkdir -p "$OUT" slurm_out

export APPTAINERENV_HF_HOME=$BASE/hf_cache
export APPTAINERENV_HF_HUB_CACHE=$BASE/hf_cache/hub
export APPTAINERENV_HF_XET_CACHE=$BASE/hf_cache/xet
export APPTAINERENV_PYTHONNOUSERSITE=1
export APPTAINERENV_TOKENIZERS_PARALLELISM=false

source /vast/parcc/sw/etc/profile.d/Z98-lmod.sh
module load apptainer/1.4.1
nvidia-smi --query-gpu=name,power.limit --format=csv,noheader

apptainer exec --nv \
  -B "$SLURM_SUBMIT_DIR:/workspace" -B "$BASE/hf_cache:$BASE/hf_cache" \
  "$SIF" bash -lc '
set -u; cd /workspace
run_cfg () {   # $1 = tag, rest = VAR=VAL ...
  tag=$1; shift
  echo ""; echo "############ CONFIG: $tag  [$*] ############"
  ( export "$@" 2>/dev/null
    python3 -u energy_profile_load.py --mode load --model '"$MODEL"' \
      --dtype float16 --task sharegpt --concurrency 16 --output_len 128 \
      --duration_s 30 --max_model_len 8192 --gpu_memory_utilization 0.90 \
      --run_dir '"$OUT"'/$tag ) > '"$OUT"'/$tag.log 2>&1
  bins=$(tail -n +2 '"$OUT"'/$tag/binned_table.csv 2>/dev/null | wc -l)
  toks=$(python3 -c "import json;print(round(json.load(open(\"'"$OUT"'/$tag/results.json\"))[\"aggregate_tokens_per_sec\"],1))" 2>/dev/null || echo NA)
  if grep -qE "illegal memory access|EngineDeadError|AcceleratorError" '"$OUT"'/$tag.log; then
      echo "RESULT $tag: *** CRASHED *** (tok/s=$toks)"
  else
      echo "RESULT $tag: OK  tok/s=$toks"
  fi
  # bins only exist after energy_model; report raw iter rows instead
  iters=$(tail -n +2 '"$OUT"'/$tag/iter_log.csv 2>/dev/null | wc -l)
  echo "       iters=$iters  (a healthy 30 s c16 sharegpt run is ~1000+)"
}

# 1. baseline: exactly what the sweep did -- must reproduce the crash
run_cfg baseline VLLM_DUMMY=1
# 2. the targeted fix: pin OFF the TRTLLM path so no mid-run kernel switching
run_cfg trtllm_off VLLM_USE_TRTLLM_ATTENTION=0
# 3. fallback: leave FlashInfer entirely, use the Triton attention kernel
run_cfg triton VLLM_ATTENTION_BACKEND=TRITON_ATTN_VLLM_V1
# 4. control: force TRTLLM always ON (is the switching, or TRTLLM itself, at fault?)
run_cfg trtllm_on VLLM_USE_TRTLLM_ATTENTION=1
'

echo
echo "================= SUMMARY ================="
grep -hE "^RESULT|^       iters=" slurm_out/b200-attnfix-$SLURM_JOB_ID.out 2>/dev/null
for t in baseline trtllm_off triton trtllm_on; do
  s=$(grep -qE "illegal memory access|EngineDeadError|AcceleratorError" $OUT/$t.log 2>/dev/null && echo CRASH || echo ok)
  b=$(grep -ohE "Using (TRTLLM attention|FlashInfer backend|Triton.*)" $OUT/$t.log 2>/dev/null|sort -u|tr '\n' ' ')
  printf "  %-12s %-6s %s\n" "$t" "$s" "$b"
done
echo "==== attnfix end $(date) ===="
