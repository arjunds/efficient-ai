#!/bin/bash
#SBATCH --job-name=a5k-ncu
#SBATCH --partition=debug
#SBATCH --account=exwong
#SBATCH --gres=gpu:a5000:1
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=microbench/slurm-%x-%j.out
# (1) ncu counter-permission probe + one ncu_microbench op on A5000;
# (2) end-to-end smoke of the FULL default energy suite with tiny windows
#     (guards the B200 run against integration bugs). Output -> microbench/smoke_full/
set -u
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs/A5000_gpu_limits microbench/ncu_A5000 microbench/smoke_full
nvidia-smi --query-gpu=index,pci.bus_id,power.limit,enforced.power.limit,power.default_limit --format=csv > logs/A5000_gpu_limits/job_${SLURM_JOB_ID}.csv
SIF=/shared_data0/fwzhang/vllm.sif
export APPTAINERENV_PYTHONNOUSERSITE=1
NCU=/usr/local/cuda/bin/ncu
M=dram__bytes_read.sum,dram__bytes_write.sum,lts__t_bytes.sum,l1tex__t_bytes.sum
apptainer exec --nv -B "$SLURM_SUBMIT_DIR:/workspace" -B /shared_data0 "$SIF" bash -lc "
  cd /workspace
  $NCU --metrics dram__bytes_read.sum python3 -W ignore microbench/ncu_microbench.py --op probe > microbench/ncu_A5000/probe.txt 2>&1
  tail -8 microbench/ncu_A5000/probe.txt
  if ! grep -q ERR_NVGPUCTRPERM microbench/ncu_A5000/probe.txt; then
    for op in dram_stream l2_stream_half gemv_b1; do
      $NCU --profile-from-start off --cache-control none --clock-control none --metrics $M --csv --page raw \
        --log-file microbench/ncu_A5000/\$op.csv python3 -W ignore microbench/ncu_microbench.py --op \$op --launches 3 \
        --meta microbench/ncu_A5000/\$op.json 2>&1 | tail -2
    done
    python3 microbench/ncu_microbench.py --parse microbench/ncu_A5000
  fi
  python3 -W ignore -u microbench/energy_microbench.py --out microbench/smoke_full/results_smoke.json \
     --repeats 1 --steady_s 1.5 --settle_s 0.5 --idle_s 1 --duty_repeats 1 --duty_steady_s 1 \
     --duties 0,0.03 --duties10 0.03 --scan_lens_ms 0.4 2>&1 | grep -v '^  \[' | tail -30
"
