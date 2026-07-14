#!/bin/bash
#SBATCH --job-name=egy-reanalyze
#SBATCH --partition=standby
#SBATCH --account=exwong
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=%x-%j.out
cd ~/efficient-ai/GPU_power
for d in logs/first_run/c1 logs/first_run/c8; do
  echo "================ $d ================"
  python3 energy_model.py --run_dir "$d" 2>&1
  echo "---- waveform ----"
  python3 validate_waveform.py --run_dir "$d" 2>&1
done
