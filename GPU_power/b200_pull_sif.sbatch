#!/bin/bash
#SBATCH --job-name=sif-pull
#SBATCH --partition=genoa-std-mem
#SBATCH --qos=genoa-std
#SBATCH --account=leebcc-systems-architecture-int
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=5632M
#SBATCH --output=slurm_out/%x-%j.out
#
# CPU-only job: pull the vLLM container to a .sif. No GPU, so no GPU charge.
#
# CRITICAL: apptainer's unpack step needs a filesystem with extended-attribute
# support. /vast is NFS and returns ENOTSUP for lremovexattr, which fails the
# build with:
#     unpriv.lclearxattrs: unpriv.lremovexattr: invalid argument
# So APPTAINER_TMPDIR/CACHEDIR must live on NODE-LOCAL disk (/tmp). Only the
# finished .sif is written to /vast. Preflight below verifies both space and
# xattr support before burning 30 minutes on a doomed pull.
#
# Both images are CUDA 12.8.1 based -- the first CUDA with sm_100 (B200) support.
#
#   sbatch b200_pull_sif.sbatch                   # v0.10.2 only (matches H200)
#   PULL_FALLBACK=1 sbatch b200_pull_sif.sbatch   # also pull v0.11.0
set -u
CTR=/vast/projects/leebcc/systems-architecture-int/adsampat/containers
NEED_GB=60
mkdir -p "$CTR"

# ---- pick a scratch dir that is local AND supports xattrs ----
pick_tmp () {
  for cand in "/tmp/$USER.apptainer.$SLURM_JOB_ID" "/dev/shm/$USER.apptainer.$SLURM_JOB_ID"; do
    mkdir -p "$cand" 2>/dev/null || continue
    avail=$(df -BG --output=avail "$cand" 2>/dev/null | tail -1 | tr -dc '0-9')
    python3 -c "
import os,sys,tempfile
fd,p=tempfile.mkstemp(dir='$cand'); os.close(fd)
try: os.setxattr(p,b'user.t',b'1'); os.removexattr(p,b'user.t')
except OSError as e: os.unlink(p); sys.exit(1)
os.unlink(p)
" 2>/dev/null || { echo "  [no] $cand: no xattr support"; rm -rf "$cand"; continue; }
    if [ "${avail:-0}" -lt "$NEED_GB" ]; then
      echo "  [no] $cand: only ${avail}G free, need ${NEED_GB}G"; rm -rf "$cand"; continue
    fi
    echo "  [ok] $cand: ${avail}G free, xattr OK"; echo "$cand"; return 0
  done
  return 1
}

echo "==== sif pull start $(date) on $(hostname) ===="
echo "--- selecting scratch ---"
TMPD=$(pick_tmp | tee /dev/stderr | tail -1)
if [ -z "${TMPD:-}" ] || [ ! -d "${TMPD:-/nonexistent}" ]; then
  echo "FATAL: no local scratch with >=${NEED_GB}G and xattr support."; exit 1
fi
export APPTAINER_TMPDIR="$TMPD/tmp" APPTAINER_CACHEDIR="$TMPD/cache"
mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"
trap 'rm -rf "$TMPD"' EXIT     # always reclaim node-local scratch

source /vast/parcc/sw/etc/profile.d/Z98-lmod.sh
module load apptainer/1.4.1
echo "apptainer: $(apptainer --version)   scratch: $TMPD"

pull () {  # $1 = docker tag, $2 = output sif
  if [ -s "$2" ]; then echo "[skip] $2 exists ($(du -h "$2" | cut -f1))"; return 0; fi
  echo "=== pulling $1 -> $2 ==="
  if apptainer pull "$2" "docker://vllm/vllm-openai:$1"; then
    echo "[ok] $2 ($(du -h "$2" | cut -f1))"
  else
    echo "[FAIL] $1"; rm -f "$2"
  fi
  rm -rf "${APPTAINER_CACHEDIR:?}"/* 2>/dev/null   # cache is a full 2nd copy
}

pull v0.10.2 "$CTR/vllm-0.10.2.sif"
[ "${PULL_FALLBACK:-0}" = "1" ] && pull v0.11.0 "$CTR/vllm-0.11.0.sif"

echo "=== containers ==="; ls -lh "$CTR"
echo "==== end $(date) ===="
