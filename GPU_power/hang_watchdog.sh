#!/bin/bash
# Detect and kill a DEADLOCKED provisioning/trace job.
#
# Why: jobs 10061617/10062213 (2026-11-04) hung on the mooncake_tool trace and
# sat at SLURM state RUNNING for 3.5 h doing nothing, burning 7.8 GPU-h before a
# human noticed. Polling job STATE cannot see this -- the job really is running.
#
# The hang signature is a divergence between two writers in the same run dir:
#   power_trace.csv  -- NVML logger thread, still being appended (process alive)
#   iter_log.csv     -- written per engine step, FROZEN (engine not stepping)
# A live run advances both. A deadlocked one advances only the power trace, which
# is also why it silently grows to ~100 MB of useless samples.
#
# Usage: ./hang_watchdog.sh <stale_min> <jobid>...     (kills via scancel)
#        DRY=1 ./hang_watchdog.sh 15 123   -> report only
set -u
STALE_MIN="${1:-15}"; shift
JOBS="$*"
ROOT="${PROV_LOG_ROOT:-logs/B200_prov}"
while :; do
  live=""; present=""
  for j in $JOBS; do
    st=$(squeue -j "$j" -h -o %T 2>/dev/null)
    [ -n "$st" ] && present="$present $j"
    [ "$st" = "RUNNING" ] && live="$live $j"
  done
  # PENDING counts as present: keep waiting rather than exiting before the start.
  [ -z "$(echo $present)" ] && { echo "[watchdog $(date +%H:%M:%S)] all jobs left the queue; exit"; break; }
  [ -z "$(echo $live)" ] && { sleep 60; continue; }
  # An ACTIVE run dir = power_trace.csv touched in the last 3 min.
  for pt in $(find "$ROOT" -name power_trace.csv -mmin -3 2>/dev/null); do
    d=$(dirname "$pt"); il="$d/iter_log.csv"
    [ -s "$il" ] || continue                      # model load: no steps yet, not a hang
    if [ -z "$(find "$il" -newermt "-${STALE_MIN} min" 2>/dev/null)" ]; then
      age=$(( ( $(date +%s) - $(stat -c %Y "$il") ) / 60 ))
      echo "[watchdog $(date +%H:%M:%S)] HANG: $d"
      echo "   power_trace.csv live ($(du -h "$pt"|cut -f1)) but iter_log.csv frozen ${age} min"
      # Kill ONLY the job that owns this run dir. The first version scancel'd every
      # live job and so killed a healthy 32B/MoE job (10069519) for a hang that
      # belonged to 10069518 -- it lost its remaining runs for nothing. Ownership
      # comes from the arm component of the path vs the arms= each job logged.
      arm=$(echo "${d#$ROOT/}" | cut -d/ -f1)
      owner=""
      for j in $live; do
        hdr=$(head -1 "slurm_out/b200-trace-$j.out" 2>/dev/null)
        case ",$(echo "$hdr" | grep -o 'arms=[^ ]*' | cut -d= -f2)," in
          *",$arm,"*) owner="$j" ;;
        esac
      done
      [ -z "$owner" ] && { echo "   WARN: no job claims arm '$arm'; killing nothing"; exit 8; }
      echo "   arm=$arm owned by job $owner (other live jobs left alone:$(echo $live | sed "s/$owner//"))"
      if [ "${DRY:-0}" = "1" ]; then echo "   DRY=1, not killing"; else
        echo "   scancel $owner"; scancel "$owner"; fi
      exit 9
    fi
  done
  sleep 60
done
