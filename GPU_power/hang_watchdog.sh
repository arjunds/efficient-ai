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
      if [ "${DRY:-0}" = "1" ]; then echo "   DRY=1, not killing"; else
        echo "   scancel$live"; scancel $live; fi
      exit 9
    fi
  done
  sleep 60
done
