#!/usr/bin/env bash
# The weekly CUDA gate as a night job: exits 75 (night.py re-queues it until its --retry-until) when another run holds nobara's timing lock, so it never overlaps an ad hoc
# booking that took the lock outside the night queue; otherwise it is run-gate-gpu.sh.
set -uo pipefail
SRC=$(cd "$(dirname "$0")/../../.." && pwd)
python3 "$SRC/scripts/timing_lock.py" check --tool gate-gpu-weekly || { echo "nobara's timing lock is held by another run; exit 75 to retry"; exit 75; }
exec bash "$SRC/docs/measurements/gate-gpu-night-2026-10-06/run-gate-gpu.sh" "$@"
