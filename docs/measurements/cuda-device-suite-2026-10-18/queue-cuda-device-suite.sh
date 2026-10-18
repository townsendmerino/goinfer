#!/usr/bin/env bash
# Queue tonight's cuda-device-suite on nobara, once: run by goinfer-cuda-device-suite-queue.timer at 18:00. night.py has no recurring jobs, so this is the
# recurrence. It adds the job and does nothing else (the owner starts the queue at bedtime); it skips when a job of that name is already queued.
set -uo pipefail
REPO=${GOINFER_NIGHT_REPO:-$HOME/mycode/goinfer}
cd "$REPO" || { echo "FATAL: $REPO is missing"; exit 1; }
SCRIPT=docs/measurements/cuda-device-suite-2026-10-18/run-cuda-device-suite.sh
[ -f "$SCRIPT" ] || { echo "FATAL: $REPO has no $SCRIPT (is this checkout behind origin/main?)"; exit 1; }
if python3 scripts/night.py list 2>&1 | grep -q 'cuda-device-suite'; then
  echo "cuda-device-suite is already queued; skipping"; exit 0
fi
exec python3 scripts/night.py add cuda-device-suite --est 10 --by "systemd timer, device-only failures" \
  --doc docs/measurements/cuda-device-suite-2026-10-18/README.md -- bash "$SCRIPT"
