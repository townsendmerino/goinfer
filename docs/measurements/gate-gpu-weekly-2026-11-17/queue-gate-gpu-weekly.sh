#!/usr/bin/env bash
# Queue this week's CUDA gate (`gate gpu`, heavy tier included, about 2 h 20 min) on nobara's night queue: run by goinfer-gate-gpu-weekly.timer. night.py has no recurring
# jobs, so this is the recurrence. It adds the job and does nothing else; it skips when a job of that name is already queued. The weekly run is the heavy tier's
# heartbeat: a release night then confirms a run that was green a week earlier. It does not replace the release record, which must be at the freeze commit.
#
# Like the C12 launcher's nobarafree, it looks at nobara first. The timer fires twice on the night (18:00, 20:30). The first firing exits 75 and queues nothing when the night
# queue has another booked job (any but cuda-device-suite) or the timing lock is held by something that is not the night runner; the 20:30 firing queues it regardless, at
# priority 80, which runs after the cuda-device-suite (50) and after any ad hoc booking, so the runner never overlaps them. The job itself re-checks the lock when it starts.
set -uo pipefail
REPO=${GOINFER_NIGHT_REPO:-$HOME/mycode/goinfer}
cd "$REPO" || { echo "FATAL: $REPO is missing"; exit 1; }
SCRIPT=docs/measurements/gate-gpu-weekly-2026-11-17/run-gate-gpu-weekly.sh
[ -f "$SCRIPT" ] || { echo "FATAL: $REPO has no $SCRIPT (is this checkout behind origin/main?)"; exit 1; }
listing=$(python3 scripts/night.py list 2>&1)
if grep -q -E '(^|[[:space:]])gate-gpu-weekly([[:space:]]|$)' <<<"$listing"; then
  echo "gate-gpu-weekly is already queued; skipping"; exit 0
fi
final=0; [ "$(( 10#${GATE_WEEKLY_HOUR:-$(date +%H)} ))" -ge 20 ] && final=1  # GATE_WEEKLY_HOUR: the test's clock
status=$(python3 scripts/night.py status 2>&1 | head -1)
lock=$(python3 scripts/timing_lock.py status 2>&1 | head -1)
others=$(grep -E '^[[:space:]]*[0-9]+[[:space:]]+[A-Za-z0-9_.-]+' <<<"$listing" | grep -v -E '(^|[[:space:]])cuda-device-suite([[:space:]]|$)' || true)
running_other=$(grep -E '^running ' <<<"$status" | grep -v 'cuda-device-suite' || true)
lock_foreign=0; { grep -q '^free' <<<"$lock" || grep -q '^running ' <<<"$status"; } || lock_foreign=1
if [ "$final" = 0 ] && { [ -n "$others" ] || [ -n "$running_other" ] || [ "$lock_foreign" = 1 ]; }; then
  echo "nobara is booked or its lock is held (queue: ${others:-none}; status: $status; lock: $lock); exit 75, the 20:30 firing retries"; exit 75
fi
exec python3 scripts/night.py add gate-gpu-weekly --est 150 --priority 80 --not-before 22:00 --retry-until 06:00 --by "systemd timer, weekly heavy night" \
  --doc docs/measurements/gate-gpu-weekly-2026-11-17/README.md -- bash "$SCRIPT"
