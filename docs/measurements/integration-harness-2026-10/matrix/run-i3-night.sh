#!/bin/bash
# I3, the real-harness matrix, as an unattended night job (docs/tasks/task-integration-harness-2026-10.md, "I3 run plan,
# registered 2026-10-13"). Runs the configurations named in ONLY, in order, from PINNED copies: the serve binary and the
# harness scripts are copied into $B at queue time, so the tree may move before tonight. Every run is in the sandbox
# (matrix/sandbox.py), whose self-test must pass first; if it does not, nothing runs. night.py holds the timing lock.
#
#   queued as:  ONLY=q3-4b-16k,q3-4b-28k-cc bash <this script>                              (job A, Qwen3-4B; it was the 7B until 2026-10-13)
#               ONLY=mellum-16k,mellum-32k-cc bash <this script>                            (job B, Mellum2.1)
#               ONLY=q3-4b-9k-codex-ovf,q3-4b-11k-opencode-ovf,q3-4b-28k-cc-ovf bash <this script>  (job C, I1's live gate)
set -u
B=${B:-$HOME/goinfer-bench/harness-2026-10-13}
ONLY=${ONLY:?name the configurations to run}
DEADLINE=${DEADLINE_MINUTES:-170}
OUT=${OUT:-$B/i3-out-$(date +%F)-$(echo "$ONLY" | tr ',' '+' | cut -c1-40)}
PLAN=${PLAN:-$B/matrix-pinned/plan-i3.json}
mkdir -p "$OUT" || exit 2
for f in "$B/bin/serve-cuda-i3" "$B/matrix-pinned/matrix.py" "$B/matrix-pinned/sandbox.py" "$PLAN" "$HOME/harnesses/node_modules/.bin/claude" \
         "$HOME/harnesses/node_modules/.bin/codex" "$HOME/harnesses/node_modules/.bin/opencode" "$HOME/harnesses/tools/bin/rg" /usr/bin/bwrap /usr/bin/socat; do
  [ -e "$f" ] || { echo "FATAL: $f is missing" | tee "$OUT/FATAL.txt"; exit 2; }
done
{
  echo "started: $(date '+%F %T %Z')"; echo "host: $(uname -srm)"; echo "serve: $(cat "$B/bin/serve-cuda-i3.rev" 2>/dev/null)"
  echo "driver: $(nvidia-smi --query-gpu=driver_version,name --format=csv,noheader)"; echo "load: $(uptime)"; free -m | sed -n 1,3p
  for h in claude codex opencode; do echo "$h: $("$HOME/harnesses/node_modules/.bin/$h" --version 2>&1 | head -1)"; done
  echo "bwrap: $(bwrap --version)"; echo "only: $ONLY"
} > "$OUT/provenance.txt"
cd "$B/matrix-pinned" || exit 2
python3 -u matrix.py "$PLAN" "$OUT" --only "$ONLY" --deadline-minutes "$DEADLINE" 2>&1 | tee "$OUT/job.log"
rc=${PIPESTATUS[0]}
# Assert the outputs exist: a job that refused to run must not read as a success.
[ -s "$OUT/records.jsonl" ] || { echo "FAILED: no records were written (matrix.py exit $rc)" | tee -a "$OUT/job.log"; exit 1; }
python3 "$B/matrix-pinned/gate_i1.py" "$OUT/records.jsonl" > "$OUT/gate-i1.txt" 2>&1
echo "finished: $(date '+%F %T %Z') (matrix.py exit $rc)" | tee -a "$OUT/job.log"
exit "$rc"
