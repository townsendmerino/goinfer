#!/bin/bash
# The end-to-end agent run as one unattended night job (README.md): Claude Code, Codex and opencode against Mellum2.1 on the 8 GB card,
# each configured by `serve connect <agent> -write` (matrix/run_cell.py does that, file unedited), tasks read, search, edit, one run each,
# from the binary and harness pin-e2e.sh pinned. One matrix.py invocation per agent, so an agent that cannot start costs seconds and is
# named in $OUT/job.log without stopping the others. night.py holds the timing lock for the whole job.
#
#   queued as:  B=$HOME/goinfer-bench/e2e-agents-2026-10-16 bash <this script>
#   DRY_RUN=1 checks everything and prints the matrix commands without running them (by day, into an empty OUT).
set -u
B=${B:?the pinned directory}
OUT=${OUT:-$B/e2e-out}
PLAN=$B/matrix-pinned/plan-e2e.json
DEADLINE=${DEADLINE_MINUTES:-70}
mkdir -p "$OUT" || exit 2
for f in "$B/bin/serve-cuda-e2e" "$B/matrix-pinned/matrix.py" "$B/matrix-pinned/sandbox.py" "$B/matrix-pinned/e2e_metrics.py" "$PLAN" \
         "$HOME/harnesses/node_modules/.bin/claude" "$HOME/harnesses/node_modules/.bin/codex" "$HOME/harnesses/node_modules/.bin/opencode" \
         "$HOME/harnesses/tools/bin/rg" /usr/bin/bwrap /usr/bin/socat; do
  [ -e "$f" ] || { echo "FATAL: $f is missing" | tee "$OUT/FATAL.txt"; exit 2; }
done
[ "$(sha256sum "$B/bin/serve-cuda-e2e" | cut -d' ' -f1)" = "$(cat "$B/bin/serve-cuda-e2e.sha256")" ] ||
  { echo "FATAL: the pinned binary does not match its recorded sha256" | tee "$OUT/FATAL.txt"; exit 2; }
{
  echo "started: $(date '+%F %T %Z')"; echo "host: $(uname -srm)"; echo "serve: $(cat "$B/bin/serve-cuda-e2e.rev")"
  echo "driver: $(nvidia-smi --query-gpu=driver_version,name --format=csv,noheader)"; echo "load: $(uptime)"; free -m | sed -n 1,3p
  echo "gpu processes: $(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | tr '\n' ';')"
  for h in claude codex opencode; do echo "$h: $("$HOME/harnesses/node_modules/.bin/$h" --version 2>&1 | head -1)"; done
  echo "bwrap: $(bwrap --version)"
} >> "$OUT/provenance.txt"
cd "$B/matrix-pinned" || exit 2
T0=$(date +%s)
for cfg in e2e-opencode e2e-codex e2e-claude-code; do
  left=$(( DEADLINE - ( $(date +%s) - T0 ) / 60 ))
  if [ "$left" -le 0 ]; then echo "deadline: $cfg NOT RUN (the $DEADLINE-minute budget is spent)" | tee -a "$OUT/job.log"; continue; fi
  CMD=(python3 -u matrix.py "$PLAN" "$OUT" --only "$cfg" --deadline-minutes "$left")
  if [ "${DRY_RUN:-0}" = 1 ]; then echo "DRY RUN: ${CMD[*]}" | tee -a "$OUT/job.log"; continue; fi
  "${CMD[@]}" 2>&1 | tee -a "$OUT/job.log"
  rc=${PIPESTATUS[0]}
  n=$(grep -c "\"config\": \"$cfg\"" "$OUT/records.jsonl" 2>/dev/null || true)
  echo "$cfg: matrix.py exit $rc, ${n:-0} records ($(date '+%F %T %Z'))" | tee -a "$OUT/job.log"
  if [ "${n:-0}" = 0 ]; then echo "AGENT FOR $cfg COULD NOT START: no record written; see $OUT/logs and the matrix output above" | tee -a "$OUT/job.log"; fi
done
[ "${DRY_RUN:-0}" = 1 ] && exit 0
# Assert the outputs exist: a job that refused to run must not read as a success.
[ -s "$OUT/records.jsonl" ] || { echo "FAILED: no records were written" | tee -a "$OUT/job.log"; exit 1; }
{
  echo "# End-to-end agent run: extractor output ($(date '+%F %T %Z'))"
  python3 -I e2e_metrics.py "$OUT" --label "this run (serve $(cat "$B/bin/serve-cuda-e2e.rev" | cut -c1-8))"
  echo; echo "## Baselines (cross-session: different builds, different days; read as context, not as a paired difference)"
  [ -d "$HOME/goinfer-bench/ids-cc-2026-10/out" ] && python3 -I e2e_metrics.py "$HOME/goinfer-bench/ids-cc-2026-10/out" --label "opencode, the day run of 2026-10-15 23:05 (serve 2a0d11bb, one rep)"
  I3="$HOME/goinfer-bench/harness-2026-10-13/i3-out-2026-10-13-mellum-16k+mellum-32k-cc"
  [ -d "$I3" ] && python3 -I e2e_metrics.py "$I3" --label "I3 job B, 2026-10-13 (before carried ids; five tasks, so rows hold read,search,edit,write,fix)"
  [ -d "$HOME/goinfer-bench/ids3-2026-10/ids3-out" ] && python3 -I e2e_metrics.py "$HOME/goinfer-bench/ids3-2026-10/ids3-out" --config 'ids3-on' --label "I3 after-run, Codex with carried ids on (2026-10-15; three reps of five tasks)"
} > "$OUT/e2e-metrics.md" 2>&1
echo "finished: $(date '+%F %T %Z'); figures in $OUT/e2e-metrics.md" | tee -a "$OUT/job.log"
exit 0
