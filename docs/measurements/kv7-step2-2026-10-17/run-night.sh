#!/bin/bash
# KV7 step 2's night rerun (README.md): the e2e run's Claude Code configuration with request bodies and token ids kept, against a serve with the boundary
# snapshot, then the offline checks: the extractor's table, and every restored prefix against the ids an earlier request wrote. night.py holds the timing lock.
#
#   queued as:  B=$HOME/goinfer-bench/kv7-step2-2026-10 bash <this script>
#   DRY_RUN=1 checks everything and prints the command without running it (by day, into an empty OUT).
set -u
B=${B:?the pinned directory}
OUT=${OUT:-$B/night-out}
PLAN=$B/matrix-pinned/plan-kv7s2.json
mkdir -p "$OUT" || exit 2
for f in "$B/bin/serve-cuda-kv7s2" "$B/matrix-pinned/matrix.py" "$B/matrix-pinned/sandbox.py" "$B/matrix-pinned/e2e_metrics.py" "$PLAN" "$B/wt/go.work" \
         "$HOME/harnesses/node_modules/.bin/claude" "$HOME/harnesses/tools/bin/rg" /usr/bin/bwrap /usr/bin/socat; do
  [ -e "$f" ] || { echo "FATAL: $f is missing" | tee "$OUT/FATAL.txt"; exit 2; }
done
[ "$(sha256sum "$B/bin/serve-cuda-kv7s2" | cut -d' ' -f1)" = "$(cat "$B/bin/serve-cuda-kv7s2.sha256")" ] || { echo "FATAL: the pinned binary does not match its sha256" | tee "$OUT/FATAL.txt"; exit 2; }
{
  echo "started: $(date '+%F %T %Z')"; echo "host: $(uname -srm)"; echo "serve: $(cat "$B/bin/serve-cuda-kv7s2.rev")"
  echo "driver: $(nvidia-smi --query-gpu=driver_version,name --format=csv,noheader)"; echo "load: $(uptime)"; free -m | sed -n 1,3p
  echo "gpu processes: $(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | tr '\n' ';')"
  echo "claude: $("$HOME/harnesses/node_modules/.bin/claude" --version 2>&1 | head -1)"; echo "bwrap: $(bwrap --version)"
} >> "$OUT/provenance.txt"
cd "$B/matrix-pinned" || exit 2
CMD=(python3 -u matrix.py "$PLAN" "$OUT" --only kv7s2-claude-code --deadline-minutes "${DEADLINE_MINUTES:-45}")
if [ "${DRY_RUN:-0}" = 1 ]; then echo "DRY RUN: ${CMD[*]}" | tee -a "$OUT/job.log"; exit 0; fi
"${CMD[@]}" 2>&1 | tee -a "$OUT/job.log"
rc=${PIPESTATUS[0]}
[ -s "$OUT/records.jsonl" ] || { echo "FAILED: no records were written (matrix.py exit $rc)" | tee -a "$OUT/job.log"; exit 1; }
{
  echo "# KV7 step 2 night rerun: extractor output ($(date '+%F %T %Z'))"
  python3 -I e2e_metrics.py "$OUT" --label "this run (serve $(cut -c1-8 "$B/bin/serve-cuda-kv7s2.rev"))"
  echo; echo "## Restored prefixes against the ids an earlier request wrote"; echo '```'
  (cd "$B/wt" && SIDE_OUT="$OUT" SIDE_CONFIG=kv7s2-claude-code go test -count=1 -timeout 20m -run '^TestSideRequests_snapshotPrefixes$' -v ./internal/serveapp/ 2>&1 | grep -E '^\s+side_requests|^(---|ok|FAIL)')
  echo '```'
  echo; echo "## Report-only: how much of a session start is shared past the boundary (owner's question, 2026-10-17)"
  echo "Across this run's session starts (a sandbox with no CLAUDE.md: the leading reminder is gitStatus) and the five recorded real sessions of side-req-2026-10 (p1 to p5, with CLAUDE.md)."
  echo '```'
  S=$HOME/goinfer-bench/side-req-2026-10
  (cd "$B/wt" && SIDE_BODIES="$OUT/bodies-kv7s2-claude-code,$S/p1/req/00002.req.json.gz,$S/p2/req/00002.req.json.gz,$S/p3/req/00002.req.json.gz,$S/p4/req/00002.req.json.gz,$S/p5/req/00002.req.json.gz" \
    go test -count=1 -timeout 20m -run '^TestSideRequests_sessionStartShare$' -v ./internal/serveapp/ 2>&1 | grep -E 'side_requests|^(---|ok|FAIL)')
  echo '```'
} > "$OUT/kv7s2-metrics.md" 2>&1
echo "finished: $(date '+%F %T %Z'); figures in $OUT/kv7s2-metrics.md" | tee -a "$OUT/job.log"
exit 0
