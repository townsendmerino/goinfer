#!/bin/bash
# G-ids-3, the carried-ids live gate, as unattended night jobs (ids-carried-2026-10-15.md, "Tier 2"; registered in
# id-roundtrip-step0-2026-10-15.md §(c) and Amendment 1). One job per repetition: Codex x Mellum2.1 (mellum-16k-codex-f), the
# five tasks, each run under the feature off (--carry-ids=false, the do-nothing arm) and on, interleaved by task, from the
# binary and harness pin-ids3.sh pinned. All three jobs write to one OUT; matrix.py resumes, so a repeated job adds nothing.
#
#   queued as:  REP=1 B=$HOME/goinfer-bench/ids3-2026-10 bash <this script>
#               REP=2 B=...  bash <this script>      (skips itself when rep 1 met the registered early stop)
#               REP=3 B=...  bash <this script>      (likewise)
#   DRY_RUN=1 checks everything and prints the matrix command without running it (by day, into an empty OUT).
#
# night.py holds the timing lock for the whole job.
set -u
B=${B:?the pinned directory}
REP=${REP:?1, 2 or 3}
OUT=${OUT:-$B/ids3-out}
PLAN=$B/matrix-pinned/plan-ids3.json
DEADLINE=${DEADLINE_MINUTES:-170}
mkdir -p "$OUT" || exit 2
for f in "$B/bin/serve-cuda-ids3" "$B/matrix-pinned/matrix.py" "$B/matrix-pinned/sandbox.py" "$B/matrix-pinned/gids3_check.py" "$PLAN" \
         "$HOME/harnesses/node_modules/.bin/codex" "$HOME/harnesses/tools/bin/rg" /usr/bin/bwrap /usr/bin/socat; do
  [ -e "$f" ] || { echo "FATAL: $f is missing" | tee "$OUT/FATAL-r$REP.txt"; exit 2; }
done
[ "$(sha256sum "$B/bin/serve-cuda-ids3" | cut -d' ' -f1)" = "$(cat "$B/bin/serve-cuda-ids3.sha256")" ] ||
  { echo "FATAL: the pinned binary does not match its recorded sha256" | tee "$OUT/FATAL-r$REP.txt"; exit 2; }
ONLY=$(python3 -c 'import json,sys; print(",".join(c["name"] for c in json.load(open(sys.argv[1]))["configs"] if c["label"]["rep"] == int(sys.argv[2])))' "$PLAN" "$REP")
[ -n "$ONLY" ] || { echo "FATAL: no configuration for rep $REP" | tee "$OUT/FATAL-r$REP.txt"; exit 2; }
{
  echo "rep $REP started: $(date '+%F %T %Z')"; echo "host: $(uname -srm)"; echo "serve: $(cat "$B/bin/serve-cuda-ids3.rev")"
  echo "driver: $(nvidia-smi --query-gpu=driver_version,name --format=csv,noheader)"; echo "load: $(uptime)"; free -m | sed -n 1,3p
  echo "gpu processes: $(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | tr '\n' ';')"
  echo "codex: $("$HOME/harnesses/node_modules/.bin/codex" --version 2>&1 | head -1)"; echo "bwrap: $(bwrap --version)"; echo "only: $ONLY"
} >> "$OUT/provenance.txt"
if [ "$REP" != 1 ]; then
  python3 -I "$B/matrix-pinned/gids3_check.py" "$OUT" --stop-check > "$OUT/stop-check-r$REP.txt" 2>&1
  if [ $? = 10 ]; then
    echo "rep $REP NOT RUN: the registered early stop holds ($(tail -1 "$OUT/stop-check-r$REP.txt"))" | tee -a "$OUT/job.log"
    exit 0
  fi
fi
cd "$B/matrix-pinned" || exit 2
CMD=(python3 -u matrix.py "$PLAN" "$OUT" --only "$ONLY" --deadline-minutes "$DEADLINE")
if [ "${DRY_RUN:-0}" = 1 ]; then
  echo "DRY RUN: ${CMD[*]}" | tee -a "$OUT/job.log"
  python3 -I "$B/matrix-pinned/gids3_check.py" "$OUT" --stop-check
  exit 0
fi
"${CMD[@]}" 2>&1 | tee -a "$OUT/job.log"
rc=${PIPESTATUS[0]}
# Assert the outputs exist: a job that refused to run must not read as a success.
n=$(grep -c "\"config\": \"ids3-[a-z]*-[a-z]*-r$REP\"" "$OUT/records.jsonl" 2>/dev/null || true)
[ "${n:-0}" -gt 0 ] || { echo "FAILED: no rep-$REP records were written (matrix.py exit $rc)" | tee -a "$OUT/job.log"; exit 1; }
python3 -I "$B/matrix-pinned/gids3_check.py" "$OUT" --stop-check > "$OUT/count-after-r$REP.txt" 2>&1
echo "rep $REP finished: $(date '+%F %T %Z') (matrix.py exit $rc, $n records)" | tee -a "$OUT/job.log"
exit "$rc"
