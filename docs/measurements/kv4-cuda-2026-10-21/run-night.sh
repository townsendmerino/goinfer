#!/bin/bash
# KV4's G1d heavy suites (docs/measurements/kv4-cuda-2026-10-21.md §2, G1d): the tagged CUDA suite with GOINFER_HEAVY_TESTS=1 in KV3's two
# partitions, from a worktree pinned at queue time (the tree may move before the job runs). night.py holds the timing lock.
#
#   JOB=suite-a   the main tier for tests named Test[A-L] with GOINFER_ISOLATE_DEFER=1, then each needsFreshProcess test as its own
#                 process, then the drainsDevice group (est 75 min)
#   JOB=suite-b   the main tier for tests named Test[M-Z] (est 85 min)
#
#   queued as:  env JOB=<job> W=<pinned worktree> bash <this script>
set -u
export PATH=$PATH:/usr/local/go/bin
JOB=${JOB:?name the job}
W=${W:?name the pinned worktree}
OUT=${OUT:-$HOME/goinfer-bench/kv4-night/night-$(date +%F)-$JOB}
mkdir -p "$OUT" || exit 2
[ -e "$W/go.work" ] || { echo "FATAL: $W/go.work is missing" | tee "$OUT/FATAL.txt"; exit 2; }
{
  echo "started: $(date '+%F %T %Z')"; echo "job: $JOB"; echo "host: $(uname -srm)"; nvidia-smi --query-gpu=driver_version,name --format=csv,noheader
  echo "tree: $W at $(git -C "$W" rev-parse HEAD)"; echo "load: $(uptime)"
  free -m | sed -n 1,3p
} > "$OUT/provenance.txt"
rc=0
RUN='^Test[A-L]'; [ "$JOB" = suite-b ] && RUN='^Test[M-Z]'
case $JOB in suite-a|suite-b) ;; *) echo "unknown JOB $JOB"; exit 2 ;; esac
(cd "$W" && GOINFER_HEAVY_TESTS=1 GOINFER_ISOLATE_DEFER=1 go test -count=1 -timeout 120m -tags 'cuda goinfer_testhooks' -run "$RUN" -v ./cuda/) 2>&1 \
  | tee "$OUT/cuda-main.log"; rc=${PIPESTATUS[0]}
if [ "$JOB" = suite-a ]; then
  # marked <call>: every top-level test in cuda/*_test.go whose body calls it (cmd/gate's markedTests)
  marked() { awk -v m="$1" '/^func Test[A-Za-z0-9_]*\(/ { n = $2; sub(/\(.*/, "", n); next } index($0, m) && n != "" { print n; n = "" }' "$W"/cuda/*_test.go | sort -u; }
  iso=$(marked 'needsFreshProcess(t,'); drain=$(marked 'drainsDevice(t,')
  echo "fresh-process tests: $(echo $iso)" | tee -a "$OUT/provenance.txt"; echo "drain group: $(echo $drain)" | tee -a "$OUT/provenance.txt"
  for name in $iso; do
    (cd "$W" && GOINFER_HEAVY_TESTS=1 go test -count=1 -timeout 30m -tags 'cuda goinfer_testhooks' -run "^$name\$" -v ./cuda/) 2>&1 \
      | tee -a "$OUT/cuda-iso.log"; [ "${PIPESTATUS[0]}" = 0 ] || rc=1
  done
  (cd "$W" && GOINFER_HEAVY_TESTS=1 GOINFER_DRAIN_GROUP=1 go test -count=1 -timeout 20m -tags 'cuda goinfer_testhooks' \
    -run "^($(echo $drain | tr ' ' '|'))\$" -v ./cuda/) 2>&1 | tee "$OUT/cuda-drain.log"; [ "${PIPESTATUS[0]}" = 0 ] || rc=1
  # Reconcile both ways: each marked test ran alone exactly once, and nothing marked fell into neither half.
  ni=$(echo $iso | wc -w); nd=$(echo $drain | wc -w)
  ri=$(grep -c "ISOLATED-RUN" "$OUT/cuda-iso.log"); rd=$(grep -c "DRAIN-GROUP-RUN" "$OUT/cuda-drain.log")
  echo "reconcile: fresh-process marked=$ni ran-alone=$ri; drain marked=$nd ran=$rd" | tee -a "$OUT/provenance.txt"
  [ "$ni" -gt 0 ] && [ "$ri" = "$ni" ] && [ "$nd" -gt 0 ] && [ "$rd" = "$nd" ] || rc=1
fi
# Assert outputs exist: a job that refused to run must not read as a success.
grep -q "^ok" "$OUT/cuda-main.log" || rc=1
echo "finished: $(date '+%F %T %Z') rc=$rc" | tee -a "$OUT/provenance.txt"
exit "$rc"
