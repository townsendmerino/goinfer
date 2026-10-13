#!/bin/bash
# KV3, CUDA's half: the night jobs of docs/measurements/kv3-cuda-2026-10-13.md (§2.3 and §3.1), each from PINNED artifacts made at queue
# time (a worktree at the branch's commit and two serve binaries), so the tree may move before tonight. night.py holds the timing lock.
#
#   JOB=fidelity   the real-checkpoint fidelity set past the 12,160-key bound (go test, the 1.5B; est 30 min)
#   JOB=speed      Claude Code first request, new against old, three pairs, with the swap reading (est 30 min)
#   JOB=ttft       K = 512 and 3,900 on the 1.5B, new against old, three pairs (est 10 min)
#   JOB=suite      the whole ./cuda/ suite, identity (est 60 min)
#
#   queued as:  env JOB=<job> B=$HOME/goinfer-bench/kv3-2026-10 bash <this script>
set -u
export PATH=$PATH:/usr/local/go/bin
B=${B:-$HOME/goinfer-bench/kv3-2026-10}
JOB=${JOB:?name the job}
W=$B/wt-pinned
OUT=${OUT:-$B/night-$(date +%F)-$JOB}
mkdir -p "$OUT" || exit 2
for f in "$W/go.work" "$B/bin/serve-cuda-new" "$B/bin/serve-cuda-old" "$HOME/goinfer-bench/kv0-2026-10/bodies/qwen3-4b-28k-cc-claude-code-chain-r1/0001.json" "$HOME/models/qwen3-4b/config.json"; do
  [ -e "$f" ] || { echo "FATAL: $f is missing" | tee "$OUT/FATAL.txt"; exit 2; }
done
{
  echo "started: $(date '+%F %T %Z')"; echo "job: $JOB"; echo "host: $(uname -srm)"; nvidia-smi --query-gpu=driver_version,name --format=csv,noheader
  echo "new: $(cat "$B/bin/serve-cuda-new.rev")"; echo "old: $(cat "$B/bin/serve-cuda-old.rev")"; echo "load: $(uptime)"
  free -m | sed -n 1,3p
} > "$OUT/provenance.txt"
HERE=$W/docs/measurements/kv3-cuda-2026-10-13
rc=0
case $JOB in
  fidelity)
    # the registered real set: 12,500, 16,384 and 20,000 tokens, two windows each; the per-token reference is about 100 tokens a second
    (cd "$W" && GOINFER_HEAVY_TESTS=1 KV3_LENGTHS=12500,16384,20000 KV3_WINDOWS=2 go test -count=1 -timeout 90m -tags 'cuda goinfer_testhooks' \
      -run '^TestKV3_batchedPrefillPastTheSpanBound_real1p5B$' -v ./cuda/) 2>&1 | tee "$OUT/fidelity-real.log"; rc=${PIPESTATUS[0]} ;;
  speed)
    python3 -u "$HERE/cc_first_request_ab.py" "$B/bin/serve-cuda-old" "$B/bin/serve-cuda-new" "$HOME/models/qwen3-4b" \
      "$HOME/goinfer-bench/kv0-2026-10/bodies/qwen3-4b-28k-cc-claude-code-chain-r1/0001.json" "$OUT/cc-first-request.json" --ctx 28672 2>&1 | tee "$OUT/cc-first-request.log"; rc=${PIPESTATUS[0]} ;;
  ttft)
    python3 -u "$HERE/ttft_ab.py" "$B/bin/serve-cuda-old" "$B/bin/serve-cuda-new" "$OUT/ttft-ab.json" 2>&1 | tee "$OUT/ttft-ab.log"; rc=${PIPESTATUS[0]} ;;
  suite)
    (cd "$W" && GOINFER_HEAVY_TESTS=1 go test -count=1 -timeout 120m -tags 'cuda goinfer_testhooks' -v ./cuda/) 2>&1 | tee "$OUT/cuda-suite.log"; rc=${PIPESTATUS[0]} ;;
  *) echo "unknown JOB $JOB"; exit 2 ;;
esac
# Assert outputs exist: a job that refused to run must not read as a success.
case $JOB in
  fidelity) grep -q "^--- PASS: TestKV3_batchedPrefillPastTheSpanBound_real1p5B" "$OUT/fidelity-real.log" || rc=1 ;;
  speed) [ -s "$OUT/cc-first-request.json" ] && grep -q SUMMARY "$OUT/cc-first-request.log" || rc=1 ;;
  ttft) grep -q SUMMARY "$OUT/ttft-ab.log" || rc=1 ;;
  suite) grep -q "^ok" "$OUT/cuda-suite.log" || rc=1 ;;
esac
echo "finished: $(date '+%F %T %Z') rc=$rc" | tee -a "$OUT/provenance.txt"
exit "$rc"
