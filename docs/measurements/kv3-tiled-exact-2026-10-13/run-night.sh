#!/bin/bash
# KV3 tiled exact attention: the night jobs of docs/measurements/kv3-tiled-exact-2026-10-13.md (G4, G5, G6), each from PINNED artifacts
# made at queue time (a worktree at the branch's commit and two serve binaries), so the tree may move before tonight. night.py holds
# the timing lock.
#
#   JOB=identity   G4: batched against per-token past the bound on Gemma-3-4B and Gemma-4-E4B (go test; est 20 min)
#   JOB=speed      G5: time to first token at 12,400 and 20,000 tokens, batched against per-token, both models (est 45 min)
#   JOB=ttft       G6: K = 512 and 3,900 on the 1.5B and Gemma-3-4B, this branch against kv3-cuda-2026-10, three pairs (est 15 min)
#   JOB=suite      the whole ./cuda/ suite on this branch, with the ignored fixtures linked into the pinned worktree (est 60 min)
#
#   queued as:  env JOB=<job> bash <this script>          (B and OUT may be set; the KV3_TILED_* variables pass through to the job)
set -u
export PATH=$PATH:/usr/local/go/bin
B=${B:-$HOME/goinfer-bench/kv3-tiled-2026-10}
JOB=${JOB:?name the job}
W=$B/wt-pinned
OUT=${OUT:-$B/night-$(date +%F)-$JOB}
mkdir -p "$OUT" || exit 2
for f in "$W/go.work" "$B/bin/serve-cuda-tiled" "$B/bin/serve-cuda-tiled.rev" "$B/bin/serve-cuda-base" "$B/bin/serve-cuda-base.rev" \
  "$HOME/models/gemma-3-4b-it-Q4_K_M.gguf" "$HOME/models/gemma-4-e4b-gguf/gemma-4-E4B_q4_0-it.gguf" "$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"; do
  [ -e "$f" ] || { echo "FATAL: $f is missing" | tee "$OUT/FATAL.txt"; exit 2; }
done
{
  echo "started: $(date '+%F %T %Z')"; echo "job: $JOB"; echo "host: $(uname -srm)"; nvidia-smi --query-gpu=driver_version,name --format=csv,noheader
  echo "tiled: $(cat "$B/bin/serve-cuda-tiled.rev")"; echo "base: $(cat "$B/bin/serve-cuda-base.rev")"; echo "worktree: $(git -C "$W" rev-parse HEAD)"; echo "load: $(uptime)"
  free -m | sed -n 1,3p
} > "$OUT/provenance.txt"
HERE=$W/docs/measurements/kv3-tiled-exact-2026-10-13
rc=0
case $JOB in
  identity)
    (cd "$W" && GOINFER_HEAVY_TESTS=1 KV3_TILED_REAL=1 go test -count=1 -timeout 60m -tags 'cuda goinfer_testhooks' \
      -run '^TestKV3Tiled_batchedEqualsPerToken_real$' -v ./cuda/) 2>&1 | tee "$OUT/identity-real.log"; rc=${PIPESTATUS[0]} ;;
  speed)
    python3 -u "$HERE/ttft_tiled.py" "$B/bin/serve-cuda-tiled" "$OUT/speed.json" 2>&1 | tee "$OUT/speed.log"; rc=${PIPESTATUS[0]} ;;
  ttft)
    python3 -u "$HERE/ttft_ab_models.py" "$B/bin/serve-cuda-base" "$B/bin/serve-cuda-tiled" "$OUT/ttft-ab.json" 2>&1 | tee "$OUT/ttft-ab.log"; rc=${PIPESTATUS[0]} ;;
  suite)
    # KV3_TILED_SUITE_RUN narrows it (the day check of this job's plumbing); unset, every test of the package runs.
    (cd "$W" && GOINFER_HEAVY_TESTS=1 go test -count=1 -timeout 120m -tags 'cuda goinfer_testhooks' ${KV3_TILED_SUITE_RUN:+-run "$KV3_TILED_SUITE_RUN"} -v ./cuda/) 2>&1 | tee "$OUT/cuda-suite.log"; rc=${PIPESTATUS[0]}
    { echo "PASS $(grep -c '^--- PASS' "$OUT/cuda-suite.log")"; echo "FAIL $(grep -c '^--- FAIL' "$OUT/cuda-suite.log")"; echo "SKIP $(grep -c '^--- SKIP' "$OUT/cuda-suite.log")"; grep '^--- SKIP\|^--- FAIL' "$OUT/cuda-suite.log"; } > "$OUT/cuda-suite-counts.txt" ;;
  *) echo "unknown JOB $JOB"; exit 2 ;;
esac
# Assert outputs exist: a job that refused to run, or a test that skipped, must not read as a success.
case $JOB in
  identity)
    for sub in gemma-3-4b gemma-4-e4b; do
      grep -q "^    --- PASS: TestKV3Tiled_batchedEqualsPerToken_real/$sub " "$OUT/identity-real.log" || rc=1
    done
    [ "$(grep -c '\[kv3-tiled\] G4 ' "$OUT/identity-real.log")" -ge 3 ] || rc=1 ;;
  speed) [ -s "$OUT/speed.json" ] && grep -q "^SUMMARY" "$OUT/speed.log" || rc=1 ;;
  ttft) [ -s "$OUT/ttft-ab.json" ] && grep -q "^SUMMARY" "$OUT/ttft-ab.log" || rc=1 ;;
  # a narrowed run must still have run something; the whole suite must have run the row kernels' own gate
  suite) grep -q "^ok" "$OUT/cuda-suite.log" && ! grep -q "^--- FAIL\|^FAIL" "$OUT/cuda-suite.log" && grep -q "^--- PASS" "$OUT/cuda-suite.log" \
           && { [ -n "${KV3_TILED_SUITE_RUN:-}" ] || grep -q "^--- PASS: TestPrefillSplitKV_rowsEqualAttnBatched" "$OUT/cuda-suite.log"; } || rc=1 ;;
esac
echo "finished: $(date '+%F %T %Z') rc=$rc" | tee -a "$OUT/provenance.txt"
exit "$rc"
