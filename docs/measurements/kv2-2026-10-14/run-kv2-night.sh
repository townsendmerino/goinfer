#!/bin/bash
# KV2's night jobs on one box (docs/measurements/kv2-2026-10-14.md §2). Runs with nobody watching: REV and BASE are
# commits named when it is queued; binaries and test worktrees are built here from detached worktrees of them; the
# harnesses and graders are the copies beside this script at REV. night.py holds the timing lock; children inherit it.
#
#   JOB=share  G5, the sharing gate: kv2_share.py on the 1.5B, graded; on the Mac also WebGPU, reported (2 blocks, busy).
#   JOB=speed  G6, a lone request: KV1's decode and time-to-first-token cells, graded by grade_kv2_speed.py.
#   JOB=heavy  the real-checkpoint arms of G0 to G3 and G7 (go test, untimed but memory-hungry).
#
#   python3 scripts/night.py add kv2-<job>-<backend> --est <min> --by "<who>, KV2" --doc docs/tasks/task-kv-architecture-2026-10.md \
#     -- env JOB=<job> REV=<sha> BASE=<sha> bash docs/measurements/kv2-2026-10-14/run-kv2-night.sh
#   DRY=1 ...   stops after the builds and the provenance block.
#
# Estimates on the Mac: share about 95 minutes (the graded cells about 630 s a block at the 1.5B's 613 tok/s on a 20K
# prompt, 6 blocks; WebGPU's reported cells not on record, about 30); speed about 62 (KV1's measured wall); heavy about
# 20. nobara: share about 30 (the CUDA prefill at 20K is several times the Mac's), speed about 31 (KV1's), heavy about 15.
# Each harness prints a line per cell; go test runs -v.
set -uo pipefail
JOB=${JOB:?JOB is share, speed or heavy}
REV=${REV:?REV is required: the commit under test}
BASE=${BASE:?BASE is required: main at the branch point}
B=${B:-$HOME/goinfer-bench/kv2-2026-10}
OUT=${OUT:-$HOME/goinfer-logs/kv2-2026-10-14/night-$(date +%F)-$JOB}
M15=$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf
M7=$HOME/models/qwen2.5-7b-instruct-q4_k_m.gguf
command -v go >/dev/null 2>&1 || export PATH=/usr/local/go/bin:/opt/homebrew/bin:$PATH
case "$(uname -s)" in
  Darwin) BE=metal; SRC=$HOME/tmcode/goinfer; DECODE_QUADS=6; KEEP_AWAKE="caffeinate -i" ;;
  Linux)  BE=cuda;  SRC=$HOME/mycode/goinfer; DECODE_QUADS=4; KEEP_AWAKE="" ;;
  *) echo "FATAL: unknown system"; exit 2 ;;
esac
for f in "$M15" "$M7"; do [ -e "$f" ] || { echo "FATAL: $f is missing"; exit 2; }; done
mkdir -p "$B/bin" "$OUT/serve-logs" || exit 2
cd "$SRC" || exit 2
git fetch -q origin || echo "warning: git fetch failed; using what is here"
worktree() { # worktree <name> <commit>: a detached worktree with this box's go.work
  local wt=$B/wt-$1 rev
  rev=$(git rev-parse --verify "$2^{commit}") || { echo "FATAL: $2 is not a commit here"; exit 2; }
  [ -d "$wt" ] && git worktree remove --force "$wt"
  git worktree prune
  git worktree add -q --detach "$wt" "$rev" || { echo "FATAL: cannot check out $rev"; exit 2; }
  if [ "$BE" = metal ]; then
    (cd "$wt" && go work init . ./gpu ./metal)
  else
    printf 'go 1.27.0\n\ntoolchain go1.27.2\n\nuse (\n\t.\n\t./cuda\n\t./gpu\n)\n' > "$wt/go.work"  # aikit at go.mod's version
  fi
  echo "$rev" > "$wt.rev"
}
build() { # build <name> <commit>: serve for this box's backend (and, on the Mac, for WebGPU)
  worktree "$1" "$2"
  local wt=$B/wt-$1
  if [ "$BE" = metal ]; then
    (cd "$wt" && CGO_ENABLED=0 go build -o "$B/bin/serve-$1" ./metal/cmd/serve) || { echo "FATAL: the $1 Metal build failed"; exit 2; }
    (cd "$wt/gpu" && go build -tags gpu -o "$B/bin/serve-webgpu-$1" ./cmd/serve) || { echo "FATAL: the $1 WebGPU build failed"; exit 2; }
  else
    (cd "$wt/cuda" && CGO_ENABLED=0 go build -tags cuda -o "$B/bin/serve-$1" ./cmd/serve) || { echo "FATAL: the $1 CUDA build failed"; exit 2; }
  fi
  cp "$wt.rev" "$B/bin/serve-$1.rev"
}
{
  echo "job: $JOB"; echo "started: $(date '+%F %T %Z')"; echo "host: $(uname -srm)"; echo "backend: $BE"
  echo "new (REV): $(git rev-parse "$REV")"; echo "old (BASE, main at the branch point): $(git rev-parse "$BASE")"
  [ "$BE" = cuda ] && { echo "driver: $(nvidia-smi --query-gpu=driver_version,name --format=csv,noheader)"; echo "aikit (go.mod): $(git show "$REV:go.mod" | awk '$1=="github.com/townsendmerino/aikit"{print $2}')"; }
  echo "load: $(uptime)"
} | tee "$OUT/provenance.txt"
rc=0
case "$JOB" in
share)
  build new "$REV"; build old "$BASE"
  H=$B/wt-new/docs/measurements/kv2-2026-10-14
  [ "${DRY:-0}" = 1 ] && { echo "DRY=1: stopping after the builds"; exit 0; }
  echo "== G5 graded, $BE 1.5B: started $(date '+%T')"
  $KEEP_AWAKE python3 -u "$H/kv2_share.py" "$OUT/share-$BE-1.5B.json" --backend "$BE" --new "$B/bin/serve-new" --old "$B/bin/serve-old" \
    --model "$M15" --name 1.5B --blocks 6 --log-dir "$OUT/serve-logs" 2>&1 | tee "$OUT/share-$BE.log"
  [ "${PIPESTATUS[0]}" = 0 ] || rc=1
  python3 "$H/grade_kv2_share.py" "$OUT/share-$BE-1.5B.json" --graded 2>&1 | tee "$OUT/verdict-share-$BE.txt"
  [ "${PIPESTATUS[0]}" = 0 ] || rc=1
  if [ "$BE" = metal ]; then
    echo "== G5 reported, webgpu 1.5B: started $(date '+%T')"
    $KEEP_AWAKE python3 -u "$H/kv2_share.py" "$OUT/share-webgpu-1.5B.json" --backend webgpu --new "$B/bin/serve-webgpu-new" --old "$B/bin/serve-webgpu-old" \
      --model "$M15" --name 1.5B --blocks 2 --arms main,kv2 --scripts busy --log-dir "$OUT/serve-logs" 2>&1 | tee "$OUT/share-webgpu.log"
    python3 "$H/grade_kv2_share.py" "$OUT/share-webgpu-1.5B.json" 2>&1 | tee "$OUT/report-share-webgpu.txt"
  fi
  ;;
speed)
  build new "$REV"; build old "$BASE"
  H=$B/wt-new/docs/measurements; NEW=$B/bin/serve-new; OLD=$B/bin/serve-old
  [ "${DRY:-0}" = 1 ] && { echo "DRY=1: stopping after the builds"; exit 0; }
  echo "== G6 decode: started $(date '+%T')"
  BEU=$(echo "$BE" | tr a-z A-Z)
  (cd "$B/wt-new" && env BENCH_ENGINES=goinfer,goinfer_old BENCH_BACKENDS=$BE BENCH_MODELS=1.5B,7B BENCH_DEPTHS=none BENCH_ABBA=$DECODE_QUADS \
    BENCH_REQUIRE_BACKEND=1 BENCH_SERVE_LOG_DIR="$OUT/serve-logs" BENCH_LOAD_TIMEOUT=420 "GOINFER_SERVE_$BEU=$NEW" "GOINFER_SERVE_${BEU}_OLD=$OLD" \
    $KEEP_AWAKE python3 -u scripts/bench_peer.py "$OUT/decode-$BE.json") 2>&1 | tee "$OUT/decode-$BE.log"
  [ "${PIPESTATUS[0]}" = 0 ] && [ -s "$OUT/decode-$BE.json" ] || { echo "!! the decode pass failed or wrote no results"; rc=1; }
  for m in "1.5B $M15" "7B $M7"; do
    set -- $m
    echo "== G6 time to first token, $1: started $(date '+%T')"
    $KEEP_AWAKE python3 -u "$H/kv1-2026-10-13/kv1_ttft.py" "$OUT/ttft-$BE-$1.json" --backend "$BE" --new "$NEW" --old "$OLD" --model "$2" --name "$1" \
      --blocks 12 --log-dir "$OUT/serve-logs" 2>&1 | tee -a "$OUT/ttft-$BE.log"
    [ "${PIPESTATUS[0]}" = 0 ] || rc=1
  done
  python3 "$H/kv2-2026-10-14/grade_kv2_speed.py" "$OUT" 2>&1 | tee "$OUT/verdict-speed-$BE.txt"
  [ "${PIPESTATUS[0]}" = 0 ] || rc=1
  ;;
heavy)
  worktree heavy "$REV"
  WT=$B/wt-heavy
  # the gitignored fixtures the tests read, linked from this checkout (a directory as a directory of symlinks)
  while IFS= read -r p; do
    p=${p%/}; [ -e "$WT/$p" ] && continue
    if [ -d "$SRC/$p" ]; then mkdir -p "$WT/$p"; for c in "$SRC/$p"/* "$SRC/$p"/.[!.]*; do [ -e "$c" ] && ln -s "$c" "$WT/$p/"; done
    else mkdir -p "$(dirname "$WT/$p")"; ln -s "$SRC/$p" "$WT/$p"; fi
  done < <(git ls-files --others --ignored --exclude-standard --directory -- testdata decoder/testdata)
  [ "${DRY:-0}" = 1 ] && { echo "DRY=1: stopping after the worktree"; exit 0; }
  cd "$WT" || exit 2
  export GOWORK=$WT/go.work
  if [ "$BE" = metal ]; then
    for q in int4 int8int8; do
      echo "== heavy, metal $q: started $(date '+%T')"
      env GOINFER_METAL_MC3=1 GOINFER_METAL_MC3_QUANT=$q go test -count=1 -timeout 40m -v -tags goinfer_testhooks \
        -run 'TestKV2_|Reuse|Slot|MC3|MC4|MC1|Resident' ./metal/ > "$OUT/heavy-metal-$q.log" 2>&1 || rc=1
      grep -E '^(ok|FAIL)|^--- FAIL|kv2-g' "$OUT/heavy-metal-$q.log"
    done
    echo "== heavy, metal 7B copy cost: started $(date '+%T')"
    env GOINFER_METAL_MC3=1 GOINFER_METAL_MC3_MODEL=$M7 go test -count=1 -timeout 20m -v -run 'TestKV2_copyCost' ./metal/ > "$OUT/heavy-metal-7B-copycost.log" 2>&1 || rc=1
    grep -E '^(ok|FAIL)|kv2-g7' "$OUT/heavy-metal-7B-copycost.log"
    echo "== heavy, webgpu: started $(date '+%T')"
    (cd gpu && env GOINFER_HEAVY_TESTS=1 go test -count=1 -timeout 40m -v -tags 'gpu goinfer_testhooks' \
      -run 'TestKV2_|Reuse|Slot|MC1|Resident' . > "$OUT/heavy-webgpu.log" 2>&1) || rc=1
    grep -E '^(ok|FAIL)|^--- FAIL|kv2-g' "$OUT/heavy-webgpu.log"
  fi
  ;;
*) echo "FATAL: JOB=$JOB"; exit 2 ;;
esac
echo "finished: $(date '+%F %T %Z') rc=$rc" | tee -a "$OUT/provenance.txt"
cd "$SRC" && for w in new old heavy; do git worktree remove --force "$B/wt-$w" 2>/dev/null; done
exit $rc
