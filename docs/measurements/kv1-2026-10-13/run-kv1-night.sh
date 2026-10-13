#!/bin/bash
# KV1's night job on one box (docs/measurements/kv1-2026-10-13.md): G5b, a lone request's decode rate and time to first
# token, this branch against main, and on nobara G2b first (a windowed real model on the fast prefill kernels). Runs with
# nobody watching: REV and BASE are commits named when it is queued, both binaries are built here from detached
# worktrees, and this script, kv1_ttft.py and grade_kv1_speed.py are the copies beside it. night.py holds the timing
# lock; bench_peer.py inherits it.
#
#   python3 scripts/night.py add kv1-speed-<backend> --est <min> --by "<who>, KV1" --doc docs/tasks/task-kv-architecture-2026-10.md \
#     -- env REV=<branch head> BASE=<main at the branch point> bash ~/goinfer-bench/kv1-2026-10/run-kv1-night.sh
#   DRY=1 ...   stops after the builds and the provenance block.
#
# Estimates. Mac: about 115 minutes (decode 24 cells a model at about 45 s on the 1.5B and 75 s on the 7B; time to first
# token 24 cells a model at about 35 s and 130 s). nobara: about 85 minutes (G2b 10; decode 16 cells a model at about 40 s
# and 60 s; time to first token 24 cells a model at about 25 s and 60 s). Each harness prints a line per cell.
set -uo pipefail
REV=${REV:?REV is required: the commit under test}
BASE=${BASE:?BASE is required: main at the branch point}
HERE=$(cd "$(dirname "$0")" && pwd)
B=${B:-$HOME/goinfer-bench/kv1-2026-10}
OUT=${OUT:-$HOME/goinfer-logs/kv1-2026-10-13/night-$(date +%F)}
M15=$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf
M7=$HOME/models/qwen2.5-7b-instruct-q4_k_m.gguf
command -v go >/dev/null 2>&1 || export PATH=/usr/local/go/bin:/opt/homebrew/bin:$PATH
case "$(uname -s)" in
  Darwin) BE=metal; SRC=$HOME/tmcode/goinfer; DECODE_QUADS=6; KEEP_AWAKE="caffeinate -i" ;;
  Linux)  BE=cuda;  SRC=$HOME/mycode/goinfer; DECODE_QUADS=4; KEEP_AWAKE="" ;;
  *) echo "FATAL: unknown system"; exit 2 ;;
esac
for f in "$M15" "$M7" "$HERE/kv1_ttft.py" "$HERE/grade_kv1_speed.py"; do [ -e "$f" ] || { echo "FATAL: $f is missing"; exit 2; }; done
mkdir -p "$B/bin" "$OUT/serve-logs" || exit 2
cd "$SRC" || exit 2
git fetch -q origin || echo "warning: git fetch failed; using what is here"
build() { # build <name> <commit>: a detached worktree and a serve binary for this box's backend
  local name=$1 rev wt=$B/wt-$1
  rev=$(git rev-parse --verify "$2^{commit}") || { echo "FATAL: $2 is not a commit here"; exit 2; }
  [ -d "$wt" ] && git worktree remove --force "$wt"
  git worktree add -q --detach "$wt" "$rev" || { echo "FATAL: cannot check out $rev"; exit 2; }
  if [ "$BE" = metal ]; then
    (cd "$wt" && go work init . ./gpu ./metal && CGO_ENABLED=0 go build -o "$B/bin/serve-$name" ./metal/cmd/serve) || { echo "FATAL: the $name build failed"; exit 2; }
  else
    printf 'go 1.27.0\n\ntoolchain go1.27.2\n\nuse (\n\t.\n\t./cuda\n\t./gpu\n\t%s\n)\n' "$HOME/mycode/aikit/aikit" > "$wt/go.work"
    (cd "$wt/cuda" && CGO_ENABLED=0 go build -tags cuda -o "$B/bin/serve-$name" ./cmd/serve) || { echo "FATAL: the $name build failed"; exit 2; }
  fi
  echo "$rev" > "$B/bin/serve-$name.rev"
}
build new "$REV"
build old "$BASE"
NEW=$B/bin/serve-new; OLD=$B/bin/serve-old
{
  echo "started: $(date '+%F %T %Z')"; echo "host: $(uname -srm)"; echo "backend: $BE"
  echo "new: $(cat "$NEW.rev")"; echo "old (main at the branch point): $(cat "$OLD.rev")"
  [ "$BE" = cuda ] && { echo "driver: $(nvidia-smi --query-gpu=driver_version,name --format=csv,noheader)"; echo "aikit (both arms): $(git -C "$HOME/mycode/aikit/aikit" describe --tags)"; }
  echo "load: $(uptime)"
} | tee "$OUT/provenance.txt"
[ "${DRY:-0}" = 1 ] && { echo "DRY=1: stopping after the builds and the provenance block"; exit 0; }
rc=0
if [ "$BE" = cuda ]; then
  echo "== G2b: started $(date '+%T'), about 10 minutes"
  (cd "$B/wt-new/cuda" && GOINFER_HEAVY_TESTS=1 go test -tags 'cuda goinfer_testhooks' . -run 'TestWindowedKV_offsetsFastClass_mistral7B' -count=1 -v -timeout 30m) > "$OUT/g2b-mistral7b.log" 2>&1
  echo "== G2b: exit $? at $(date '+%T')"; grep -E 'READING|^--- |^ok|^FAIL' "$OUT/g2b-mistral7b.log" | cut -c1-240
  grep -q -- '^--- PASS: TestWindowedKV_offsetsFastClass_mistral7B' "$OUT/g2b-mistral7b.log" || { echo "!! G2b did not pass (a skip is not a pass)"; rc=1; }
fi
echo "== decode: started $(date '+%T')"
BEU=$(echo "$BE" | tr a-z A-Z)
(cd "$B/wt-new" && env BENCH_ENGINES=goinfer,goinfer_old BENCH_BACKENDS=$BE BENCH_MODELS=1.5B,7B BENCH_DEPTHS=none BENCH_ABBA=$DECODE_QUADS \
  BENCH_REQUIRE_BACKEND=1 BENCH_SERVE_LOG_DIR="$OUT/serve-logs" BENCH_LOAD_TIMEOUT=420 "GOINFER_SERVE_$BEU=$NEW" "GOINFER_SERVE_${BEU}_OLD=$OLD" \
  $KEEP_AWAKE python3 -u scripts/bench_peer.py "$OUT/decode-$BE.json") 2>&1 | tee "$OUT/decode-$BE.log"
[ "${PIPESTATUS[0]}" = 0 ] && [ -s "$OUT/decode-$BE.json" ] || { echo "!! the decode pass failed or wrote no results (bench_peer exits 0 when it refuses a busy box)"; rc=1; }
for m in "1.5B $M15" "7B $M7"; do
  set -- $m
  echo "== time to first token, $1: started $(date '+%T')"
  $KEEP_AWAKE python3 -u "$HERE/kv1_ttft.py" "$OUT/ttft-$BE-$1.json" --backend "$BE" --new "$NEW" --old "$OLD" --model "$2" --name "$1" --blocks 12 --log-dir "$OUT/serve-logs" 2>&1 | tee -a "$OUT/ttft-$BE.log"
  [ "${PIPESTATUS[0]}" = 0 ] || rc=1
done
python3 "$HERE/grade_kv1_speed.py" "$OUT" 2>&1 | tee "$OUT/verdict.txt"
[ "${PIPESTATUS[0]}" = 0 ] || rc=1
echo "finished: $(date '+%F %T %Z') rc=$rc" | tee -a "$OUT/provenance.txt"
cd "$SRC" && git worktree remove --force "$B/wt-new" 2>/dev/null; git worktree remove --force "$B/wt-old" 2>/dev/null
exit $rc
