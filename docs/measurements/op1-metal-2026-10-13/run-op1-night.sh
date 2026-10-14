#!/bin/bash
# OP1 on Metal, the speed guard G2-S (docs/tasks/task-forward-architecture-2026-10.md §4.6): no model got slower.
# Two arms, new (REV) against base (BASE, main before any OP1 Metal code), served, no peer arm. Runs with nobody
# watching: both serve binaries are built here from detached worktrees at commits named when the job is queued.
# night.py holds the timing lock; bench_peer.py inherits it.
#
#   python3 scripts/night.py add op1-speed-metal --est 75 --by "<who>, OP1 Metal (G2-S)" \
#     --doc docs/tasks/task-forward-architecture-2026-10.md \
#     -- env REV=<merged commit> BASE=<main before OP1's Metal code> bash docs/measurements/op1-metal-2026-10-13/run-op1-night.sh
#   DRY=1 ...   stops after the builds and the provenance block.
#
# Cells, 4 paired blocks each (BENCH_ABBA=2 is two mirrored quads, A B then B A):
#   registered: decode on the 1.5B at depth 128 and 3900; decode on Gemma 3 4B at depth 128; time to first token on
#               the 1.5B, a prompt of about 2,400 words, three newcomers a block.
#   added, reported and not gating: decode on the 0.5B at depth 128 and 3900. At depth 3900 the 0.5B takes
#               attention_fa_blk64, the one split-KV kernel step 1 routed; the 1.5B's takes the unrouted block kernel.
# Estimate: decode 4 cells x 4 blocks x 2 arms at about 45 s, about 25 min; Gemma 3 4B 8 arm-cells at about 90 s with
# its load, about 12 min, plus writing its sidecar once; time to first token 8 cells at about 60 s, about 8 min; two
# builds, about 3 min. About 50 min; queue at 75.
#
# Disk: Gemma 3 4B has no Metal sidecar on this Mac, so its first load writes one of about 2.5 GB. The script removes
# any sidecar that was not there when it started.
set -uo pipefail
REV=${REV:?REV is required: the commit under test}
BASE=${BASE:?BASE is required: main before OP1 Metal code}
HERE=$(cd "$(dirname "$0")" && pwd)
SRC=$HOME/tmcode/goinfer
B=${B:-$HOME/goinfer-bench/op1-metal-2026-10}
OUT=${OUT:-$HOME/goinfer-logs/op1-metal-2026-10-13/night-$(date +%F)}
G3DIR=$HOME/models/gemma-3-4b-it-gguf
command -v go >/dev/null 2>&1 || export PATH=/usr/local/go/bin:/opt/homebrew/bin:$PATH
[ "$(uname -s)" = Darwin ] || { echo "FATAL: this is the Metal half's script"; exit 2; }
for f in "$HOME/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf" "$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf" \
  "$G3DIR/gemma-3-4b-it-Q4_K_M.gguf" "$HERE/grade_speed.py"; do
  [ -e "$f" ] || { echo "FATAL: $f is missing"; exit 2; }
done
mkdir -p "$B/bin" "$OUT/serve-logs" || exit 2
cd "$SRC" || exit 2
git fetch -q origin || echo "warning: git fetch failed; using what is here"
build() { # build <name> <commit>: a detached worktree and a Metal serve binary
  local name=$1 rev wt=$B/wt-$1
  rev=$(git rev-parse --verify "$2^{commit}") || { echo "FATAL: $2 is not a commit here"; exit 2; }
  [ -d "$wt" ] && git worktree remove --force "$wt"
  git worktree add -q --detach "$wt" "$rev" || { echo "FATAL: cannot check out $rev"; exit 2; }
  (cd "$wt" && go work init . ./gpu ./metal && CGO_ENABLED=0 go build -o "$B/bin/serve-$name" ./metal/cmd/serve) || { echo "FATAL: the $name build failed"; exit 2; }
  echo "$rev" > "$B/bin/serve-$name.rev"
}
build new "$REV"
build old "$BASE"
NEW=$B/bin/serve-new; OLD=$B/bin/serve-old
{
  echo "started: $(date '+%F %T %Z')"; echo "host: $(uname -srm)"; echo "backend: metal"
  echo "new: $(cat "$NEW.rev")"; echo "old (main before OP1's Metal code): $(cat "$OLD.rev")"
  echo "load: $(uptime)"; echo "disk free: $(df -h "$HOME" | tail -1 | awk '{print $4}')"
} | tee "$OUT/provenance.txt"
[ "${DRY:-0}" = 1 ] && { echo "DRY=1: stopping after the builds and the provenance block"; exit 0; }
rc=0
peer() { # peer <results file> <models> <depths>: one bench_peer.py pass, new against old, from the new worktree (its prompts.json and model table)
  (cd "$B/wt-new" && env BENCH_ENGINES=goinfer,goinfer_old BENCH_BACKENDS=metal BENCH_DEPTH_BACKEND=metal BENCH_MODELS="$2" BENCH_DEPTHS="$3" BENCH_ABBA=2 \
    BENCH_REQUIRE_BACKEND=1 BENCH_SERVE_LOG_DIR="$OUT/serve-logs" BENCH_LOAD_TIMEOUT=420 GOINFER_SERVE_METAL="$NEW" GOINFER_SERVE_METAL_OLD="$OLD" \
    caffeinate -i python3 -u scripts/bench_peer.py "$1") 2>&1 | tee -a "$OUT/decode-metal.log"
  [ "${PIPESTATUS[0]}" = 0 ] && [ -s "$1" ] || { echo "!! a decode pass failed or wrote no results: $1 (bench_peer exits 0 when it refuses a busy box)"; rc=1; }
}
echo "== decode, 0.5B and 1.5B at depth 128 and 3900: started $(date '+%T')"
peer "$OUT/decode-metal-qwen.json" 0.5B,1.5B 3900
echo "== decode, Gemma 3 4B at depth 128: started $(date '+%T')"
before=$(ls "$G3DIR"/*.giw "$G3DIR"/*.giw.verified 2>/dev/null | sort)
peer "$OUT/decode-metal-gemma3.json" gemma3-4b none
for f in $(ls "$G3DIR"/*.giw "$G3DIR"/*.giw.verified 2>/dev/null | sort); do
  echo "$before" | grep -qxF "$f" || { echo "removing the sidecar this run wrote: $f"; rm -f "$f"; }
done
echo "== time to first token, 1.5B, three newcomers a cell: started $(date '+%T')"
for k in old_1 new_1 new_2 old_2 old_3 new_3 new_4 old_4; do
  case $k in old*) bin=$OLD ;; new*) bin=$NEW ;; esac
  echo "$(date '+%T') cell $k"
  (cd "$B/wt-new" && GOINFER_SERVE_CPU=$bin caffeinate -i python3 -u scripts/bench_prefill_stall.py "$OUT/ttft-metal-1.5B.json" --key "$k" --decoders 0 --newcomers 3 \
    --prompt-words 2400 --server-log "$OUT/serve-logs/ttft-servers.log") 2>&1 | tee -a "$OUT/ttft-metal.log"
  [ "${PIPESTATUS[0]}" = 0 ] || { echo "!! time-to-first-token cell $k failed"; rc=1; }
done
python3 "$HERE/grade_speed.py" "$OUT" 2>&1 | tee "$OUT/verdict.txt"
[ "${PIPESTATUS[0]}" = 0 ] || rc=1
echo "finished: $(date '+%F %T %Z') rc=$rc" | tee -a "$OUT/provenance.txt"
cd "$SRC" && git worktree remove --force "$B/wt-new" 2>/dev/null; git worktree remove --force "$B/wt-old" 2>/dev/null
exit $rc
