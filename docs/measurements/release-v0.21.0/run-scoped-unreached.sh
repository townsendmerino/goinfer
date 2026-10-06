#!/usr/bin/env bash
# RELEASING.md §C1 for v0.21.0, the scoped re-run of the four real-checkpoint gates the full sweep never reached.
# The full sweep (run-parity-sweep.sh, 2026-10-05 20:05-22:42 PDT, REV 7b50947a) failed on its 120m realckpt-cell timeout, not on numerics:
# 1788 tests had finished, 0 FAIL lines, and TestQwen3MoeReal_oracle was in flight when the timeout panic fired. v0.20.0's same cell took ~1h46, and this
# run's archive loads (35B, Laguna 63 GB, Llama 4 37 GB) were slower. Its log: ~/goinfer-logs/parity-sweep-v021-2026-10-05/parity-sweep.log.
#
# The four gates it reported DID NOT RUN, all read from ~/models (no archive override needed):
#   TestQwen3MoeReal_oracle   (qwen3moe-30b-a3b-bf16; passed in v0.20.0's scoped run)
#   TestSmolLM3_3bReal_gate, TestSpark25Real_gate   (small; confirmed at 2732e75b / 650c1b5f)
#   TestQwen3NextReal_oracle  (expected SKIP: neverConfirmed, needs ~59.3 GB against this box's ~43.4 GB fit guard; "SKIP - asset missing, but in
#                              neverConfirmed (NOT a blocker)" in v0.20.0. It is listed so the verdict line has nothing unaccounted for.)
#
# Pass rule, written before it runs: the gate's own verdict line "ALL REQUIRED GATES GREEN" for this scoped checkset (it will print "THIS IS A SCOPED
# RE-RUN, NOT A RELEASE-READY VERDICT"; the owner accepted that form for v0.20.0, and the full sweep's other gates stand), TestQwen3MoeReal_oracle passes
# (not skips), SmolLM3 and Spark25 pass or are NOT-FOUND-skips named in the log, and Qwen3Next is the only permitted skip. Any FAIL, or a timeout,
# is a blocker: stop and report, no tag. About 30 min (Qwen3MoE load+oracle ~15-20 min, the rest minutes), TIMEOUT=90m so a hang fails loudly.
#
#   python3 scripts/timing_lock.py run --label release-v021-scoped-unreached -- bash docs/measurements/release-v0.21.0/run-scoped-unreached.sh
set -euo pipefail
REV=${REV:-7b50947a}
SRC=$HOME/mycode/goinfer
BASE=${BASE:-$HOME/goinfer-bench/release-v0.21.0}
WT=$BASE/scoped-wt
LOG=${LOG:-$HOME/goinfer-logs/parity-scoped-unreached-v021-$(date +%F)}
export PATH=/usr/local/go/bin:$PATH
export TIMEOUT=${TIMEOUT:-90m}
GATES=(TestQwen3MoeReal_oracle TestQwen3NextReal_oracle TestSmolLM3_3bReal_gate TestSpark25Real_gate)
RE="^($(IFS='|'; echo "${GATES[*]}"))\$"
for p in "$HOME/models/qwen3moe-30b-a3b-bf16" "$HOME/models/smollm3-3b" "$HOME/models/spark25-1.7b"; do
  [ -e "$p" ] || { echo "FATAL: $p is missing; its gate would skip"; exit 2; }
done
mkdir -p "$BASE" "$LOG"
cd "$SRC"
git fetch -q origin
if [ -d "$WT" ]; then git worktree remove --force "$WT"; fi
git worktree add -q --detach "$WT" "$REV"
n=0
for d in testdata decoder/testdata; do
  while IFS= read -r p; do
    p=${p%/}
    [ -e "$WT/$p" ] && continue
    if [ -d "$SRC/$p" ]; then
      mkdir -p "$WT/$p"
      for c in "$SRC/$p"/* "$SRC/$p"/.[!.]*; do [ -e "$c" ] && ln -s "$c" "$WT/$p/"; done
    else
      mkdir -p "$(dirname "$WT/$p")"
      ln -s "$SRC/$p" "$WT/$p"
    fi
    n=$((n + 1))
  done < <(git ls-files --others --ignored --exclude-standard --directory "$d")
done

cd "$WT"
{
  echo "rev:        $(git rev-parse HEAD) ($(grep 'townsendmerino/aikit v' go.mod | head -1 | xargs))"
  echo "started:    $(date '+%F %T %Z')"
  echo "fixtures:   $n gitignored entries symlinked from $SRC"
  echo "tree:       $(git status --porcelain | wc -l) entries in git status (0 = clean)"
  echo "go:         $(go version)"
  echo "timeout:    $TIMEOUT"
  echo "gates:      ${#GATES[@]}: $RE"
} | tee "$LOG/provenance.txt"
[ "${DRY:-0}" = 1 ] && { echo "DRY=1: stopping before the sweep"; exit 0; }

GOWORK=off GATE_RUN="$RE" EMIT_MANIFEST=1 go run ./cmd/gate parity 2>&1 | tee "$LOG/scoped.log" || true
cp testdata/parity_manifest.json "$LOG/parity_manifest.after.json"
echo "finished:   $(date '+%F %T %Z')" | tee -a "$LOG/provenance.txt"
