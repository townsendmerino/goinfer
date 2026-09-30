#!/usr/bin/env bash
# Thinking-on-the-wire client matrix, the three checkpoints the 0.8B daytime run does not cover.
# Night-queue job: nobody watching, no prompts. Queue it with:
#
#   python3 scripts/night.py add think-matrix --est 90 --by "vscode-claude, think-prompt task" \
#       --doc docs/tasks/task-qwen35-think-prompt-2026-09.md \
#       -- bash docs/measurements/think-matrix-2026-09-30/run-matrix.sh
#
# Each leg starts a serve binary PRE-BUILT at a pinned revision (the tree may move before tonight), runs
# scripts/think_matrix.py against it (copied here for the same reason), and stops the server by PID. The matrix asserts the
# invariant on every cell (docs/tasks/task-qwen35-think-prompt-2026-09.md), so a leg's exit status is its verdict; this
# script exits 1 if any leg failed, 2 if every cell that ran passed but some path was NOT EXERCISED (the model never finished
# thinking, say — that is not a pass), 0 only when all three legs proved everything. All three legs always run.
#
# Legs (the prompt deltas are what each template writes; they are pinned against HF in chat/reasoning_test.go):
#   qwen3.5-9b      default ON, open prompt     as-is P; off +4 tokens; on +2
#   qwen3-1.7b      default ON, nothing written as-is P; off +4 tokens; on +0
#   gemma-4-E2B     default OFF, closed scaffold as-is P; off +0 tokens; on +3
# No speed is measured here; int4 is serve's default and is enough for plumbing. One copy of each model is in ~/models.
set -uo pipefail
D=$HOME/goinfer-bench/think-matrix-2026-09-30
BIN=$D/goinfer-serve
MATRIX=$D/think_matrix.py
OUT=$D/results
mkdir -p "$OUT"
[ -x "$BIN" ] && [ -f "$MATRIX" ] || {
  echo "MISSING $BIN or $MATRIX. Pre-build them at the revision under test:"
  echo "  mkdir -p $D && go build -o $BIN ./cmd/serve && cp scripts/think_matrix.py $MATRIX && git rev-parse HEAD > $D/REV"
  exit 2
}
echo "revision under test: $(cat "$D/REV" 2>/dev/null || echo unknown)"
echo "serve binary: $("$BIN" -version 2>&1 | head -1)"
t0=$(date +%s)
phase() { echo "[$(date +%H:%M:%S)] ($(( $(date +%s) - t0 ))s elapsed) $*"; }
rc=0
leg() { # name model-arg port on-delta off-delta tags max maxthink
  local name=$1 model=$2 port=$3 on=$4 off=$5 tags=$6 mx=$7 mt=$8
  phase "leg $name: starting server"
  "$BIN" -model "$model" -addr 127.0.0.1:"$port" >"$OUT/$name.serve.log" 2>&1 </dev/null &
  local pid=$!
  echo "$pid" >"$OUT/$name.pid"
  local up=0
  for _ in $(seq 1 600); do   # up to 20 min: a 9B's first load transcodes
    if curl -s "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1; then up=1; break; fi
    kill -0 "$pid" 2>/dev/null || break
    sleep 2
  done
  if [ "$up" != 1 ]; then
    phase "leg $name: server did not come up"; tail -5 "$OUT/$name.serve.log"
    kill "$pid" 2>/dev/null; rc=1; return
  fi
  grep -m1 "loaded" "$OUT/$name.serve.log" || true
  phase "leg $name: matrix"
  python3 "$MATRIX" --url "http://127.0.0.1:$port" --max "$mx" --max-think "$mt" --on-delta "$on" --off-delta "$off" \
      --tags "$tags" --json "$OUT/$name.json" 2>&1 | tee "$OUT/$name.log"
  case "${PIPESTATUS[0]}" in
    0) ;;
    2) [ "$rc" = 1 ] || rc=2; phase "leg $name: all cells that ran passed, but some path was NOT EXERCISED (see its log)" ;;
    *) rc=1; phase "leg $name: FAILED" ;;
  esac
  kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
  phase "leg $name: done"
}
leg qwen35-9b   "$HOME/models/qwen3.5-9b"                          8097  2 4 "<think>,</think>"           120 1500
leg qwen3-1.7b  "$HOME/models/qwen3-1.7b-bf16"                     8096  0 4 "<think>,</think>"           120 1500
leg gemma4-e2b  "$HOME/models/gemma-4-e2b-gguf/gemma-4-E2B_q4_0-it.gguf" 8095  3 0 "<|channel>,<channel|>" 120 1500
phase "done rc=$rc"
exit $rc
