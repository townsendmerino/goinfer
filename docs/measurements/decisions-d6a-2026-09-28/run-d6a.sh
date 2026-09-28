#!/bin/bash
# D6a (docs/tasks/task-constrained-confidence.md, pre-registered before any graded run): Route A on Qwen3.5-9B through
# the production CLI — `goinfer-chat decisions-calibrate` on calib-sample.jsonl, then `goinfer-chat decide` (raw, no
# calibration) on eval-sample.jsonl; analyze.py applies each arm's fitted temperature and the decision rule.
#   A: Qwen3.5-9B, chat-v1 (graded)    B: Qwen3.5-9B, bare-v1 (control: reproduces the authors' B0?)
#   C: qwen2.5-coder-1.5b-instruct, chat-v1 (reported: what the any-model path gives on a small model)
# The default -quant, an explicit -ctx 4096 (every prompt is well under 2,000 tokens, and an unpinned context
# auto-pins far above what a GPU resident takes), models from ~/models. The samples come from select.py over the
# pinned corpus. BIN is a goinfer-chat built with the backend's module (cuda/cmd/chat, metal/cmd/chat) and named by
# its commit; BACKEND is the --backend it runs. Each run's stderr must show the RESIDENT decode path — a CPU
# fallback is not what this measures on a GPU box, and takes days.
set -u
REV=${REV:?}
BACKEND=${BACKEND:?}
B=${B:-$HOME/goinfer-bench/decisions-d6a}
S=$B/samples
BIN=${BIN:?}
M9=$HOME/models/Qwen3.5-9B-Q4_K_M.gguf
M15=$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf
ts() { date '+%H:%M:%S'; }
for m in "$M9" "$M15"; do
  case "$m" in /srv/models/*|/Volumes/*) echo "model $m is on the archive, not the bench set"; exit 1;; esac
  [ -f "$m" ] || { echo "no model at $m"; exit 1; }
done
[ -x "$BIN" ] || { echo "no binary $BIN"; exit 1; }
arm() {
  name=$1; model=$2; tmpl=$3
  echo "$(ts) == arm $name: $(basename "$model") $tmpl — calibrate"
  "$BIN" decisions-calibrate --model "$model" --backend "$BACKEND" --ctx 4096 --template "$tmpl" -o "$B/cal-$name.json" "$S/calib-sample.jsonl" 2> "$B/cal-$name.stderr"
  echo "$(ts) == arm $name calibrate exit $?; $(grep -m1 -E 'declined|resident' "$B/cal-$name.stderr" | cut -c1-160)"
  "$BIN" decide --model "$model" --backend "$BACKEND" --ctx 4096 --template "$tmpl" -o "$B/eval-$name.jsonl" "$S/eval-sample.jsonl" 2> "$B/eval-$name.stderr"
  echo "$(ts) == arm $name eval exit $?"
}
echo "$(ts) == D6a start; rev $REV; backend $BACKEND; bin $BIN; $(uname -srm)"
arm A "$M9" chat-v1
arm B "$M9" bare-v1
arm C "$M15" chat-v1
echo "$(ts) == DONE"
