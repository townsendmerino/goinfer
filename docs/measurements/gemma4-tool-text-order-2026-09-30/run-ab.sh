#!/usr/bin/env bash
# Gemma 4 canonical template: text beside a tool call, goinfer's own order (arm A) vs the template's (arm B). Night-queue job: nobody
# watching, no prompts. Queue with:
#
#   python3 scripts/night.py add gemma4-text-order --est 120 --by "vscode-claude, Gemma 4 renderer gaps" \
#       --doc docs/tasks/task-qwen35-think-prompt-2026-09.md -- bash docs/measurements/gemma4-tool-text-order-2026-09-30/run-ab.sh
#
# Pre-registration: PREREGISTERED.md in this directory (the rule, written before either arm ran). Both arms run the SAME pre-built binary
# (pinned revision in $D/REV) and differ only by -tool-format. Model: Gemma-4-12B-it QAT q4_0, whose embedded template is the canonical one,
# on the CPU backend (the 8 GB card cannot hold its KV). Each arm is ~45 min: 24 probe replies, each followed by a loop turn.
#
# Exit status: 0 ADOPT, 3 PARK, 4 REJECT (the verdict, mechanically), 2 VOID (an arm did not run, the load log did not say the template is
# MANAGED, or a reply set is incomplete), 1 anything else. A VOID is not a result.
set -uo pipefail
D=$HOME/goinfer-bench/gemma4-tool-text-order-2026-09-30
BIN=$D/serve; PROBE=$D/tool_format_probe.py; VERDICT=$D/ab_verdict.py
MODEL=$HOME/models/gemma-4-12b-gguf/gemma-4-12b-it-qat-q4_0.gguf
PORT=18433
# SMOKE=1: one prompt, one sample per arm — proves the mechanics (server up, managed-template check, probe, shutdown), measures nothing.
PROBE_ARGS=(--samples 3 --max-tokens 600 --loop --preamble "Let me check that for you.")
[ -n "${SMOKE:-}" ] && PROBE_ARGS=(--samples 1 --only weather --max-tokens 600 --loop --preamble "Let me check that for you.")
[ -x "$BIN" ] && [ -f "$PROBE" ] && [ -f "$VERDICT" ] && [ -f "$MODEL" ] || {
  echo "MISSING one of: $BIN $PROBE $VERDICT $MODEL"
  echo "  mkdir -p $D && go build -o $BIN ./cmd/serve && cp scripts/tool_format_probe.py $PROBE && cp docs/measurements/gemma4-tool-text-order-2026-09-30/ab_verdict.py $VERDICT && git rev-parse HEAD > $D/REV"
  exit 2
}
OUT=$D/results/$(date +%Y%m%d-%H%M%S); mkdir -p "$OUT"
echo "revision under test: $(cat "$D/REV" 2>/dev/null || echo unknown); results: $OUT"
t0=$(date +%s); phase() { echo "[$(date +%H:%M:%S)] ($(( $(date +%s) - t0 ))s) $*"; }

arm() { # name tool-format
  local name=$1 fmt=$2 pid base
  base=$(free -m | awk '/Swap/{print $3}')
  phase "arm $name (-tool-format $fmt): starting server"
  "$BIN" -model "m=$MODEL" -backend cpu -ctx 8192 -tool-format "$fmt" -addr 127.0.0.1:$PORT > "$OUT/$name.serve.log" 2>&1 &
  pid=$!
  local up=0
  for _ in $(seq 1 200); do
    kill -0 $pid 2>/dev/null || { phase "arm $name: server exited early"; tail -3 "$OUT/$name.serve.log"; return 1; }
    curl -s -o /dev/null http://127.0.0.1:$PORT/v1/models && { up=1; break; }
    sleep 3
  done
  [ $up = 1 ] || { phase "arm $name: server never came up"; kill $pid; return 1; }
  if ! grep -q "thinking: template default off" "$OUT/$name.serve.log"; then
    phase "arm $name: VOID — the load log does not say the template is managed:"; grep "chat:" "$OUT/$name.serve.log" | head -2
    kill $pid; wait $pid 2>/dev/null; return 2
  fi
  grep "chat:" "$OUT/$name.serve.log" | head -1
  phase "arm $name: probing (each reply is followed by a loop turn; a line appears per reply)"
  timeout 4500 python3 "$PROBE" http://127.0.0.1:$PORT "${PROBE_ARGS[@]}" --json "$OUT/$name.json" > "$OUT/$name.log" 2>&1 &
  local ppid=$!
  while kill -0 $ppid 2>/dev/null; do
    sleep 60
    sw=$(free -m | awk '/Swap/{print $3}'); av=$(free -m | awk '/Mem/{print $7}')
    if [ $((sw-base)) -gt 2000 ] || [ "$av" -lt 6000 ]; then phase "arm $name: STOP swap +$((sw-base)) MB, avail ${av} MB"; kill $ppid; kill $pid; return 1; fi
    phase "arm $name: $(grep -cE '^[a-z]+ +s[0-9]' "$OUT/$name.log") replies done (of 24 in a full run)"
  done
  wait $ppid; local rc=$?
  kill $pid; wait $pid 2>/dev/null
  phase "arm $name: probe exited $rc"
  [ -s "$OUT/$name.json" ] || return 1
  return 0
}

arm A hermes;   ra=$?
arm B template; rb=$?
[ $ra = 2 ] || [ $rb = 2 ] && { phase "VOID: an arm's template was not managed"; exit 2; }
[ $ra = 0 ] && [ $rb = 0 ] || { phase "an arm failed (A=$ra B=$rb)"; exit 1; }
[ -n "${SMOKE:-}" ] && { phase "SMOKE ok: both arms ran and shut down; no verdict (the reply set is deliberately incomplete)"; exit 0; }
phase "applying the pre-registered rule"
python3 "$VERDICT" "$OUT/A.json" "$OUT/B.json" | tee "$OUT/VERDICT.txt"
exit ${PIPESTATUS[0]}
