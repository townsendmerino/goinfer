#!/usr/bin/env bash
# The book's "Try it" commands that print deterministic figures (docs/book/, docs/measurements/book-try-it-2026-10.md):
# token counts, sizes, fit plans, next-token probabilities under greedy decoding, draft acceptance under greedy, the
# compiled-in backends, and the server's own check. No timing is read from this script, so it may run by day. Run with
# the released v0.20.0 darwin-arm64 binaries, against checkpoints in ~/models.
#
#   bash docs/measurements/book-try-it-2026-10/run-day.sh
set -uo pipefail
BIN=$HOME/goinfer-bench/book
CHAT=$BIN/goinfer-chat-darwin-arm64
SERVE=$BIN/goinfer-serve-darwin-arm64
M05=$HOME/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf
M15=$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf
M26=$HOME/models/gemma4-26b-int4-v14st.metal.giw
OUT=$(cd "$(dirname "$0")" && pwd)/day-2026-10-02
PORT=18181
mkdir -p "$OUT"
for f in "$CHAT" "$SERVE" "$M05" "$M15" "$M26"; do [ -e "$f" ] || { echo "missing $f"; exit 1; }; done
{
  echo "date:     $(date '+%F %T %Z')"
  echo "machine:  $(sysctl -n machdep.cpu.brand_string), $(( $(sysctl -n hw.memsize) / 1073741824 )) GB, macOS $(sw_vers -productVersion)"
  echo "chat:     $($CHAT --version | head -1); sha256 $(shasum -a 256 "$CHAT" | cut -c1-16)"
  echo "serve:    $($SERVE --version | head -1); sha256 $(shasum -a 256 "$SERVE" | cut -c1-16)"
  for m in "$M05" "$M15" "$M26"; do echo "model:    $(basename "$m") sha256 $(shasum -a 256 "$m" | cut -c1-16)"; done
} | tee "$OUT/provenance.txt"

step() { echo "== $1" | tee -a "$OUT/provenance.txt"; }

step "ch10: goinfer-serve --version"
"$SERVE" --version > "$OUT/ch10-version.txt" 2>&1
step "ch4/ch5: fit on the 1.5B at int4 and int8int8"
"$CHAT" fit "$M15" > "$OUT/ch4-fit-1.5b-int4.txt" 2>&1
"$CHAT" fit "$M15" -quant int8int8 > "$OUT/ch5-fit-1.5b-int8int8.txt" 2>&1
step "ch6/ch7: fit on the 26B MoE"
"$CHAT" fit "$M26" > "$OUT/ch6-fit-26b.txt" 2>&1
step "ch9: the 1.5B drafted by the 0.5B, greedy, CPU"
echo "Write a Go function that reverses a string." | "$CHAT" --model "$M15" --draft "$M05" --temp 0 --max 64 --backend cpu \
  > "$OUT/ch9-spec.out" 2> "$OUT/ch9-spec.err"

step "serve the 0.5B on CPU for ch1, ch3 and ch11"
"$SERVE" --model "$M05" --backend cpu --addr 127.0.0.1:$PORT > "$OUT/serve.log" 2>&1 &
pid=$!
for _ in $(seq 1 120); do curl -sf "http://127.0.0.1:$PORT/health" > /dev/null && break; sleep 1; done
step "ch1: count_tokens"
curl -s "http://127.0.0.1:$PORT/v1/messages/count_tokens" -H 'content-type: application/json' \
  -d "{\"model\":\"$(basename "$M05" .gguf)\",\"messages\":[{\"role\":\"user\",\"content\":\"The quick brown fox jumps over the lazy dog.\"}]}" > "$OUT/ch1-count-tokens.json"
step "ch3: next-token probabilities, greedy"
curl -s "http://127.0.0.1:$PORT/v1/chat/completions" -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"What is the capital of France? Answer in one word."}],"temperature":0,"max_tokens":1,"logprobs":true,"top_logprobs":3}' \
  > "$OUT/ch3-logprobs.json"
step "ch11: goinfer-serve check"
"$SERVE" check -long-prompt 0 "http://127.0.0.1:$PORT" > "$OUT/ch11-check.txt" 2>&1
echo "check rc=$?" | tee -a "$OUT/provenance.txt"
kill "$pid"; wait "$pid" 2>/dev/null
echo "finished: $(date '+%F %T %Z')" | tee -a "$OUT/provenance.txt"
