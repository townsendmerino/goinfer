#!/usr/bin/env bash
# Night job: the book's timed "Try it" figures (docs/measurements/book-try-it-2026-10.md), with the released v0.20.0
# darwin-arm64 binaries, the 1.5B and 0.5B from ~/models (the files hf: fetches; sha256 checked 2026-10-02).
#   ch2 / ch8: `goinfer-serve check` against the 1.5B served on the CPU, three times: its "chat, streamed" row's tok/s
#              (decode) and its long-prompt row's TTFT (prefill of a 2,000-word prompt). Then the same on Metal (ch5).
#   ch9:       the drafted 1.5B (0.5B drafting, greedy, CPU) against the plain 1.5B, three alternating pairs, tok/s.
# Medians of three are what a chapter may quote. Runs under night.py, which holds the timing lock.
#   python3 scripts/night.py add book-try-it-timed --est 20 --by "Claude, book try-it endings" \
#     --doc docs/measurements/book-try-it-2026-10.md -- bash docs/measurements/book-try-it-2026-10/run-night.sh
set -uo pipefail
BIN=$HOME/goinfer-bench/book
CHAT=$BIN/goinfer-chat-darwin-arm64
SERVE=$BIN/goinfer-serve-darwin-arm64
M05=$HOME/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf
M15=$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf
OUT=$(cd "$(dirname "$0")" && pwd)/night-$(date +%F)
PORT=18183
mkdir -p "$OUT"
for f in "$CHAT" "$SERVE" "$M05" "$M15"; do [ -e "$f" ] || { echo "missing $f"; exit 1; }; done
{
  echo "date:     $(date '+%F %T %Z')"
  echo "machine:  $(sysctl -n machdep.cpu.brand_string), $(( $(sysctl -n hw.memsize) / 1073741824 )) GB, macOS $(sw_vers -productVersion)"
  echo "load:     $(uptime | sed 's/.*load averages*: //')"
  echo "therm:    $(pmset -g therm 2>/dev/null | tr '\n' ' ')"
  echo "chat:     $($CHAT --version | head -1); sha256 $(shasum -a 256 "$CHAT" | cut -c1-16)"
  echo "serve:    $($SERVE --version | head -1); sha256 $(shasum -a 256 "$SERVE" | cut -c1-16)"
} | tee "$OUT/provenance.txt"

checks() { # <backend>
  echo "== check, 1.5B on $1 — $(date '+%T')" | tee -a "$OUT/provenance.txt"
  "$SERVE" --model "$M15" --backend "$1" --addr 127.0.0.1:$PORT > "$OUT/serve-$1.log" 2>&1 &
  local pid=$!
  for _ in $(seq 1 180); do curl -sf "http://127.0.0.1:$PORT/health" > /dev/null && break; sleep 1; done
  for i in 1 2 3; do
    "$SERVE" check "http://127.0.0.1:$PORT" > "$OUT/check-$1-$i.txt" 2>&1
    echo "check $1 run $i rc=$?" | tee -a "$OUT/provenance.txt"
  done
  kill "$pid"; wait "$pid" 2>/dev/null
  sleep 10
}
checks cpu
checks metal

echo "== ch9: drafted vs plain, CPU — $(date '+%T')" | tee -a "$OUT/provenance.txt"
P="Write a Go function that reverses a string."
for i in 1 2 3; do
  echo "$P" | "$CHAT" --model "$M15" --temp 0 --max 64 --backend cpu > /dev/null 2> "$OUT/ch9-plain-$i.err"
  echo "$P" | "$CHAT" --model "$M15" --draft "$M05" --temp 0 --max 64 --backend cpu > /dev/null 2> "$OUT/ch9-draft-$i.err"
done

{
  for b in cpu metal; do for i in 1 2 3; do echo "-- check $b $i"; grep -E 'chat, streamed|long prompt|checks' "$OUT/check-$b-$i.txt"; done; done
  for i in 1 2 3; do for a in plain draft; do echo "-- ch9 $a $i: $(tail -1 "$OUT/ch9-$a-$i.err" | sed 's/\x1b\[[0-9;]*m//g')"; done; done
} | tee "$OUT/results.txt"
echo "finished: $(date '+%F %T %Z')" | tee -a "$OUT/provenance.txt"
