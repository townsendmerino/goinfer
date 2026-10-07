#!/usr/bin/env bash
# S9's speed record (docs/tasks/task-multimodal-support-2026-10.md, S9 on Metal): E2B time to first token with the
# E-model layer-major prefill (arm "s9") against the path before it (arm "old": GOINFER_BATCHED_PREFILL=0, so a text
# prompt prefills one token at a time on the GPU and an image turn prefills on the CPU and uploads). Two requests per
# fresh server, so no feature cache or prefix reuse helps either arm: table.png ("What does this image show? Answer
# briefly.") and a ~512-token text prompt, 8 tokens each, streamed; TTFT is the time to the first content chunk. Three
# passes, the arm order alternating, under the timing lock. A record, not a gate: the pass stays on unless this reads
# below 1.00x (the 4b precedent).
#
# Pinned: $BIN/serve-metal (rev in serve-metal.rev; built from s2-towers with main merged, for -vision-device). The
# tower runs on the CPU in both arms (-vision-device cpu), as in G-S9c, so the image TTFT difference is the prefill.
# Estimate: ~45 s a server (load, two requests), 2 arms x 3 passes, ~5 min; queued at 15.
#   python3 scripts/night.py add s9-speed --est 15 --by "Claude, S9" --doc docs/tasks/task-multimodal-support-2026-10.md -- bash docs/measurements/multimodal-support-2026-10/run-s9-speed.sh
set -uo pipefail
SRC=$(cd "$(dirname "$0")/../../.." && pwd)
BIN=${BIN:-$HOME/goinfer-bench/s9}
OUT=${1:-$HOME/goinfer-bench/s9/run-$(date +%F)}
MODEL=$HOME/models/gemma-4-e2b-gguf/gemma-4-E2B_q4_0-it.gguf
VIS=$HOME/models/gemma-4-E2B-unq
[ -x "$BIN/serve-metal" ] || { echo "FATAL: $BIN/serve-metal is missing"; exit 2; }
for p in "$MODEL" "$VIS"; do [ -e "$p" ] || { echo "FATAL: $p is missing"; exit 2; }; done
mkdir -p "$OUT"
{ echo "binary:   $BIN/serve-metal (rev $(cat "$BIN/serve-metal.rev" 2>/dev/null))"; echo "started:  $(date '+%F %T %Z')"
  sw_vers 2>/dev/null | tr '\n' ' '; echo; pmset -g batt | head -1; } | tee "$OUT/provenance.txt"
cd "$SRC" || exit 2
PORT=18457
one() { # arm pass
  local arm=$1 pass=$2 env=()
  [ "$arm" = old ] && env=(GOINFER_BATCHED_PREFILL=0)
  env ${env[@]+"${env[@]}"} "$BIN/serve-metal" --model "$MODEL" --vision "$VIS" --backend metal -vision-device cpu --addr 127.0.0.1:$PORT \
    > "$OUT/serve-$arm-$pass.log" 2>&1 </dev/null &
  local pid=$!
  for _ in $(seq 1 300); do
    curl -s -o /dev/null -w '%{http_code}' 127.0.0.1:$PORT/v1/models 2>/dev/null | grep -q 200 && break
    kill -0 $pid 2>/dev/null || { echo "serve exited ($arm $pass)"; return 1; }
    sleep 1
  done
  python3 - "$arm" "$pass" "$PORT" "$OUT" <<'EOF'
import base64, json, sys, time, urllib.request
arm, pas, port, out = sys.argv[1:5]
def ttft(content):
    body = {"messages": [{"role": "user", "content": content}], "max_tokens": 8, "temperature": 0, "stream": True}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1200) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            ch = d.get("choices") or []
            if ch and (ch[0].get("delta") or {}).get("content"):
                return time.time() - t0
    return float("nan")
img = base64.b64encode(open("testdata/glm_ocr/table.png", "rb").read()).decode()
t_img = ttft([{"type": "image_url", "image_url": {"url": "data:image/png;base64," + img}},
              {"type": "text", "text": "What does this image show? Answer briefly."}])
para = ("The quarterly report covers unit sales across five regions, with notes on supply, pricing and the effect of "
        "seasonal demand on each product line. ")
t_txt = ttft([{"type": "text", "text": para * 20 + "Summarise the report in one sentence."}])
rec = {"arm": arm, "pass": int(pas), "image_ttft_s": round(t_img, 3), "text_ttft_s": round(t_txt, 3)}
print(json.dumps(rec))
open(f"{out}/records.jsonl", "a").write(json.dumps(rec) + "\n")
EOF
  local rc=$?
  grep -E "vision: decoded|prefill path" "$OUT/serve-$arm-$pass.log" | sed 's/^/  /' | cut -c1-160
  kill $pid; wait $pid 2>/dev/null || true; sleep 3
  return $rc
}
run_all() {
  for pass in $(seq 1 "${PASSES:-3}"); do
    if [ $((pass % 2)) -eq 1 ]; then order="old s9"; else order="s9 old"; fi
    for arm in $order; do one "$arm" "$pass" || return 1; done
  done
}
if [ -z "${S9_SPEED_LOCK_HELD:-}" ]; then
  python3 "$SRC/scripts/timing_lock.py" run --label s9-speed -- env S9_SPEED_LOCK_HELD=1 bash "$0" "$OUT"
  exit $?
fi
run_all
rc=$?
python3 - "$OUT" <<'EOF'
import json, statistics, sys
recs = [json.loads(l) for l in open(sys.argv[1] + "/records.jsonl")]
for k in ["image_ttft_s", "text_ttft_s"]:
    by = {a: [r[k] for r in recs if r["arm"] == a] for a in ["old", "s9"]}
    if all(by.values()):
        mo, ms = statistics.median(by["old"]), statistics.median(by["s9"])
        print(f"{k}: old {by['old']} (median {mo:.2f}), s9 {by['s9']} (median {ms:.2f}); old/s9 {mo / ms:.2f}x")
EOF
echo "finished: $(date '+%F %T %Z') rc=$rc" | tee -a "$OUT/provenance.txt"
exit $rc
