#!/usr/bin/env bash
# G-Q3 of docs/tasks/task-metal-q4k-2026-10.md (registered before any code): one greedy chat request, the peer harness's
# depth-128 Phi-3 prompt (scripts/prompts.json "phi3-mini:128"), 64 tokens with top-3 log-probabilities, through ONE serve
# binary, once with --backend cpu (the reference) and once with --backend metal, both at -quant q4k and -ctx 512. Each
# arm's decode path is read from its serve log and must be the one it names (cpu (q4k) / metal-resident (q4k)), or the
# script fails. PASS: identical reply text, or a first differing token at a near-tie (the reference's p(other) >= half
# its own top token's p).
#
# Usage: run-gq3-served.sh <serve binary> <out dir> <model .gguf>
# Run from the repo root (it reads scripts/prompts.json). The model comes from ~/models, never the archive.
set -euo pipefail
BIN=$1 OUT=$2 MODEL=$3
[ -x "$BIN" ] || { echo "no serve binary at $BIN" >&2; exit 2; }
[ -f scripts/prompts.json ] || { echo "run from the repo root" >&2; exit 2; }
case "$MODEL" in /Volumes/*|/srv/models*) echo "$MODEL is the archive (CLAUDE.md)" >&2; exit 2;; esac
[ -f "$MODEL" ] || { echo "no model at $MODEL" >&2; exit 2; }
mkdir -p "$OUT"
PORT=${GQ3_PORT:-18455}
for arm in cpu metal; do
  want="decode path: cpu (q4k)"
  [ "$arm" = metal ] && want="decode path: metal-resident (q4k)"
  echo "[$(date '+%H:%M:%S')] arm $arm (-backend $arm -quant q4k -ctx 512)"
  "$BIN" -model "bench=$MODEL" -backend "$arm" -quant q4k -ctx 512 -addr 127.0.0.1:$PORT >"$OUT/gq3-$arm.log" 2>&1 </dev/null &
  pid=$!
  trap 'kill $pid 2>/dev/null' ERR EXIT # a failed request must not leave this arm's server running
  for _ in $(seq 1 600); do
    curl -s -o /dev/null -w '%{http_code}' 127.0.0.1:$PORT/v1/models 2>/dev/null | grep -q 200 && break
    kill -0 $pid 2>/dev/null || { echo "serve exited; see $OUT/gq3-$arm.log" >&2; exit 1; }
    sleep 1
  done
  grep -E "decode path|prefill path" "$OUT/gq3-$arm.log" | cut -c1-200 || true
  grep -qF "$want" "$OUT/gq3-$arm.log" || { echo "VOID: arm $arm did not decode on the path it names (want \"$want\")" >&2; exit 1; }
  python3 - "$OUT" "$arm" "$PORT" <<'EOF'
import json, sys, time, urllib.request
out, arm, port = sys.argv[1:4]
prompt = json.load(open('scripts/prompts.json'))['phi3-mini:128']['text']
body = {"model": "bench", "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 64, "temperature": 0, "logprobs": True, "top_logprobs": 3}
t = time.time()
req = urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions', data=json.dumps(body).encode(),
                             headers={'Content-Type': 'application/json'})
r = json.load(urllib.request.urlopen(req, timeout=1800))
c = r['choices'][0]
open(f'{out}/gq3-reply-{arm}.txt', 'w').write(c['message']['content'])
json.dump((c.get('logprobs') or {}).get('content') or [], open(f'{out}/gq3-logprobs-{arm}.json', 'w'))
print(f'  {arm} ({time.time() - t:.1f}s, {r.get("usage", {}).get("completion_tokens")} tokens): {c["message"]["content"][:200]!r}')
EOF
  trap - ERR EXIT
  kill $pid; wait $pid 2>/dev/null || true; sleep 2
done
python3 - "$OUT" <<'EOF'
import json, math, sys
out = sys.argv[1]
rd, md = (json.load(open(f'{out}/gq3-logprobs-{a}.json')) for a in ('cpu', 'metal'))
if open(f'{out}/gq3-reply-cpu.txt').read() == open(f'{out}/gq3-reply-metal.txt').read():
    print('G-Q3 PASS: IDENTICAL replies'); sys.exit(0)
i = 0
while i < min(len(rd), len(md)) and rd[i]['token'] == md[i]['token']:
    i += 1
if i >= min(len(rd), len(md)):
    print('G-Q3 FAIL: the replies differ in length only'); sys.exit(1)
top = rd[i]['top_logprobs']; ptop = math.exp(top[0]['logprob'])
po = [math.exp(c['logprob']) for c in top if c['token'] == md[i]['token']]
near = bool(po and po[0] >= ptop / 2)
print(f'first differing generated token {i}: cpu {rd[i]["token"]!r}, metal {md[i]["token"]!r}')
print('  cpu top-3: ' + ', '.join(f'{c["token"]!r} {math.exp(c["logprob"]):.3f}' for c in top))
print(f'G-Q3 {"PASS" if near else "FAIL"}: near-tie (p(other) >= half p(top) = {ptop / 2:.3f}): {near}')
sys.exit(0 if near else 1)
EOF
