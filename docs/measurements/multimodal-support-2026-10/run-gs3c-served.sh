#!/usr/bin/env bash
# G-S3c's served comparison (docs/tasks/task-multimodal-support-2026-10.md, S3): one image request per checkpoint
# (testdata/glm_ocr/table.png, "What does this image show? Answer briefly.", 32 greedy tokens, top-3 log-probabilities),
# through ONE serve binary, once per arm. Every arm keeps the vision tower on the CPU (-vision-device cpu), so only the
# decoder's backend differs. The first arm is compared against each later one: identical reply text, or the first
# differing token and the near-tie read (the reference arm's p(other token) >= half its own top token's p).
#
# Usage: run-gs3c-served.sh <serve binary> <out dir> <arm>[,<arm>...] <model dir>...
#   an arm is a --backend value, optionally :<-vision-device> (default cpu), e.g. metal:auto puts the tower on Metal too
#   (G-S3b: metal:cpu,metal:auto); a repeated arm runs again as a control (its files get a numeric suffix).
# Example (nobara): run-gs3c-served.sh ~/goinfer-bench/s3/serve-cuda ~/goinfer-logs/s3c-cuda cpu,cuda,cpu \
#                     ~/models/qwen25vl-3b-instruct ~/models/gemma-3-4b-it
# Run from the repo root (it reads testdata/). Checkpoints come from ~/models, never the archive.
set -euo pipefail
BIN=$1 OUT=$2 ARMS=$3; shift 3
[ -x "$BIN" ] || { echo "no serve binary at $BIN" >&2; exit 2; }
[ -f testdata/glm_ocr/table.png ] || { echo "run from the repo root" >&2; exit 2; }
mkdir -p "$OUT"
PORT=${GS3C_PORT:-18454}
for dir in "$@"; do
  case "$dir" in /Volumes/*|/srv/models*) echo "$dir is the archive (CLAUDE.md)" >&2; exit 2;; esac
done
IFS=, read -r -a arms <<<"$ARMS"
for dir in "$@"; do
  fam=$(basename "$dir")
  labels=()
  for arm in "${arms[@]}"; do
    be=${arm%%:*} vd=cpu
    [[ $arm == *:* ]] && vd=${arm#*:}
    base=$be
    [ "$vd" = cpu ] || base=$be-tower$vd
    lab=$base n=2
    while [[ " ${labels[*]-} " == *" $lab "* ]]; do lab=$base$n; n=$((n+1)); done
    labels+=("$lab")
    echo "[$(date '+%H:%M:%S')] $fam: arm $lab (--backend $be -vision-device $vd)"
    "$BIN" --model "$dir" --backend "$be" -vision-device "$vd" --addr 127.0.0.1:$PORT >"$OUT/gs3c-$fam-$lab.log" 2>&1 </dev/null &
    pid=$!
    for _ in $(seq 1 600); do
      curl -s -o /dev/null -w '%{http_code}' 127.0.0.1:$PORT/v1/models 2>/dev/null | grep -q 200 && break
      kill -0 $pid 2>/dev/null || { echo "serve exited; see $OUT/gs3c-$fam-$lab.log" >&2; exit 1; }
      sleep 1
    done
    grep -E "decode path|prefill path|vision" "$OUT/gs3c-$fam-$lab.log" | cut -c1-200 || true
    python3 - "$OUT" "$fam" "$lab" "$PORT" <<'EOF'
import base64, json, sys, time, urllib.request
out, fam, lab, port = sys.argv[1:5]
img = base64.b64encode(open('testdata/glm_ocr/table.png', 'rb').read()).decode()
body = {"messages": [{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": "data:image/png;base64," + img}},
    {"type": "text", "text": "What does this image show? Answer briefly."}]}],
    "max_tokens": 32, "temperature": 0, "logprobs": True, "top_logprobs": 3}
t = time.time()
req = urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions', data=json.dumps(body).encode(),
                             headers={'Content-Type': 'application/json'})
r = json.load(urllib.request.urlopen(req, timeout=1800))
c = r['choices'][0]
open(f'{out}/gs3c-reply-{fam}-{lab}.txt', 'w').write(c['message']['content'])
json.dump((c.get('logprobs') or {}).get('content') or [], open(f'{out}/gs3c-logprobs-{fam}-{lab}.json', 'w'))
print(f'  {lab} ({time.time() - t:.1f}s): {c["message"]["content"][:150]!r}')
EOF
    kill $pid; wait $pid 2>/dev/null || true; sleep 2
  done
  python3 - "$OUT" "$fam" "${labels[@]}" <<'EOF'
import json, math, sys
out, fam, ref, *others = sys.argv[1:]
rd = json.load(open(f'{out}/gs3c-logprobs-{fam}-{ref}.json'))
for o in others:
    od = json.load(open(f'{out}/gs3c-logprobs-{fam}-{o}.json'))
    if open(f'{out}/gs3c-reply-{fam}-{ref}.txt').read() == open(f'{out}/gs3c-reply-{fam}-{o}.txt').read():
        print(f'{fam}: {o} against {ref}: IDENTICAL replies'); continue
    i = 0
    while i < min(len(rd), len(od)) and rd[i]['token'] == od[i]['token']:
        i += 1
    if i >= min(len(rd), len(od)):
        print(f'{fam}: {o} against {ref}: replies differ in length only'); continue
    top = rd[i]['top_logprobs']; ptop = math.exp(top[0]['logprob'])
    po = [math.exp(c['logprob']) for c in top if c['token'] == od[i]['token']]
    print(f'{fam}: {o} against {ref}: first differing generated token {i}: {ref} {rd[i]["token"]!r}, {o} {od[i]["token"]!r}')
    print(f'  {ref} top-3: ' + ', '.join(f'{c["token"]!r} {math.exp(c["logprob"]):.3f}' for c in top))
    print(f'  near-tie (p(other) >= half p(top) = {ptop / 2:.3f}): {bool(po and po[0] >= ptop / 2)}')
EOF
done
