#!/usr/bin/env bash
# G-S10j step 1's reference: the served first-token top-5 probabilities on table.png with the served prompt, for the off and the on serve binary (docs/tasks/task-multimodal-support-2026-10.md, "G-S10j").
# Usage: s10j_served_reference.sh <out.json>   (run from the repo root; binaries in ~/goinfer-bench/s6)
set -euo pipefail
OUT=$1; PORT=18462; TMP=$(mktemp -d)
for arm in dsoff dson; do
  ~/goinfer-bench/s6/serve-cuda-$arm --model ~/models/qwen3-vl-2b-instruct --backend cuda -vision-device auto --addr 127.0.0.1:$PORT > "$TMP/$arm.log" 2>&1 < /dev/null &
  pid=$!
  for _ in $(seq 1 120); do curl -s -o /dev/null -w '%{http_code}' 127.0.0.1:$PORT/v1/models 2>/dev/null | grep -q 200 && break; sleep 1; done
  python3 - "$arm" "$PORT" "$TMP/$arm.json" <<'PY'
import base64, json, math, sys, urllib.request
arm, port, out = sys.argv[1:4]
img = base64.b64encode(open('testdata/glm_ocr/table.png', 'rb').read()).decode()
body = {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + img}}, {"type": "text", "text": "What does this image show? Answer briefly."}]}],
        "max_tokens": 1, "temperature": 0, "logprobs": True, "top_logprobs": 5}
runs = []
for _ in range(2):  # the server's repeat must be identical
    r = json.load(urllib.request.urlopen(urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions', data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'}), timeout=600))
    runs.append({'usage': r['usage']['prompt_tokens'], 'top5': {c['token']: math.exp(c['logprob']) for c in r['choices'][0]['logprobs']['content'][0]['top_logprobs']}})
json.dump({'arm': arm, 'identical_repeat': runs[0] == runs[1], 'run': runs[0]}, open(out, 'w'))
PY
  kill $pid; wait $pid 2>/dev/null || true; sleep 15
done
python3 - "$TMP" "$OUT" <<'PY'
import json, sys
d = {a: json.load(open(f'{sys.argv[1]}/{a}.json')) for a in ('dsoff', 'dson')}
json.dump(d, open(sys.argv[2], 'w'), indent=1)
for a, v in d.items(): print(a, 'repeat identical:', v['identical_repeat'], 'prompt tokens', v['run']['usage'], {k: round(p, 4) for k, p in v['run']['top5'].items()})
PY
