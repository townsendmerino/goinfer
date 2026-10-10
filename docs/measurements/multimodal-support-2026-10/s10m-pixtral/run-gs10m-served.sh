#!/usr/bin/env bash
# G-S10m-d (docs/tasks/task-multimodal-support-2026-10.md, "S10, Ministral 3 (Pixtral)"): served Ministral 3 image requests
# through ONE serve binary, once per arm, every arm with the same load flags (--embed-int4=false on all, G-S3c's lesson):
#   one-image: glm_ocr/table.png, "What does this table show? Answer in one sentence."
#   two-image: glm_ocr/table.png then gemma3_preprocess_image.png, "Describe each image in one sentence."
# Both with an explicit system message, 48 greedy tokens, top-3 log-probabilities. The REFERENCE arm carries a leading
# "=" (exactly one); every other arm is compared with it request by request: identical reply, or the first differing
# token and the near-tie read (the reference's p(other token) >= half its own top token's p). G-S5c's driver
# (run-gs5c-served.sh) with the requests swapped.
#
# Usage: run-gs10m-served.sh <serve binary> <out dir> <arm>[,<arm>...] -- <serve model flags...>
# Example: run-gs10m-served.sh ./serve ~/goinfer-logs/gs10m =cpu,metal,cpu -- \
#            --model ~/models/ministral3-3b-bf16 --vision ~/models/ministral3-3b-bf16
# Run from the repo root. Checkpoints from ~/models, never the archive.
set -euo pipefail
BIN=$1 OUT=$2 ARMS=$3; shift 3
[ "${1:-}" = "--" ] && shift
[ -x "$BIN" ] || { echo "no serve binary at $BIN" >&2; exit 2; }
for a in "$@"; do case "$a" in /Volumes/*|/srv/models*) echo "$a is the archive (CLAUDE.md)" >&2; exit 2;; esac; done
mkdir -p "$OUT"
PORT=${GS10M_PORT:-18457}
IFS=, read -r -a marked <<<"$ARMS"
arms=() refidx=-1
for i in "${!marked[@]}"; do
  a=${marked[$i]}
  if [[ $a == =* ]]; then
    [ "$refidx" -lt 0 ] || { echo "two arms are marked as the reference (\"=\"): $ARMS" >&2; exit 2; }
    refidx=$i a=${a#=}
  fi
  arms+=("$a")
done
[ "$refidx" -ge 0 ] || { echo "no reference arm: mark exactly one arm with a leading \"=\" (e.g. =cpu,metal)" >&2; exit 2; }
[ "${#arms[@]}" -ge 2 ] || { echo "need the reference and at least one other arm: $ARMS" >&2; exit 2; }
labels=()
for be in "${arms[@]}"; do
  lab=$be n=2
  while [[ " ${labels[*]-} " == *" $lab "* ]]; do lab=$be$n; n=$((n+1)); done
  labels+=("$lab")
  echo "[$(date '+%H:%M:%S')] arm $lab (--backend $be --embed-int4=false)"
  "$BIN" "$@" --backend "$be" --embed-int4=false --addr 127.0.0.1:$PORT >"$OUT/serve-$lab.log" 2>&1 </dev/null &
  pid=$!
  for _ in $(seq 1 600); do
    curl -s -o /dev/null -w '%{http_code}' 127.0.0.1:$PORT/v1/models 2>/dev/null | grep -q 200 && break
    kill -0 $pid 2>/dev/null || { echo "serve exited; see $OUT/serve-$lab.log" >&2; exit 1; }
    sleep 1
  done
  grep -E "decode path|Pixtral|vision tower" "$OUT/serve-$lab.log" | cut -c1-200 || true
  python3 - "$OUT" "$lab" "$PORT" <<'EOF'
import base64, json, sys, time, urllib.request
out, lab, port = sys.argv[1:4]
def img(p):
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(open("testdata/" + p, "rb").read()).decode()}}
reqs = {"one-image": [img("glm_ocr/table.png"), {"type": "text", "text": "What does this table show? Answer in one sentence."}],
        "two-image": [img("glm_ocr/table.png"), img("gemma3_preprocess_image.png"), {"type": "text", "text": "Describe each image in one sentence."}]}
for name, content in reqs.items():
    body = {"messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": content}],
            "max_tokens": 48, "temperature": 0, "logprobs": True, "top_logprobs": 3}
    t = time.time()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=1800))
    c = r["choices"][0]
    open(f"{out}/reply-{name}-{lab}.txt", "w").write(c["message"]["content"])
    json.dump((c.get("logprobs") or {}).get("content") or [], open(f"{out}/logprobs-{name}-{lab}.json", "w"))
    print(f"  {lab} {name} ({time.time() - t:.1f}s, {r['usage']['prompt_tokens']} prompt tokens): {c['message']['content'][:160]!r}", flush=True)
EOF
  grep -E "vision: decoded|image prefill|resident" "$OUT/serve-$lab.log" | sed 's/^/  /' || true
  kill $pid; wait $pid 2>/dev/null || true; sleep 2
done
python3 - "$OUT" "$refidx" "${labels[@]}" <<'EOF'
import json, math, os, sys
out, refidx, *labels = sys.argv[1:]
ref = labels[int(refidx)]
others = [l for k, l in enumerate(labels) if k != int(refidx)]
print(f"reference arm {ref}")
clips = ["one-image", "two-image"]
for clip in clips:
    rd = json.load(open(f"{out}/logprobs-{clip}-{ref}.json"))
    for o in others:
        if open(f"{out}/reply-{clip}-{ref}.txt").read() == open(f"{out}/reply-{clip}-{o}.txt").read():
            print(f"{clip}: {o} against {ref}: IDENTICAL replies"); continue
        od = json.load(open(f"{out}/logprobs-{clip}-{o}.json"))
        i = 0
        while i < min(len(rd), len(od)) and rd[i]["token"] == od[i]["token"]:
            i += 1
        if i >= min(len(rd), len(od)):
            print(f"{clip}: {o} against {ref}: replies differ in length only"); continue
        top = rd[i]["top_logprobs"]; ptop = math.exp(top[0]["logprob"])
        po = [math.exp(c["logprob"]) for c in top if c["token"] == od[i]["token"]]
        print(f"{clip}: {o} against {ref}: first differing token {i}: {ref} {rd[i]['token']!r}, {o} {od[i]['token']!r}; "
              f"{ref} top-3 " + ", ".join(f"{c['token']!r} {math.exp(c['logprob']):.3f}" for c in top)
              + f"; near-tie {bool(po and po[0] >= ptop / 2)}")
EOF
