#!/usr/bin/env bash
# S9 on CUDA part A, G3p (docs/tasks/task-multimodal-support-2026-10.md, "S9 on CUDA, part A"): the real Gemma 4 E2B served on CUDA, a ~300-token and a ~2,700-token
# text prompt, batched prefill (the NEW binary) against sequential prefill (the PRE-CHANGE binary), greedy, replies compared under the registered near-tie rule.
# The same run also records TTFT (time to the first streamed content chunk), EXPLORATORY by day and never quoted as a result: the speed record is the night job.
#
# Usage: run-s9a-served.sh <new serve binary> <old serve binary> <out dir>
# Each arm is one fresh server (cold KV). Per prompt size, per arm, TWO cold requests, each with its own nonce at the very start of the prompt so neither reuses the
# other's prefix: (1) streamed, max 32 tokens -> TTFT and the reply; (2) non-streamed with top-3 logprobs -> the reply and the logprobs the near-tie rule needs.
# Arms run in the order given, then once more reversed is NOT done here (G3p is a correctness gate; ordering matters only for the night timing record).
set -uo pipefail
NEW=${1:?new serve binary}; OLD=${2:?old serve binary}; OUT=${3:?out dir}
MODEL=${MODEL:-$HOME/models/gemma-4-e2b-gguf/gemma-4-E2B_q4_0-it.gguf}
PORT=${PORT:-18791}
for p in "$NEW" "$OLD" "$MODEL"; do [ -e "$p" ] || { echo "FATAL: $p is missing" >&2; exit 2; }; done
case "$MODEL" in /Volumes/*|/srv/models/*) echo "FATAL: $MODEL is on the archive (CLAUDE.md)" >&2; exit 2;; esac
mkdir -p "$OUT"
{ echo "new: $NEW ($(cat "$(dirname "$NEW")/rev" 2>/dev/null || echo rev-unknown))"; echo "old: $OLD ($(cat "$(dirname "$OLD")/rev" 2>/dev/null || echo rev-unknown))"
  echo "model: $MODEL"; echo "started: $(date '+%F %T %Z')"; echo "gpu: $(nvidia-smi --query-gpu=name,driver_version,memory.used --format=csv,noheader)"; echo "load: $(cat /proc/loadavg)"; } | tee "$OUT/provenance.txt"
pid=
trap '[ -n "$pid" ] && kill "$pid" 2>/dev/null; wait 2>/dev/null' EXIT
for arm in new old; do
  bin=$NEW; [ "$arm" = old ] && bin=$OLD
  "$bin" --model "$MODEL" --backend cuda --addr 127.0.0.1:$PORT >"$OUT/serve-$arm.log" 2>&1 </dev/null &
  pid=$!
  for _ in $(seq 1 240); do
    curl -s -o /dev/null -w '%{http_code}' 127.0.0.1:$PORT/v1/models 2>/dev/null | grep -q 200 && break
    kill -0 $pid 2>/dev/null || { echo "serve ($arm) exited; see $OUT/serve-$arm.log" >&2; exit 1; }
    sleep 1
  done
  grep -E "prefill path|decode path" "$OUT/serve-$arm.log" | head -3 | sed "s/^/[$arm] /"
  python3 - "$OUT" "$arm" "$PORT" <<'EOF'
import json, sys, time, urllib.request
out, arm, port = sys.argv[1:4]
SENT = ["The harbour authority published its quarterly figures on %s, and the numbers surprised the committee.",
        "Cargo volumes rose in %s while the average berth time fell by a tenth of a day.",
        "A survey of %s pilots found that most preferred the northern channel in poor visibility.",
        "The report for %s lists seventeen vessels, four of which were refitted in the old yard.",
        "Maintenance of the %s crane was deferred twice because the replacement part arrived late."]
MONTHS = ["January","February","March","April","May","June","July","August","September","October","November","December"]
def prompt(n_sent, nonce):
    body = " ".join(SENT[i % len(SENT)] % MONTHS[(i * 7) % 12] for i in range(n_sent))
    return f"[{nonce}] {body}\n\nIn one sentence, what is this text about?"
def post(msg, stream, logprobs):
    body = {"messages": [{"role": "user", "content": msg}], "max_tokens": 32, "temperature": 0, "stream": stream}
    if logprobs: body.update(logprobs=True, top_logprobs=3)
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=600)
res = {}
for size, n_sent in (("300", 14), ("2700", 126)):
    # 1: streamed, TTFT = request sent -> first chunk with non-empty content
    t0 = time.time(); ttft = None; text = ""
    with post(prompt(n_sent, f"s{size}a"), True, False) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line.endswith("[DONE]"): continue
            d = json.loads(line[5:])
            c = (d["choices"][0].get("delta") or {}).get("content") or ""
            if c and ttft is None: ttft = time.time() - t0
            text += c
    # 2: non-streamed with logprobs
    t1 = time.time()
    with post(prompt(n_sent, f"s{size}b"), False, True) as r:
        d = json.loads(r.read())
    ch = d["choices"][0]
    res[size] = {"ttft_s": ttft, "stream_text": text, "text": ch["message"]["content"], "logprobs": (ch.get("logprobs") or {}).get("content") or [],
                 "total_s_nonstream": time.time() - t1, "prompt_tokens": d.get("usage", {}).get("prompt_tokens")}
    print(f"[{arm}] {size}: prompt_tokens={res[size]['prompt_tokens']} ttft={ttft:.2f}s (exploratory) reply={ch['message']['content'][:70]!r}")
json.dump(res, open(f"{out}/g3p-{arm}.json", "w"))
EOF
  rc=$?
  kill $pid 2>/dev/null; wait $pid 2>/dev/null; pid=
  [ $rc -eq 0 ] || { echo "!! the $arm arm's client failed (rc=$rc)" >&2; exit 1; }
done
python3 - "$OUT" <<'EOF'
import json, math, sys
out = sys.argv[1]
new, old = json.load(open(f"{out}/g3p-new.json")), json.load(open(f"{out}/g3p-old.json"))
ok = True
for size in new:
    n, o = new[size], old[size]
    print(f"== {size}: prompt_tokens new={n['prompt_tokens']} old={o['prompt_tokens']}  TTFT(exploratory) new={n['ttft_s']:.2f}s old={o['ttft_s']:.2f}s")
    if n["prompt_tokens"] != o["prompt_tokens"]: ok = False; print("   !! prompt token counts differ")
    print(f"   streamed reply identical: {n['stream_text'] == o['stream_text']}")
    if n["text"] == o["text"] and n["logprobs"] == o["logprobs"]:
        print("   non-streamed reply and logprobs: IDENTICAL"); continue
    a, b = n["logprobs"], o["logprobs"]
    k = next((i for i in range(min(len(a), len(b))) if a[i]["token"] != b[i]["token"]), None)
    if k is None and a == b: continue
    if k is None:
        print("   replies identical token for token; logprob values differ (reported, not a failure under the rule)"); continue
    ptop = math.exp(b[k]["top_logprobs"][0]["logprob"]); pother = math.exp(next((t["logprob"] for t in b[k]["top_logprobs"] if t["token"] == a[k]["token"]), -1e9))
    tie = pother >= ptop / 2
    print(f"   first differing generated token {k}: old {b[k]['token']!r} (p={ptop:.3f}) new {a[k]['token']!r} (p={pother:.3f}); near-tie (p(other) >= half p(top)): {tie}")
    ok = ok and tie
print("G3p:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
EOF
