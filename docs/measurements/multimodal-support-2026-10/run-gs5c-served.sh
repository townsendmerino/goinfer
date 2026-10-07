#!/usr/bin/env bash
# G-S5c (docs/tasks/task-multimodal-support-2026-10.md, S5): one "Transcribe this audio." request per clip
# (testdata/embeddinggemma2-audio/{short,mid,long}.wav, 32 greedy tokens, top-3 log-probabilities) through ONE serve
# binary, once per arm, every arm with the same load flags (G-S3c's lesson: --embed-int4=false on all). The first arm is
# the reference; each later arm is compared clip by clip: identical reply, or the first differing token and the
# near-tie read (the reference's p(other token) >= half its own top token's p).
#
# Usage: run-gs5c-served.sh <serve binary> <out dir> <arm>[,<arm>...] -- <serve model flags...>
#   an arm is a --backend value; a repeated arm runs again as a control (its files get a numeric suffix).
# Example: run-gs5c-served.sh ./serve ~/goinfer-logs/gs5c cpu,metal,cpu -- \
#            --model ~/models/gemma-4-e2b-gguf/gemma-4-E2B_q4_0-it.gguf --vision ~/models/gemma-4-E2B-unq
# Run from the repo root. Checkpoints from ~/models, never the archive.
set -euo pipefail
BIN=$1 OUT=$2 ARMS=$3; shift 3
[ "${1:-}" = "--" ] && shift
[ -x "$BIN" ] || { echo "no serve binary at $BIN" >&2; exit 2; }
for a in "$@"; do case "$a" in /Volumes/*|/srv/models*) echo "$a is the archive (CLAUDE.md)" >&2; exit 2;; esac; done
mkdir -p "$OUT"
PORT=${GS5C_PORT:-18456}
IFS=, read -r -a arms <<<"$ARMS"
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
  grep -E "decode path|audio input" "$OUT/serve-$lab.log" | cut -c1-200 || true
  python3 - "$OUT" "$lab" "$PORT" <<'EOF'
import base64, json, sys, time, urllib.request
out, lab, port = sys.argv[1:4]
for clip in ["short", "mid", "long"]:
    wav = base64.b64encode(open(f"testdata/embeddinggemma2-audio/{clip}.wav", "rb").read()).decode()
    body = {"messages": [{"role": "user", "content": [
        {"type": "input_audio", "input_audio": {"data": wav, "format": "wav"}},
        {"type": "text", "text": "Transcribe this audio."}]}],
        "max_tokens": 32, "temperature": 0, "logprobs": True, "top_logprobs": 3}
    t = time.time()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=1800))
    c = r["choices"][0]
    open(f"{out}/reply-{clip}-{lab}.txt", "w").write(c["message"]["content"])
    json.dump((c.get("logprobs") or {}).get("content") or [], open(f"{out}/logprobs-{clip}-{lab}.json", "w"))
    print(f"  {lab} {clip} ({time.time() - t:.1f}s, {r['usage']['prompt_tokens']} prompt tokens): {c['message']['content'][:120]!r}")
EOF
  grep -E "vision: decoded" "$OUT/serve-$lab.log" | sed 's/^/  /' || true
  kill $pid; wait $pid 2>/dev/null || true; sleep 2
done
python3 - "$OUT" "${labels[@]}" <<'EOF'
import json, math, sys
out, ref, *others = sys.argv[1:]
for clip in ["short", "mid", "long"]:
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
