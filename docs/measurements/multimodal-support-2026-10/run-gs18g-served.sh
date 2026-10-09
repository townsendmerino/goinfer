#!/usr/bin/env bash
# G-S18g (docs/tasks/task-multimodal-support-2026-10.md, "S18 on the Mac, the tower", registered before the code): the
# served reply with Gemma 3's int8 Metal tower (tower_gemm_w8, weights in groups of 32, activations f32) against the f16
# Metal tower, on the same decoder. The four F2a images (table.png is one of them; the registration's "table.png and" adds no fifth), "What does this image show? Answer
# briefly.", 32 greedy tokens, top-3 log-probabilities. Arms, one fresh server each: f16 (the reference,
# -vision-quant f32), int8 (-vision-quant int8), and f16 again (a determinism control: it must be IDENTICAL to the first).
# PASS: for every image, identical replies, or the first difference at an R10 near-tie in the reference's
# log-probabilities (p(int8's token) >= half p(top)). A failure makes the int8 tower explicit-only, as registered.
#
# The decoder is the safetensors directory through its sidecar (`<dir>.int4.metal.giw`, built once 2026-10-08), as
# G-S18a served it. The first night (2026-10-08) used the ggml-org Q4_K_M GGUF instead, whose tokenizer has no
# <image_soft_token>, so every arm exited at the vision setup: VOID by design error, re-queued on the directory.
# Pinned: serve-metal from S18_REV in a detached worktree with its own go.work. Estimate: 3 servers x (load + 4 images
# at ~2-4 s of tower and ~3 s of decode) ~= 3 x 60 s, ~5 min; queued at 15.
set -uo pipefail
REV=${S18_REV:?set S18_REV to the pinned main commit}
R=$HOME/tmcode/goinfer
B=$HOME/goinfer-bench/s18
OUT=${1:-$B/gs18g-$(date +%F)}
DIR=$HOME/models/gemma-3-4b-it
for p in "$DIR" "$DIR.int4.metal.giw"; do [ -e "$p" ] || { echo "FATAL: $p is missing"; exit 2; }; done
mkdir -p "$OUT"
if [ -z "${S18_LOCK_HELD:-}" ]; then
  python3 "$R/scripts/timing_lock.py" run --label gs18g -- env S18_LOCK_HELD=1 bash "$0" "$OUT"
  exit $?
fi
w=$B/wt-$REV
[ -d "$w" ] || git -C "$R" worktree add --detach "$w" "$REV" || exit 1
[ "$(git -C "$w" rev-parse --short=8 HEAD)" = "$REV" ] || { echo "worktree is not at $REV"; exit 1; }
printf 'go 1.27.0\n\nuse (\n\t.\n\t./gpu\n\t./metal\n)\n' > "$w/go.work"
[ -x "$B/serve-metal-$REV" ] || (cd "$w/metal" && GOWORK=$w/go.work CGO_ENABLED=0 go build -o "$B/serve-metal-$REV" ./cmd/serve) || exit 1
SERVE=$B/serve-metal-$REV
{ echo "serve rev $REV (sha256 $(shasum -a 256 "$SERVE" | cut -c1-16))"; echo "started: $(date '+%F %T %Z')"
  sw_vers | tr '\n' ' '; echo; pmset -g batt | head -1; sysctl -n vm.swapusage; uptime; } | tee "$OUT/provenance.txt"
cd "$w" || exit 2
PORT=18459
rc=0
for arm in f16 int8 f16b; do
  q=f32; [ "$arm" = int8 ] && q=int8
  "$SERVE" --model "$DIR" --backend metal -vision-quant "$q" --addr 127.0.0.1:$PORT > "$OUT/serve-$arm.log" 2>&1 </dev/null &
  pid=$!
  for _ in $(seq 1 600); do
    curl -s -o /dev/null -w '%{http_code}' 127.0.0.1:$PORT/v1/models 2>/dev/null | grep -q 200 && break
    kill -0 $pid 2>/dev/null || { echo "serve exited ($arm)"; rc=1; break; }
    sleep 1
  done
  python3 - "$OUT" "$arm" "$PORT" <<'EOF' || rc=1
import base64, json, sys, time, urllib.request
out, arm, port = sys.argv[1:4]
imgs = ["glm_ocr/table.png", "gemma3_preprocess_image.png", "qwen25vl_preprocess_image.png", "glm_ocr/formula.png"]
recs = {}
for im in imgs:
    b = base64.b64encode(open("testdata/" + im, "rb").read()).decode()
    body = {"messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + b}},
        {"type": "text", "text": "What does this image show? Answer briefly."}]}],
        "max_tokens": 32, "temperature": 0, "logprobs": True, "top_logprobs": 3}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t = time.time()
    c = json.load(urllib.request.urlopen(req, timeout=1800))["choices"][0]
    recs[im] = {"reply": c["message"]["content"], "logprobs": (c.get("logprobs") or {}).get("content") or []}
    print(f"  {arm} {im} ({time.time() - t:.1f}s): {c['message']['content'][:120]!r}")
json.dump(recs, open(f"{out}/replies-{arm}.json", "w"))
EOF
  grep -E "decode path|vision tower|encoder|int8 Metal|KV plan" "$OUT/serve-$arm.log" | cut -c1-200
  kill $pid; wait $pid 2>/dev/null || true; sleep 30
done
python3 - "$OUT" <<'EOF' | tee "$OUT/verdict.txt"
import json, math, sys
out = sys.argv[1]
try:
    ref, i8, ctl = (json.load(open(f"{out}/replies-{a}.json")) for a in ("f16", "int8", "f16b"))
except Exception as e:
    print(f"G-S18g: VOID ({e})"); sys.exit(1)
ok = True
for im, r in ref.items():
    if ctl[im]["reply"] != r["reply"]:
        print(f"{im}: VOID: the f16 control differs from the f16 reference (the servers are not deterministic)"); ok = False; continue
    g = i8[im]
    if g["reply"] == r["reply"]:
        print(f"{im}: IDENTICAL"); continue
    lo, ln = r["logprobs"], g["logprobs"]
    i = 0
    while i < min(len(lo), len(ln)) and lo[i]["token"] == ln[i]["token"]:
        i += 1
    if i >= min(len(lo), len(ln)):
        print(f"{im}: FAIL: the replies differ in length only"); ok = False; continue
    top = lo[i]["top_logprobs"]
    ptop = math.exp(top[0]["logprob"])
    po = [math.exp(c["logprob"]) for c in top if c["token"] == ln[i]["token"]]
    near = bool(po and po[0] >= ptop / 2)
    print(f"{im}: first difference at generated token {i}: f16 {lo[i]['token']!r}, int8 {ln[i]['token']!r}; f16's top-3 "
          + ", ".join(f"{c['token']!r} {math.exp(c['logprob']):.3f}" for c in top) + f": {'near-tie' if near else 'NOT a near-tie'}")
    ok = ok and near
print(f"G-S18g: {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
EOF
[ "${PIPESTATUS[0]}" -eq 0 ] || rc=1
echo "finished: $(date '+%F %T %Z') rc=$rc" | tee -a "$OUT/provenance.txt"
exit $rc
