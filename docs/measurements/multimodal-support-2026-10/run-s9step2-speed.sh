#!/usr/bin/env bash
# S9 step 2's night speed job on the Mac (docs/tasks/task-multimodal-support-2026-10.md, "S9 step 2 on Metal", the
# night rule registered before it runs): E2B time to first token with the f16 batched E-model prefill (arm "batched":
# the serve binary from S9B_ON_REV, the local night-only branch s9b-night-on, which is main with emodelBatchedOn true and
# a one-time stderr line when the pass runs) against the S9 layer-major pass (arm "lm": the serve binary from S9B_REV,
# main, where the switch is off). Two requests per fresh server, so no feature cache or prefix reuse helps either arm:
# a ~512-token text prompt (the graded cell) and table.png (a record against the 2.4-2.9x projection band), 8 tokens
# each, streamed; TTFT is the time to the first content chunk. PASSES passes (default 5), the arm order alternating,
# under the timing lock. Both arms run the vision tower on the same device (-vision-device cpu, as run-s9-speed.sh), so
# the image TTFT difference is the prefill.
#
# The rule (registered in the task doc): the text cell's median of the per-pass ratios lm/batched; ship at >= 1.02 with
# the batched arm's "ran" line present in every one of its server logs; park 1.00-1.02; off below 1.00. A batched arm
# without the line voids the run (a silent fallback would read as a park).
#
# Pinned: detached worktrees with their own go.work (the global GOWORK points at the moving main checkout).
# Estimate: ~45 s a server (load, two requests), 2 arms x 5 passes, ~8 min, plus two serve builds (~2 min).
#   python3 scripts/night.py add s9step2-speed --est 15 --by "Claude, S9 step 2" --doc docs/tasks/task-multimodal-support-2026-10.md -- env S9B_REV=<main> S9B_ON_REV=<s9b-night-on> bash docs/measurements/multimodal-support-2026-10/run-s9step2-speed.sh
set -uo pipefail
REV=${S9B_REV:?set S9B_REV to the pinned main commit}
OREV=${S9B_ON_REV:?set S9B_ON_REV to the s9b-night-on commit}
R=$HOME/tmcode/goinfer
B=$HOME/goinfer-bench/s9step2
OUT=${1:-$B/run-$(date +%F)}
MODEL=$HOME/models/gemma-4-e2b-gguf/gemma-4-E2B_q4_0-it.gguf
VIS=$HOME/models/gemma-4-E2B-unq
for p in "$MODEL" "$VIS"; do [ -e "$p" ] || { echo "FATAL: $p is missing"; exit 2; }; done
mkdir -p "$OUT"
if [ -z "${S9B_LOCK_HELD:-}" ]; then
  python3 "$R/scripts/timing_lock.py" run --label s9step2-speed -- env S9B_LOCK_HELD=1 bash "$0" "$OUT"
  exit $?
fi
wt() { # wt <rev>: a detached worktree at rev with its own go.work; prints its path
  local w=$B/wt-$1
  [ -d "$w" ] || git -C "$R" worktree add --detach "$w" "$1" >&2 || return 1
  [ "$(git -C "$w" rev-parse --short=8 HEAD)" = "$1" ] || { echo "worktree is not at $1" >&2; return 1; }
  printf 'go 1.27.0\n\nuse (\n\t.\n\t./gpu\n\t./metal\n)\n' > "$w/go.work"
  echo "$w"
}
WT=$(wt "$REV") || exit 1
OWT=$(wt "$OREV") || exit 1
[ -x "$B/serve-metal-$REV" ] || (cd "$WT/metal" && GOWORK=$WT/go.work CGO_ENABLED=0 go build -o "$B/serve-metal-$REV" ./cmd/serve) || exit 1
[ -x "$B/serve-metal-$OREV" ] || (cd "$OWT/metal" && GOWORK=$OWT/go.work CGO_ENABLED=0 go build -o "$B/serve-metal-$OREV" ./cmd/serve) || exit 1
{ echo "lm rev $REV (sha256 $(shasum -a 256 "$B/serve-metal-$REV" | cut -c1-16)), batched rev $OREV (sha256 $(shasum -a 256 "$B/serve-metal-$OREV" | cut -c1-16))"
  echo "started: $(date '+%F %T %Z')"; sw_vers | tr '\n' ' '; echo; pmset -g batt | head -1; sysctl -n vm.swapusage; uptime; } | tee "$OUT/provenance.txt"
cd "$WT" || exit 2
PORT=18458
one() { # arm pass
  local arm=$1 pass=$2 bin=$B/serve-metal-$REV
  [ "$arm" = batched ] && bin=$B/serve-metal-$OREV
  "$bin" --model "$MODEL" --vision "$VIS" --backend metal -vision-device cpu --addr 127.0.0.1:$PORT \
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
para = ("The quarterly report covers unit sales across five regions, with notes on supply, pricing and the effect of "
        "seasonal demand on each product line. ")
t_txt = ttft([{"type": "text", "text": para * 20 + "Summarise the report in one sentence."}])
img = base64.b64encode(open("testdata/glm_ocr/table.png", "rb").read()).decode()
t_img = ttft([{"type": "image_url", "image_url": {"url": "data:image/png;base64," + img}},
              {"type": "text", "text": "What does this image show? Answer briefly."}])
rec = {"arm": arm, "pass": int(pas), "text_ttft_s": round(t_txt, 3), "image_ttft_s": round(t_img, 3)}
print(json.dumps(rec))
open(f"{out}/records.jsonl", "a").write(json.dumps(rec) + "\n")
EOF
  local rc=$?
  grep -E "vision: decoded|prefill path|E-model batched prefill ran" "$OUT/serve-$arm-$pass.log" | sed 's/^/  /' | cut -c1-160
  kill $pid; wait $pid 2>/dev/null || true; sleep 3
  return $rc
}
rc=0
for pass in $(seq 1 "${PASSES:-5}"); do
  if [ $((pass % 2)) -eq 1 ]; then order="lm batched"; else order="batched lm"; fi
  for arm in $order; do one "$arm" "$pass" || { rc=1; break 2; }; done
done
python3 - "$OUT" "${PASSES:-5}" <<'EOF' | tee "$OUT/verdict.txt"
import glob, json, statistics, sys
out, n = sys.argv[1], int(sys.argv[2])
recs = [json.loads(l) for l in open(out + "/records.jsonl")]
ran = all("E-model batched prefill ran" in open(f).read() for f in glob.glob(out + "/serve-batched-*.log"))
for k in ["text_ttft_s", "image_ttft_s"]:
    by = {(r["arm"], r["pass"]): r[k] for r in recs}
    ratios = [by[("lm", p)] / by[("batched", p)] for p in range(1, n + 1) if ("lm", p) in by and ("batched", p) in by]
    if not ratios:
        print(f"{k}: no complete pass")
        continue
    lm = [by[("lm", p)] for p in range(1, n + 1) if ("lm", p) in by]
    bt = [by[("batched", p)] for p in range(1, n + 1) if ("batched", p) in by]
    print(f"{k}: lm {lm} (median {statistics.median(lm):.3f}), batched {bt} (median {statistics.median(bt):.3f}); "
          f"per-pass lm/batched {[round(x, 3) for x in ratios]}, median {statistics.median(ratios):.3f}x")
    if k == "text_ttft_s":
        m = statistics.median(ratios)
        if not ran or len(ratios) < n:
            v = "VOID (" + ("the batched arm's ran line is missing" if not ran else f"{len(ratios)} of {n} passes") + ")"
        else:
            v = "SHIP" if m >= 1.02 else "PARK" if m >= 1.00 else "OFF"
        print(f"S9 step 2 text-prefill rule: {v}")
    else:
        print("image TTFT: a record against the projection band 2.4-2.9x")
EOF
echo "finished: $(date '+%F %T %Z') rc=$rc" | tee -a "$OUT/provenance.txt"
exit $rc
