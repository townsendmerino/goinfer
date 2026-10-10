#!/usr/bin/env bash
# G-S10q-d (docs/tasks/task-multimodal-support-2026-10.md, "S10, Qwen3-VL MoE", registered before this script): served
# Qwen3-VL-30B-A3B at int4 on nobara.
#   0. sidecar   when the CPU int4 sidecar is missing, one serve load with the arms' own flags builds it (~17 GB, about
#                11 min) under a 40 min wait, then stops. The driver waits only 600 s for a server, which a first load
#                outlasts: the first run of this job was void that way (gs10q-d-void/).
#   1. graded    two CPU arms through ONE pinned serve binary (=cpu,cpu), G-S10m-d's driver (run-gs10m-served.sh, the
#                same one-image and two-image requests, 48 greedy tokens): PASS is identical replies on both requests.
#   2. reported  one --backend cuda arm on the 8 GB card: resident (with expert paging) or a named decline, its replies
#                set beside the CPU arm's. Not graded.
# Pinned in $BIN: serve (built at the rev in $BIN/rev) and run-gs10m-served.sh from that rev. Runs from $SRC (the driver
# reads testdata/ images relative to it). Checkpoints from ~/models (local NVMe), never the archive. Correctness gate,
# not timed.
# Estimate ~60 min (the sidecar ~12, each CPU arm's two requests ~10-15, the CUDA arm ~10); queue at 90:
#   python3 scripts/night.py add gs10q-d-2 --est 90 --by "Claude, S10 Qwen3-VL MoE G-S10q-d" --doc docs/tasks/task-multimodal-support-2026-10.md -- bash <this script>
# GS10QD_DRY=1 checks the preconditions and prints the plan.
set -uo pipefail
SRC=${SRC:-$HOME/wt/goinfer-gs10qd}
BIN=${BIN:-$HOME/goinfer-bench/gs10q-d}
OUT=${1:-$HOME/goinfer-logs/gs10q-d-$(date +%F)}
MODEL=$HOME/models/qwen3-vl-30b-a3b-instruct
PORT=${GS10M_PORT:-18459}
fatal() { echo "FATAL: $*" >&2; exit 2; }
for f in serve run-gs10m-served.sh rev; do [ -e "$BIN/$f" ] || fatal "$BIN/$f is missing"; done
for f in "$MODEL/config.json" "$MODEL/preprocessor_config.json" "$SRC/testdata/glm_ocr/table.png" "$SRC/testdata/gemma3_preprocess_image.png"; do
  [ -e "$f" ] || fatal "$f is missing"
done
for p in "$MODEL" "$OUT"; do case "$p" in /srv/models*|/Volumes/*) fatal "$p is the archive (CLAUDE.md)";; esac; done
free_gb=$(df -BG --output=avail "$HOME/models" | tail -1 | tr -dc '0-9'); [ "$free_gb" -ge 30 ] || fatal "only ${free_gb} GB free under ~/models, need 30 (the int4 sidecar is ~17 GB)"
mkdir -p "$OUT/graded" "$OUT/cuda"
{ echo "binaries: $BIN (rev $(cat "$BIN/rev"))"; echo "started: $(date '+%F %T %Z')"; echo "free: ${free_gb} GB; load: $(cat /proc/loadavg)"; free -g | sed -n 2p; nvidia-smi --query-gpu=driver_version,memory.total,memory.used --format=csv,noheader 2>/dev/null; } | tee "$OUT/provenance.txt"
SIDECAR=$MODEL.int4.cpu-amd64.giw
[ -n "${GS10QD_DRY:-}" ] && { echo "DRY: preconditions hold; plan = sidecar ($([ -e "$SIDECAR" ] && echo present || echo "to build")), graded =cpu,cpu at int4, then the reported cuda arm"; exit 0; }
cd "$SRC"
FLAGS=(--model "$MODEL" --vision "$MODEL" --quant int4)

if [ -e "$SIDECAR" ]; then
  echo "[$(date +%T)] 0/2 sidecar: present ($SIDECAR)"
else
  echo "[$(date +%T)] 0/2 sidecar: building $SIDECAR"
  t0=$SECONDS
  "$BIN/serve" "${FLAGS[@]}" --backend cpu --embed-int4=false --addr 127.0.0.1:$PORT >"$OUT/serve-sidecar.log" 2>&1 </dev/null &
  spid=$!
  sup=0
  for i in $(seq 1 2400); do
    curl -s -o /dev/null -w '%{http_code}' 127.0.0.1:$PORT/v1/models 2>/dev/null | grep -q 200 && { sup=1; break; }
    kill -0 $spid 2>/dev/null || break
    [ $((i % 60)) -eq 0 ] && echo "[$(date +%T)] ... sidecar load running $((SECONDS - t0))s; tmp $(du -sh "${SIDECAR%.giw}.tmp.giw" 2>/dev/null | cut -f1)"
    sleep 1
  done
  kill $spid 2>/dev/null; wait $spid 2>/dev/null || true
  sleep 3
  grep -E "transcod|cache ready|loaded" "$OUT/serve-sidecar.log" | cut -c1-200 | sed 's/^/  /' || true
  [ "$sup" = 1 ] && [ -e "$SIDECAR" ] || fatal "the sidecar load did not come up in $((SECONDS - t0))s (see $OUT/serve-sidecar.log): no reading"
  echo "[$(date +%T)] sidecar built in $((SECONDS - t0))s"
fi

echo "[$(date +%T)] 1/2 graded: =cpu,cpu"
GS10M_PORT=$PORT bash "$BIN/run-gs10m-served.sh" "$BIN/serve" "$OUT/graded" =cpu,cpu -- "${FLAGS[@]}" 2>&1 | tee "$OUT/graded/run.log"
rc1=${PIPESTATUS[0]}
echo "[$(date +%T)] graded step rc=$rc1"

echo "[$(date +%T)] 2/2 reported: --backend cuda"
"$BIN/serve" "${FLAGS[@]}" --backend cuda --embed-int4=false --addr 127.0.0.1:$PORT >"$OUT/cuda/serve-cuda.log" 2>&1 </dev/null &
pid=$!
trap 'kill $pid 2>/dev/null || true' EXIT
up=0
for _ in $(seq 1 1200); do
  curl -s -o /dev/null -w '%{http_code}' 127.0.0.1:$PORT/v1/models 2>/dev/null | grep -q 200 && { up=1; break; }
  kill -0 $pid 2>/dev/null || break
  sleep 1
done
grep -E "decode path|backend|resident|declin|paging|vision tower" "$OUT/cuda/serve-cuda.log" | cut -c1-240 | sed 's/^/  /' || true
if [ "$up" = 1 ]; then
  python3 - "$OUT/cuda" "$PORT" "$OUT/graded" <<'EOF'
import base64, json, sys, time, urllib.request
out, port, graded = sys.argv[1:4]
def img(p):
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(open("testdata/" + p, "rb").read()).decode()}}
reqs = {"one-image": [img("glm_ocr/table.png"), {"type": "text", "text": "What does this table show? Answer in one sentence."}],
        "two-image": [img("glm_ocr/table.png"), img("gemma3_preprocess_image.png"), {"type": "text", "text": "Describe each image in one sentence."}]}
for name, content in reqs.items():
    body = {"messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": content}],
            "max_tokens": 48, "temperature": 0, "logprobs": True, "top_logprobs": 3}
    t = time.time()
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        r = json.load(urllib.request.urlopen(req, timeout=1800))
    except Exception as e:
        print(f"  cuda {name}: request failed: {e}"); continue
    c = r["choices"][0]["message"]["content"]
    open(f"{out}/reply-{name}-cuda.txt", "w").write(c)
    try:
        cpu = open(f"{graded}/reply-{name}-cpu.txt").read()
        same = "IDENTICAL to the cpu arm" if cpu == c else "differs from the cpu arm"
    except OSError:
        same = "no cpu reply to compare"
    print(f"  cuda {name} ({time.time() - t:.1f}s): {same}: {c[:160]!r}", flush=True)
EOF
else
  echo "  the cuda arm did not come up (a decline or a failed load): see $OUT/cuda/serve-cuda.log"
  tail -5 "$OUT/cuda/serve-cuda.log" | sed 's/^/  /'
fi
kill $pid 2>/dev/null; wait $pid 2>/dev/null || true
echo "[$(date +%T)] done; graded rc=$rc1"
exit "$rc1"
