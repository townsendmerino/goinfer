#!/usr/bin/env bash
# G-S10q-d (docs/tasks/task-multimodal-support-2026-10.md, "S10, Qwen3-VL MoE", registered before this script): served
# Qwen3-VL-30B-A3B at int4 on nobara.
#   0. sidecar   when the CPU int4 sidecar is missing, one serve load with the arms' own flags builds it (~17 GB, about
#                11 min) under a 40 min wait, then stops. The driver waits only 600 s for a server, which a first load
#                outlasts: the first run of this job was void that way (gs10q-d-void/).
#   1. graded    two CPU arms through ONE pinned serve binary (=cpu,cpu), G-S10m-d's driver (run-gs10m-served.sh, the
#                same one-image and two-image requests, 48 greedy tokens): PASS is identical replies on both requests.
#   2. graded    (amendment A2) one --backend cuda --moe-cache-experts arm on the 8 GB card (expert paging: the plain resident build declines
#                for memory, checked by day 2026-10-10; with the flag it loads at 22 of 64 expert slots per layer, ctx 4096, the
#                vision tower on CUDA), graded against the CPU arm by G-S10m-d's near-tie rule; exit code 0 pass, 3 fail, 4 void.
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
[ -n "${GS10QD_DRY:-}" ] && { echo "DRY: preconditions hold; plan = sidecar ($([ -e "$SIDECAR" ] && echo present || echo "to build")), graded =cpu,cpu at int4, then the graded cuda arm (--moe-cache-experts)"; exit 0; }
cd "$SRC"
# int4 is serve's default and is left unset on purpose: at the pinned rev an explicit --quant int4 on a MoE directory is
# refused against the sidecar the same load builds (its header says int4mix; the task doc's G-S10q-d record).
FLAGS=(--model "$MODEL" --vision "$MODEL")

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

echo "[$(date +%T)] 2/2 graded (A2): --backend cuda --moe-cache-experts"
"$BIN/serve" "${FLAGS[@]}" --backend cuda --embed-int4=false --moe-cache-experts --addr 127.0.0.1:$PORT >"$OUT/cuda/serve-cuda.log" 2>&1 </dev/null &
pid=$!
trap 'kill $pid 2>/dev/null || true' EXIT
up=0
for _ in $(seq 1 1200); do
  curl -s -o /dev/null -w '%{http_code}' 127.0.0.1:$PORT/v1/models 2>/dev/null | grep -q 200 && { up=1; break; }
  kill -0 $pid 2>/dev/null || break
  sleep 1
done
grep -E "decode path|backend|resident|declin|paging|vision tower" "$OUT/cuda/serve-cuda.log" | cut -c1-240 | sed 's/^/  /' || true
# Without --moe-cache-experts the resident build declines by name and serve continues on the CPU path, whose replies are CPU replies: if that ever happens here
# (a decline of the paged build too), comparing them with the CPU arm proves nothing, so the comparison is labelled.
FELL_BACK=0; grep -q "decode path: cpu" "$OUT/cuda/serve-cuda.log" 2>/dev/null && FELL_BACK=1
[ "$FELL_BACK" = 1 ] && echo "  NOTE: the cuda arm fell back to the CPU path (a named decline); its replies below are CPU replies, not CUDA ones"
if [ "$up" = 1 ]; then
  FELL_BACK=$FELL_BACK python3 - "$OUT/cuda" "$PORT" "$OUT/graded" <<'EOF'
import base64, json, os, sys, time, urllib.request
out, port, graded = sys.argv[1:4]
fell = os.environ.get("FELL_BACK") == "1"
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
    json.dump((r["choices"][0].get("logprobs") or {}).get("content") or [], open(f"{out}/logprobs-{name}-cuda.json", "w"))
    try:
        cpu = open(f"{graded}/reply-{name}-cpu.txt").read()
        same = "IDENTICAL to the cpu arm" if cpu == c else "differs from the cpu arm"
        if fell:
            same = "(CPU fallback, not a CUDA reply) " + same
    except OSError:
        same = "no cpu reply to compare"
    print(f"  cuda {name} ({time.time() - t:.1f}s): {same}: {c[:160]!r}", flush=True)
EOF
  # GRADED (amendment A2 to the registration, 2026-10-10, before the run): the cuda arm is graded with the near-tie rule G-S10m-d uses for a non-reference arm.
  CUDA_VOID=""
  [ "$FELL_BACK" = 1 ] && CUDA_VOID="the server fell back to the CPU path"
  grep -q "decode path: cuda-resident" "$OUT/cuda/serve-cuda.log" 2>/dev/null || CUDA_VOID="${CUDA_VOID:-the build is not cuda-resident}"
  CUDA_VOID="$CUDA_VOID" python3 - "$OUT/cuda" "$OUT/graded" <<'EOF'
import json, math, os, sys
cuda, graded = sys.argv[1:3]
if os.environ.get("CUDA_VOID"):
    print(f"G-S10q-d CUDA GRADE: VOID ({os.environ['CUDA_VOID']})"); sys.exit(4)
ok = True
for clip in ("one-image", "two-image"):
    try:
        rd = json.load(open(f"{graded}/logprobs-{clip}-cpu.json")); od = json.load(open(f"{cuda}/logprobs-{clip}-cuda.json"))
        rt = open(f"{graded}/reply-{clip}-cpu.txt").read(); ot = open(f"{cuda}/reply-{clip}-cuda.txt").read()
    except OSError as e:
        print(f"G-S10q-d CUDA GRADE: VOID ({clip}: {e})"); sys.exit(4)
    if rt == ot:
        print(f"{clip}: cuda against cpu: IDENTICAL replies: PASS"); continue
    i = 0
    while i < min(len(rd), len(od)) and rd[i]["token"] == od[i]["token"]:
        i += 1
    if i >= min(len(rd), len(od)):
        print(f"{clip}: cuda against cpu: replies differ in length only: PASS"); continue
    top = rd[i]["top_logprobs"]; ptop = math.exp(top[0]["logprob"])
    po = [math.exp(c["logprob"]) for c in top if c["token"] == od[i]["token"]]
    near = bool(po and po[0] >= ptop / 2)
    ok &= near
    print(f"{clip}: cuda against cpu: first differing token {i}: cpu {rd[i]['token']!r}, cuda {od[i]['token']!r}; cpu top-3 "
          + ", ".join(f"{c['token']!r} {math.exp(c['logprob']):.3f}" for c in top) + f"; near-tie {near}: {'PASS' if near else 'FAIL'}")
print(f"G-S10q-d CUDA GRADE: {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 3)
EOF
  rc2=$?
else
  echo "  the cuda arm did not come up (a decline or a failed load): see $OUT/cuda/serve-cuda.log"
  tail -5 "$OUT/cuda/serve-cuda.log" | sed 's/^/  /'
  echo "G-S10q-d CUDA GRADE: VOID (the arm did not come up)"; rc2=4
fi
kill $pid 2>/dev/null; wait $pid 2>/dev/null || true
echo "[$(date +%T)] done; graded (cpu) rc=$rc1, cuda grade rc=$rc2 (0 pass, 3 fail, 4 void)"
[ "$rc1" -ne 0 ] && exit "$rc1"
exit "$rc2"
