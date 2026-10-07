#!/usr/bin/env bash
# S9 on CUDA part A's speed record (docs/tasks/task-multimodal-support-2026-10.md, "S9 on CUDA, part A"): time to first token of the real Gemma 4 E2B on CUDA at ~270 and
# ~2,170 prompt tokens, batched prefill (the NEW binary) against sequential prefill (the PRE-CHANGE binary), as a night record with the registered kill rule: if the
# batched TTFT at ~270 tokens is not below the sequential one, the change parks. No other bar.
#
# Design: ROUNDS rounds; in each round both arms run, one fresh server per arm (cold KV), the order alternating round to round (new,old / old,new / ...) so a
# drifting box does not favour an arm. Per arm per size, REPS cold requests (each prompt starts with its own nonce, so none reuses another's prefix), max_tokens=1,
# non-streamed: the wall time of that request is prefill + one token. The ratio is formed PER ROUND (old median / new median) and the rounds are reported side by
# side, never pooled (CLAUDE.md, "difference matched observations").
#
# Usage: run-s9a-speed.sh [out dir]      Estimate: ~12 min (3 rounds x ~3.5 min, the old arm's 2,170-token cold requests dominate: ~25 s each x 5).
#   BIN=<dir with serve-cuda-new, serve-cuda-old> (defaults to ~/goinfer-bench/s9a; serve-cuda-old is the 928f9a41 build from ~/goinfer-bench/s1c)
# Queue:  python3 scripts/night.py add s9a-e2b-prefill-ttft --est 15 --by "nobara session, S9 on CUDA" --doc docs/tasks/task-multimodal-support-2026-10.md -- bash docs/measurements/multimodal-support-2026-10/run-s9a-speed.sh
set -uo pipefail
BIN=${BIN:-$HOME/goinfer-bench/s9a}
OUT=${1:-$HOME/goinfer-logs/s9a/speed-$(date +%F-%H%M)}
MODEL=${MODEL:-$HOME/models/gemma-4-e2b-gguf/gemma-4-E2B_q4_0-it.gguf}
ROUNDS=${ROUNDS:-3}; REPS=${REPS:-5}; PORT=${PORT:-18792}
for p in "$BIN/serve-cuda-new" "$BIN/serve-cuda-old" "$MODEL"; do [ -e "$p" ] || { echo "FATAL: $p is missing" >&2; exit 2; }; done
case "$MODEL" in /Volumes/*|/srv/models/*) echo "FATAL: $MODEL is on the archive (CLAUDE.md)" >&2; exit 2;; esac
mkdir -p "$OUT"
{ echo "bin: $BIN (new rev $(cat "$BIN/rev" 2>/dev/null || echo ?), old rev $(cat "$BIN/rev.old" 2>/dev/null || echo ?))"; echo "model: $MODEL"; echo "rounds=$ROUNDS reps=$REPS"
  echo "started: $(date '+%F %T %Z')"; echo "gpu: $(nvidia-smi --query-gpu=name,driver_version,memory.used --format=csv,noheader)"; echo "load: $(cat /proc/loadavg)"; } | tee "$OUT/provenance.txt"
pid=
trap '[ -n "$pid" ] && kill "$pid" 2>/dev/null; wait 2>/dev/null' EXIT
for round in $(seq 1 "$ROUNDS"); do
  arms="new old"; [ $((round % 2)) -eq 0 ] && arms="old new"
  for arm in $arms; do
    echo "[$(date +%T)] round $round arm $arm"
    "$BIN/serve-cuda-$arm" --model "$MODEL" --backend cuda --addr 127.0.0.1:$PORT >"$OUT/serve-r$round-$arm.log" 2>&1 </dev/null &
    pid=$!
    for _ in $(seq 1 240); do
      curl -s -o /dev/null -w '%{http_code}' 127.0.0.1:$PORT/v1/models 2>/dev/null | grep -q 200 && break
      kill -0 $pid 2>/dev/null || { echo "serve ($arm) exited; see $OUT/serve-r$round-$arm.log" >&2; exit 1; }
      sleep 1
    done
    python3 - "$OUT" "$round" "$arm" "$PORT" "$REPS" <<'EOF'
import json, sys, time, urllib.request
out, rnd, arm, port, reps = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
SENT = ["The harbour authority published its quarterly figures on %s, and the numbers surprised the committee.",
        "Cargo volumes rose in %s while the average berth time fell by a tenth of a day.",
        "A survey of %s pilots found that most preferred the northern channel in poor visibility.",
        "The report for %s lists seventeen vessels, four of which were refitted in the old yard.",
        "Maintenance of the %s crane was deferred twice because the replacement part arrived late."]
MONTHS = ["January","February","March","April","May","June","July","August","September","October","November","December"]
def prompt(n_sent, nonce):
    return f"[{nonce}] " + " ".join(SENT[i % len(SENT)] % MONTHS[(i * 7) % 12] for i in range(n_sent)) + "\n\nIn one sentence, what is this text about?"
res = {}
for size, n_sent in (("270", 14), ("2170", 126)):
    ts = []; ptok = None
    for k in range(reps):
        body = {"messages": [{"role": "user", "content": prompt(n_sent, f"r{rnd}{arm}{size}k{k}")}], "max_tokens": 1, "temperature": 0}
        req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=900) as r: d = json.loads(r.read())
        ts.append(time.time() - t0); ptok = d["usage"]["prompt_tokens"]
    res[size] = {"prompt_tokens": ptok, "times_s": ts}
    print(f"   {arm} {size} tok: " + " ".join(f"{t:.2f}" for t in ts))
json.dump(res, open(f"{out}/r{rnd}-{arm}.json", "w"))
EOF
    rc=$?; kill $pid 2>/dev/null; wait $pid 2>/dev/null; pid=
    [ $rc -eq 0 ] || { echo "!! client failed (round $round arm $arm)" >&2; exit 1; }
  done
done
python3 - "$OUT" "$ROUNDS" <<'EOF'
import json, statistics as st, sys
out, rounds = sys.argv[1], int(sys.argv[2])
verdict = "PASS"
for size in ("270", "2170"):
    print(f"== ~{size} prompt tokens (median of the reps, per round; ratio = old / new)")
    for r in range(1, rounds + 1):
        n, o = json.load(open(f"{out}/r{r}-new.json"))[size], json.load(open(f"{out}/r{r}-old.json"))[size]
        mn, mo = st.median(n["times_s"]), st.median(o["times_s"])
        print(f"   round {r}: new {mn:.2f}s (min {min(n['times_s']):.2f} max {max(n['times_s']):.2f})  old {mo:.2f}s (min {min(o['times_s']):.2f} max {max(o['times_s']):.2f})  ratio {mo/mn:.2f}x  tokens new={n['prompt_tokens']} old={o['prompt_tokens']}")
        if size == "270" and not mn < mo: verdict = "KILL (batched not below sequential at ~270 tokens: the change parks)"
print("kill rule:", verdict)
EOF
echo "finished: $(date '+%F %T %Z')" | tee -a "$OUT/provenance.txt"
