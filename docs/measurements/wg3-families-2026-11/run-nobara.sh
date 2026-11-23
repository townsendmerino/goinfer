#!/usr/bin/env bash
# WG3 families' nobara cells, registered in registration.md (this directory) before any timed run. One family per call:
#   python3 <repo>/scripts/timing_lock.py run --wait 14400 --label wg3-families -- bash run-nobara.sh <key> <base-rev> <head-rev> [decode,ttft]
# key: glm-ocr | ministral3 | cr7b. Builds both arms from clean detached worktrees, fit-checks them, calibrates the family's
# prompts into the results' prompts copy, then the decode and TTFT cells.
set -uo pipefail
KEY=$1 BASE=$2 HEAD=$3 STEPS=${4:-decode,ttft}
export PATH=/usr/local/go/bin:$HOME/go/bin:$PATH
HERE=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
W=$HOME/goinfer-bench/wg3-families-run
OUT=$W/results/$KEY
mkdir -p "$OUT/fit" "$W/bin"
export TMPDIR=$W/tmp; mkdir -p "$TMPDIR"
ts() { date '+%F %T %Z'; }
case $KEY in glm-ocr) M=$HOME/models/glm-ocr;; ministral3) M=$HOME/models/ministral3-3b-bf16;; cr7b) M=$HOME/models/command-r7b;;
  *) echo "FATAL: key $KEY"; exit 2;; esac
r=$(readlink -f "$M"); case "$r" in /srv/models*|/Volumes/*) echo "FATAL: $M resolves to the archive ($r)"; exit 2;; esac
for arm in base head; do
  rev=$([ $arm = base ] && echo $BASE || echo $HEAD)
  wt=$W/wt-$KEY-$arm
  [ -d "$wt" ] || git -C "$HERE" worktree add -q --detach "$wt" "$rev" || exit 2
  git -C "$wt" checkout -q --detach "$rev" || exit 2
  [ -z "$(git -C "$wt" status --porcelain --untracked-files=no)" ] || { echo "FATAL: $wt has tracked changes"; exit 2; }
  (cd "$wt" && bash scripts/make_work.sh >/dev/null && CGO_ENABLED=1 go build -tags gpu -o "$W/bin/serve-webgpu-$KEY-$arm" ./backend/webgpu/cmd/serve) || exit 2
done
SB=$W/bin/serve-webgpu-$KEY-base SH=$W/bin/serve-webgpu-$KEY-head
{ echo "started $(ts) $KEY base $(git -C "$HERE" rev-parse --short=9 $BASE) head $(git -C "$HERE" rev-parse --short=9 $HEAD)"
  echo "driver $(nvidia-smi --query-gpu=driver_version --format=csv,noheader); gpu procs: $(nvidia-smi --query-compute-apps=process_name,used_memory --format=csv,noheader | tr '\n' ';')"
  sha256sum "$SB" "$SH"; } | tee -a "$OUT/provenance.txt"
fit() {
  local bin=$1 log=$2
  "$bin" -model "bench=$M" -backend webgpu -quant int4 -ctx 4096 -addr 127.0.0.1:18299 >"$log" 2>&1 &
  local pid=$!
  for _ in $(seq 1 900); do grep -q "decode path:" "$log" && break; sleep 1; done
  kill $pid; wait $pid 2>/dev/null
  grep -m1 "decode path:" "$log" | sed 's/.*decode path: *//'
}
fb=$(fit "$SB" "$OUT/fit/base.log"); fh=$(fit "$SH" "$OUT/fit/head.log")
echo "fit base: $fb" | tee -a "$OUT/provenance.txt"; echo "fit head: $fh" | tee -a "$OUT/provenance.txt"
case "$fh" in webgpu:*-resident*) ;; *) echo "STOP: head is not resident"; exit 75;; esac
case "$fb" in *-resident*|"") echo "STOP: base is resident or unread"; exit 75;; esac
P=$W/results/prompts.json
[ -f "$P" ] || cp "$W/wt-$KEY-head/scripts/prompts.json" "$P"
H=$W/wt-$KEY-head
if ! python3 -c "import json,sys; d=json.load(open('$P')); sys.exit(0 if all(f'$KEY:{n}' in d for n in (128,1024)) else 1)"; then
  (cd "$H" && python3 -u scripts/bench_prompts_calibrate.py --serve "$SB" --backend webgpu --quant int4 --model "$KEY=$M" \
    --depth 128 --depth 1024 --out "$P") > "$OUT/calibrate.log" 2>&1 || { echo "STOP: calibration failed"; exit 2; }
fi
for _ in $(seq 1 90); do awk '{exit !($1 < 0.8)}' /proc/loadavg && break; sleep 10; done
echo "settled: load $(cut -d' ' -f1-3 /proc/loadavg) $(ts)" | tee -a "$OUT/provenance.txt"
export BENCH_PROMPTS=$P WG0_REPO=$H
export GOINFER_SERVE_WEBGPU=$SH GOINFER_SERVE_WEBGPU_OLD=$SB
export BENCH_ENGINES=goinfer,goinfer_old BENCH_BACKENDS=webgpu BENCH_DEPTH_BACKEND=webgpu BENCH_PHASES=B BENCH_ABBA=2 BENCH_RUNS=2 BENCH_MODELS=$KEY BENCH_DEPTHS=128
export BENCH_LOAD_TIMEOUT=900 BENCH_QUANT_OVERRIDE=$KEY=int4 BENCH_SWAP_VOID_STOP=3 BENCH_SWAP_KILL_MB=1024 BENCH_SERVE_LOG_DIR=$OUT/serve-logs
case ",$STEPS," in *,decode,*)
(cd "$H" && python3 -u "$HERE/bench.py" decode "$OUT/decode.json") > "$OUT/decode.log" 2>&1; echo "decode rc=$? $(ts)" | tee -a "$OUT/provenance.txt";;
esac
case ",$STEPS," in *,ttft,*)
(cd "$H" && python3 -u "$HERE/bench.py" ttft "$OUT/ttft.json" --arm head=$SH:webgpu --arm base=$SB:webgpu --models $KEY --quads 2 --n 6 --ctx 4096) > "$OUT/ttft.log" 2>&1
echo "ttft rc=$? finished $(ts)" | tee -a "$OUT/provenance.txt";;
esac
