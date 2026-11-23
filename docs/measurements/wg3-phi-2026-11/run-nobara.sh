#!/usr/bin/env bash
# WG3-phi's nobara cells, registered in registration.md (this directory) before any timed run.
#   python3 <repo>/scripts/timing_lock.py run --wait 10800 --label wg3-phi-speed -- bash run-nobara.sh <base-rev> <head-rev>
# Builds both serve binaries from clean detached worktrees, fit-checks both arms, then the decode and TTFT cells.
set -uo pipefail
BASE=$1 HEAD=$2
export PATH=/usr/local/go/bin:$HOME/go/bin:$PATH
HERE=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
W=$HOME/goinfer-bench/wg3-phi-run
OUT=$W/results
mkdir -p "$OUT/fit" "$W/bin"
export TMPDIR=$W/tmp; mkdir -p "$TMPDIR"
ts() { date '+%F %T %Z'; }
M=$HOME/models/phi3-mini-4k-gguf/Phi-3-mini-4k-instruct-q4.gguf
r=$(readlink -f "$M"); case "$r" in /srv/models*|/Volumes/*) echo "FATAL: $M resolves to the archive ($r)"; exit 2;; esac
for arm in base head; do
  rev=$([ $arm = base ] && echo $BASE || echo $HEAD)
  wt=$W/wt-$arm
  [ -d "$wt" ] || git -C "$HERE" worktree add -q --detach "$wt" "$rev" || exit 2
  git -C "$wt" checkout -q --detach "$rev" || exit 2
  [ "$(git -C "$wt" rev-parse HEAD)" = "$(git -C "$HERE" rev-parse "$rev")" ] || { echo "FATAL: $wt is not at $rev"; exit 2; }
  [ -z "$(git -C "$wt" status --porcelain --untracked-files=no)" ] || { echo "FATAL: $wt has tracked changes"; exit 2; }
  (cd "$wt" && bash scripts/make_work.sh >/dev/null && CGO_ENABLED=1 go build -tags gpu -o "$W/bin/serve-webgpu-$arm" ./backend/webgpu/cmd/serve) || exit 2
done
{ echo "started $(ts) base $(git -C "$HERE" rev-parse --short=9 $BASE) head $(git -C "$HERE" rev-parse --short=9 $HEAD)"
  echo "driver $(nvidia-smi --query-gpu=driver_version --format=csv,noheader); gpu procs: $(nvidia-smi --query-compute-apps=process_name,used_memory --format=csv,noheader | tr '\n' ';')"
  sha256sum "$W/bin/serve-webgpu-base" "$W/bin/serve-webgpu-head"; } | tee "$OUT/provenance.txt"
# Fit check: each arm's decode path, from its own log.
fit() {
  local arm=$1 log=$OUT/fit/$1.log
  "$W/bin/serve-webgpu-$arm" -model "bench=$M" -backend webgpu -quant int8int8 -ctx 4096 -addr 127.0.0.1:18299 >"$log" 2>&1 &
  local pid=$!
  for _ in $(seq 1 120); do grep -q "decode path:" "$log" && break; sleep 1; done
  kill $pid; wait $pid 2>/dev/null
  grep -m1 "decode path:" "$log" | sed 's/.*decode path: *//'
}
fb=$(fit base); fh=$(fit head)
echo "fit base: $fb" | tee -a "$OUT/provenance.txt"; echo "fit head: $fh" | tee -a "$OUT/provenance.txt"
case "$fh" in webgpu:vulkan-resident*) ;; *) echo "STOP: head is not resident"; exit 75;; esac
case "$fb" in *-resident*|"") echo "STOP: base is resident or unread"; exit 75;; esac
H=$W/wt-head
# The builds above load the box; bench_peer.py refuses a 1-minute load over 1.0, so settle under 0.8 first (up to 15 minutes).
for _ in $(seq 1 90); do awk '{exit !($1 < 0.8)}' /proc/loadavg && break; sleep 10; done
echo "settled: load $(cut -d' ' -f1-3 /proc/loadavg) $(ts)" | tee -a "$OUT/provenance.txt"
export GOINFER_SERVE_WEBGPU=$W/bin/serve-webgpu-head GOINFER_SERVE_WEBGPU_OLD=$W/bin/serve-webgpu-base
export BENCH_ENGINES=goinfer,goinfer_old BENCH_BACKENDS=webgpu BENCH_PHASES=B BENCH_ABBA=2 BENCH_RUNS=2 BENCH_MODELS=phi3-mini BENCH_DEPTHS=128
export BENCH_QUANT_OVERRIDE=phi3-mini=int8int8 BENCH_SWAP_VOID_STOP=3 BENCH_SWAP_KILL_MB=1024 BENCH_SERVE_LOG_DIR=$OUT/serve-logs
(cd "$H" && python3 -u scripts/bench_peer.py "$OUT/decode.json") > "$OUT/decode.log" 2>&1; echo "decode rc=$? $(ts)" | tee -a "$OUT/provenance.txt"
(cd "$H" && WG0_REPO=$H python3 -u "$HERE/ttft.py" "$OUT/ttft.json" --arm head=$W/bin/serve-webgpu-head:webgpu \
  --arm base=$W/bin/serve-webgpu-base:webgpu --models phi3-mini --quads 2 --n 6 --ctx 4096) > "$OUT/ttft.log" 2>&1
echo "ttft rc=$? finished $(ts)" | tee -a "$OUT/provenance.txt"
