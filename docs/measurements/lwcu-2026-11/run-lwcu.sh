#!/usr/bin/env bash
# LW-cu's registered decode and TTFT read (docs/measurements/lwcu-2026-11/registration.md): pinned serve-cuda-old (51ed2ae27) against serve-cuda-new (4edbe93e9),
# A B B A per cell group, every GPU step under the exclusive timing lock, G4's sampler alongside. Run from a checkout of main that has this file.
#   run-lwcu.sh <out dir>
set -euo pipefail
OUT=${1:?out dir}
mkdir -p "$OUT"
REPO=$(git rev-parse --show-toplevel)
BINS=${LWCU_BINS:-$HOME/goinfer-bench/lwcu-2026-11-12/bins}
D=$REPO/docs/measurements/kv4-cuda-2026-10-21
sha256sum "$BINS/serve-cuda-old" "$BINS/serve-cuda-new" | tee "$OUT/binaries.sha256"
grep -q 465227c366a1ce2f47b52a9e34193e92da4b59bd64493f2ebd147c156d0f06c3 "$OUT/binaries.sha256" || { echo "serve-cuda-old is not the registered build"; exit 1; }
grep -q 1d671c1e483267d402e168e8f383d71f0f60d4e74c05555dcc99efb1c1136dfa "$OUT/binaries.sha256" || { echo "serve-cuda-new is not the registered build"; exit 1; }
cd "$REPO"
python3 "$D/procmem_sampler.py" "$OUT/procmem.jsonl" serve-cuda-old serve-cuda-new &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null || true' EXIT
export GOINFER_SERVE_CUDA=$BINS/serve-cuda-new GOINFER_SERVE_CUDA_OLD=$BINS/serve-cuda-old
export BENCH_ENGINES=goinfer,goinfer_old BENCH_BACKENDS=cuda BENCH_ABBA=1 BENCH_REQUIRE_BACKEND=1 BENCH_SERVE_LOG_DIR="$OUT/serve-logs"
idle() { for _ in $(seq 90); do awk '{exit !($1 < 0.9)}' /proc/loadavg && return 0; sleep 10; done; echo "box not idle after 15 min"; return 1; }
# bench_peer.py takes the exclusive timing lock itself; ttft_ab.py is wrapped (as KV4's run-ttft.sh does).
lock() { python3 scripts/timing_lock.py run --wait 1800 --label "lwcu-$1" -- "${@:2}"; }
record() { echo "$(date '+%F %T %Z') start $1: load1 $(cut -d' ' -f1 /proc/loadavg), MemAvailable $(awk '/MemAvailable/{print $2/1024 " MB"}' /proc/meminfo)" | tee -a "$OUT/cell-starts.log"; }
for ph in qwen deep gemma; do
  idle; record "decode-$ph"
  case $ph in
    qwen)  BENCH_MODELS=1.5B,7B BENCH_DEPTHS=2048,3900 python3 scripts/bench_peer.py "$OUT/decode-qwen.json" 2>&1 | tee "$OUT/decode-qwen.log" ;;
    deep)  BENCH_CTX=12288 BENCH_MODELS=1.5B,7B BENCH_DEPTHS=8000 python3 scripts/bench_peer.py "$OUT/decode-deep.json" 2>&1 | tee "$OUT/decode-deep.log" ;;
    gemma) BENCH_MODELS=gemma3-1b BENCH_DEPTHS=2048,3900 python3 scripts/bench_peer.py "$OUT/decode-gemma3.json" 2>&1 | tee "$OUT/decode-gemma3.log" ;;
  esac
done
for mk in 1.5B 7B; do
  idle; record "ttft-$mk"
  TTFT_MODEL=$mk lock "ttft-$mk" python3 "$D/ttft_ab.py" "$BINS/serve-cuda-old" "$BINS/serve-cuda-new" "$OUT/ttft-$mk.json" 2>&1 | tee "$OUT/ttft-$mk.log"
done
python3 "$D/score_speed.py" "$OUT/procmem.jsonl" "$OUT"/decode-*.json 2>&1 | tee "$OUT/score.txt"
echo LWCU-DONE
