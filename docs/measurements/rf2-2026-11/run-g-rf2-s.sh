#!/usr/bin/env bash
# G-RF2-S and never-swap (docs/tasks/task-forward-and-memory-2026-11.md, "RF2: registration" and "RF2: addendum 1"), on nobara, by day.
# KV4's G2 instruments (docs/measurements/kv4-cuda-2026-10-21/run-speed.sh, run-ttft.sh, run-mc3.sh), with the addendum's changes:
# decode includes depth 128, and every cell is A B B A in this one session (A = goinfer_old, the merge base; B = this build).
#   run-g-rf2-s.sh <out dir> <old serve-cuda> <new serve-cuda> [phases]
# phases (default all): decode deep gemma ttft mc3 verify. verify is bench_spec_lane.py's exact attention with -spec ngram on each
# build (A = OLD, B = NEW via --build-b), cells 1.5B:3900 and 7B:7500, three pairs.
# Build each binary once from its own worktree:
#   (cd <wt>/backend/cuda && GOWORK=<wt>/go.work CGO_ENABLED=0 go build -tags cuda -o <bins>/serve-cuda-old ./cmd/serve), and serve-cuda-new
# from the new worktree. The names are required (real files, not links): the never-swap sampler and KV4's score_speed.py find the
# servers by them.
set -euo pipefail
# Every path is made absolute here, before any cd: the lock re-enters this script by SELF, and OUT, OLD and NEW are read after cd.
SELF=$(realpath "$0")
# idle: the 1-minute load under 0.9 (bench_peer refuses above 1.0), waiting up to 15 minutes.
idle() { for _ in $(seq 90); do awk '{exit !($1 < 0.9)}' /proc/loadavg && return 0; sleep 10; done; echo "box not idle after 15 min"; return 1; }
# Inside the lock (lock below re-enters this script): the idle check is re-read once the lock is held, so a cell never starts on
# a reading taken before it waited; a cell that finds the box busy exits 75 and the run stops there (void: re-run that phase).
if [ "${1:-}" = __cell ]; then
  shift
  idle || exit 75
  echo "$(date +%T) cell start load $(cut -d' ' -f1 /proc/loadavg)"
  exec "$@"
fi
OUT=$(realpath -m "${1:?out dir}"); OLD=$(realpath -e "${2:?old serve-cuda}"); NEW=$(realpath -e "${3:?new serve-cuda}"); PHASES=${4:-decode deep gemma ttft mc3 verify}
[ "$OLD" != "$NEW" ] && [ "$(sha256sum < "$OLD")" != "$(sha256sum < "$NEW")" ] || { echo "old and new are the same binary"; exit 2; }
[ "$(basename "$OLD")" = serve-cuda-old ] && [ "$(basename "$NEW")" = serve-cuda-new ] || { echo "the binaries must be named serve-cuda-old and serve-cuda-new (the never-swap sampler and score_speed.py find the servers by name)"; exit 2; }
REPO=$(cd "$(dirname "$SELF")/../../.." && pwd)
KV4=$REPO/docs/measurements/kv4-cuda-2026-10-21
mkdir -p "$OUT"; cd "$REPO"
python3 "$KV4/procmem_sampler.py" "$OUT/procmem.jsonl" "$(basename "$OLD")" "$(basename "$NEW")" &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null || true' EXIT
export GOINFER_SERVE_CUDA=$NEW GOINFER_SERVE_CUDA_OLD=$OLD
export BENCH_ENGINES=goinfer,goinfer_old BENCH_BACKENDS=cuda BENCH_ABBA=1 BENCH_REQUIRE_BACKEND=1 BENCH_SERVE_LOG_DIR="$OUT/serve-logs"
lock() { python3 scripts/timing_lock.py run --wait 1800 --label "g-rf2-s-$1" -- bash "$SELF" __cell "${@:2}"; }
for ph in $PHASES; do
  echo "$(date +%T) phase $ph"
  case $ph in
    decode) lock decode env BENCH_MODELS=1.5B,7B BENCH_DEPTHS=128,2048,3900 python3 scripts/bench_peer.py "$OUT/decode-qwen.json" 2>&1 | tee "$OUT/decode-qwen.log" ;;
    deep)   lock deep env BENCH_CTX=12288 BENCH_MODELS=1.5B,7B BENCH_DEPTHS=8000 python3 scripts/bench_peer.py "$OUT/decode-deep.json" 2>&1 | tee "$OUT/decode-deep.log" ;;
    gemma)  lock gemma env BENCH_MODELS=gemma3-1b BENCH_DEPTHS=128,2048,3900 python3 scripts/bench_peer.py "$OUT/decode-gemma3.json" 2>&1 | tee "$OUT/decode-gemma3.log" ;;
    ttft)   for mk in 1.5B 7B; do lock "ttft-$mk" env TTFT_MODEL=$mk python3 "$KV4/ttft_ab.py" "$OLD" "$NEW" "$OUT/ttft-$mk.json" 2>&1 | tee "$OUT/ttft-$mk.log"; done ;;
    mc3)
      M15=$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf
      # bench_w7_plain.py launches GOINFER_SERVE_CPU whatever the backend; the cell names its binary there and drops the global
      # GOINFER_SERVE_CUDA (=$NEW, for bench_peer), so no cell can take the other arm's binary. Each cell logs the hash it ran.
      for k in old4_1 new4_1 new4_2 old4_2 old4_3 new4_3; do
        case $k in old*) bin=$OLD;; new*) bin=$NEW;; esac
        echo "$(date +%T) mc3 cell $k binary $bin sha256 $(sha256sum < "$bin" | cut -c1-16)" | tee -a "$OUT/mc3-cells.log"
        lock "mc3-$k" env -u GOINFER_SERVE_CUDA -u GOINFER_SERVE_CUDA_OLD BENCH_W7_MODEL=$M15 GOINFER_SERVE_CPU=$bin \
          python3 -u scripts/bench_w7_plain.py "$OUT/w7.json" --clients 4 --engines goinfer \
          --backend cuda --key $k --fixed-nonce --server-log "$OUT/w7-servers.log"
      done ;;
    verify) lock verify env GOINFER_SERVE_CUDA=$OLD python3 scripts/bench_spec_lane.py --out "$OUT/verify.json" --cell 1.5B:3900 --cell 7B:7500 \
              --pairs 3 --build-b "$NEW" 2>&1 | tee "$OUT/verify.log" ;;
  esac
done
echo RUN-G-RF2-S-DONE
