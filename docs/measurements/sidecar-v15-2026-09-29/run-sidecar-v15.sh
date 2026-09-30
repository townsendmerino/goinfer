#!/bin/bash
# nobara-pc: sidecar vs direct load on the format that ships (v15) and the current defaults.
# PREFLIGHT=1 runs the builds, the sidecars and a compile of the test, and stops before any timed run.
# Pre-registered in docs/measurements/sidecar-v15-2026-09-29.md, before this ran. Nothing here needs a Claude session.
set -u
export PATH=$PATH:/usr/local/go/bin
REV=dc0699d4
R=$HOME/mycode/goinfer
W=$HOME/goinfer-bench/sidecar-v15-2026-09-29
S=$W/home    # scratch HOME: symlinks to the two .gguf files and nothing else, so every sidecar below is built fresh
RH=$HOME
ts() { date '+%H:%M:%S'; }
phase() { echo "$(ts) == $1"; }
mkdir -p "$W" "$S/models"

phase "provenance"
{
  echo "rev $REV"; echo "host $(hostname) $(uname -sr)"
  nvidia-smi --query-gpu=driver_version,name --format=csv,noheader
  echo "aikit $(git -C "$RH/mycode/aikit/aikit" log --oneline -1)"
  echo "thp $(cat /sys/kernel/mm/transparent_hugepage/enabled)"
  echo "load $(cut -d' ' -f1-3 /proc/loadavg)"; echo "free $(df -h "$RH" | tail -1 | awk '{print $4}')"
} | tee "$W/provenance.txt"

phase "worktree at $REV"
git -C "$R" fetch -q origin || { echo "fetch failed"; exit 1; }
[ -d "$W/wt" ] || git -C "$R" worktree add --detach "$W/wt" "$REV" || { echo "worktree failed"; exit 1; }
[ "$(git -C "$W/wt" rev-parse --short=8 HEAD)" = "$REV" ] || { echo "worktree is not at $REV"; exit 1; }
cp "$R/go.work" "$W/wt/go.work"; sed -i "s#\.\./aikit/aikit#$RH/mycode/aikit/aikit#" "$W/wt/go.work"

for m in qwen2.5-coder-1.5b-instruct-q4_k_m qwen2.5-7b-instruct-q4_k_m; do
  ln -sfn "$RH/models/$m.gguf" "$S/models/$m.gguf"
  test -e "$S/models/$m.gguf" || { echo "missing $m.gguf"; exit 1; }
done

phase "build prequant and the CUDA serve binary"
(cd "$W/wt" && go build -o "$W/prequant" ./cmd/prequant) || exit 1
(cd "$W/wt/cuda" && go build -tags cuda -o "$W/serve-cuda" ./cmd/serve) || exit 1

version() { python3 - "$1" <<'PY'
import struct,sys
h=open(sys.argv[1],"rb").read(256); i=h.find(b"GINFW")
print(struct.unpack("<I",h[i+5:i+9])[0] if i>=0 else "none")
PY
}

phase "build the CPU sidecars fresh (v15 expected)"
for m in qwen2.5-coder-1.5b-instruct-q4_k_m qwen2.5-7b-instruct-q4_k_m; do
  t0=$SECONDS
  "$W/prequant" -o "$S/models/$m.int4.cpu-amd64.giw" -quant int4 -target cpu "$S/models/$m.gguf" || { echo "prequant failed for $m"; exit 1; }
  echo "$(ts) transcode $m: $((SECONDS - t0)) s, $(du -h "$S/models/$m.int4.cpu-amd64.giw" | cut -f1), format v$(version "$S/models/$m.int4.cpu-amd64.giw")"
  [ "$(version "$S/models/$m.int4.cpu-amd64.giw")" = "15" ] || { echo "the sidecar is not v15; stopping rather than measure the wrong format"; exit 1; }
done

if [ "${PREFLIGHT:-}" = 1 ]; then   # everything up to the timed runs, plus a compile of the test: no measurement
  (cd "$W/wt" && env HOME="$S" GOCACHE="$(go env GOCACHE)" GOMODCACHE="$(go env GOMODCACHE)" GOPATH="$(go env GOPATH)" go test -tags goinfer_testhooks -c -o /dev/null ./decoder/) || { echo "the test does not compile"; exit 1; }
  test -f "$W/wt/scripts/bench_sidecar_cuda.py" || { echo "no CUDA harness at $REV"; exit 1; }
  phase "PREFLIGHT OK"; exit 0
fi

phase "CPU: TestCPURoofline_giwVsDirect (the 2026-09-24 test, unchanged)"
(cd "$W/wt" && env HOME="$S" GOCACHE="$(go env GOCACHE)" GOMODCACHE="$(go env GOMODCACHE)" GOPATH="$(go env GOPATH)" \
   GOINFER_HEAVY_TESTS=1 go test -tags goinfer_testhooks ./decoder/ -run TestCPURoofline_giwVsDirect -v -count=1 -timeout 60m) 2>&1 | tee "$W/cpu-giw-vs-direct.log"
echo "$(ts) CPU exit=${PIPESTATUS[0]}"

phase "CUDA: bench_sidecar_cuda.py (the 2026-09-24 harness; the first start converts, the default now carries the int4 embedding table)"
(cd "$W/wt" && env HOME="$S" BENCH_OUT="$W" GOINFER_SERVE_CUDA="$W/serve-cuda" python3 -u scripts/bench_sidecar_cuda.py) 2>&1 | tee "$W/cuda-sidecar.log"
echo "$(ts) CUDA exit=${PIPESTATUS[0]}"
[ -f "$W/result.json" ] && mv "$W/result.json" "$W/cuda-result.json"

phase "formats and sizes of everything built"
for f in "$S"/models/*.giw; do echo "$(basename "$f"): v$(version "$f"), $(du -h "$f" | cut -f1)"; done | tee "$W/sidecars.txt"
phase "DONE"
