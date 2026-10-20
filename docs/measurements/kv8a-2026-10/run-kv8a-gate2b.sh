#!/bin/bash
# KV8a gate 2(b) on nobara (README.md here): builds serve at the pinned commit and runs gate2b.py over the KV0-H pilot's recorded bodies.
# Queued on night.py (the job holds the timing lock) as `bash run-kv8a-gate2b.sh <sha>`. Exit 0 PASS, 1 FAIL, 2 VOID or could not run.
set -u
SHA=${1:?the KV8a commit}
B=$HOME/goinfer-bench/kv8a; R=$HOME/mycode/goinfer; K=$HOME/goinfer-bench/kv0h-2026-10
OUT=$B/gate2b-$(date +%F-%H%M); mkdir -p "$OUT"
exec > >(tee -a "$OUT/run.log") 2>&1
echo "=== $(date '+%F %T %Z') start, pin $SHA; $(nvidia-smi --query-gpu=name,driver_version,memory.used --format=csv,noheader); $(uptime)"
OTHERS=$(nvidia-smi --query-compute-apps=pid,used_memory,name --format=csv,noheader | grep -v kwin_wayland | grep -c .)
[ "$OTHERS" = 0 ] || { echo "VOID: another process holds the card"; exit 2; }
git -C "$R" cat-file -e "$SHA^{commit}" 2>/dev/null || git -C "$R" fetch -q "$B/kv8a.bundle" "+kv8a-2026-10:refs/mac/kv8a-2026-10" || { echo "VOID: bundle"; exit 2; }
W=$B/wt
git -C "$W" checkout -q --force --detach "$SHA" || { echo "VOID: worktree"; exit 2; }
printf 'go 1.27.0\n\nuse (\n\t.\n\t./gpu\n\t./cuda\n)\n' > "$W/go.work"
BIN=$B/serve-cuda-${SHA:0:8}
[ -x "$BIN" ] || (cd "$W/cuda" && GOWORK="$W/go.work" CGO_ENABLED=0 go build -tags cuda -o "$BIN" ./cmd/serve) || { echo "VOID: serve build"; exit 2; }
python3 -u "$W/docs/measurements/kv8a-2026-10/gate2b.py" --serve "$BIN" --model "$HOME/models/Qwen3.5-9B-Q4_K_M.gguf" --key "$K/ids.key" \
	--bodies "$K/bodies-pilot" --out "$OUT"
rc=$?
echo "=== $(date '+%F %T %Z') end, rc $rc; record $OUT"
exit $rc
