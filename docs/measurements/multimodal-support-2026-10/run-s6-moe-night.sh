#!/usr/bin/env bash
# S6's Qwen3.6-35B-A3B image check (docs/tasks/task-multimodal-support-2026-10.md, "S6 on nobara, the Qwen3.5+ MoE served check, registered 2026-10-08"):
# ONE image request (testdata/qwen35vl_preprocess_image.png, 96x64, "What does this image show? Answer briefly.", 32 greedy tokens, top-3 log-probabilities) through ONE
# CUDA serve binary, three arms, all `--backend cuda` with the C' MoE decode: the tower on the CUDA (-vision-device auto), the tower on the CPU (the reference, "="), and the
# CPU-tower arm again as the control. The image turn prefills on the CPU on every arm (no MoE hybrid image gate exists), so the one thing that differs between the first two
# is the tower. Reading: the CUDA-tower reply identical to the CPU-tower one, or a first divergence at a near-tie (p(other) >= half p(top)); the repeat IDENTICAL; each arm's log
# must say `decode path: cuda-resident` and, for the first, a vision tower on CUDA; anything else voids the reading (a silent CPU fallback is the failure this rig exists to see).
# No HF anchor exists for the 35B on this box (70 GB of bf16 against 62 GB of RAM): the claim is "runs, and the CUDA tower is the CPU tower".
# Estimate: three arms x (load ~9 min per memory of the 35B's cycles + a CPU-prefill image turn of a few minutes) = ~40 min, 60 with a cold page cache. Queue:
#   python3 scripts/night.py add s6-moe-image --est 60 --by "nobara session, S6" --doc docs/tasks/task-multimodal-support-2026-10.md -- bash docs/measurements/multimodal-support-2026-10/run-s6-moe-night.sh
# Second night (2026-10-09): the first night's arms all declined the resident (cuMemAlloc out of memory: the 35B's int4 is 18.6 GB on an 8 GB card) because this script did not pass --moe-cache-experts, the C' expert streaming its header says it uses.
# Pinned binary: $BIN/serve-cuda (built from the rev in $BIN/serve-cuda.rev). The giw and the tower directory are on the NVMe (~/models), never the archive.
set -uo pipefail
SRC=$(cd "$(dirname "$0")/../../.." && pwd)
BIN=${BIN:-$HOME/goinfer-bench/s6}
OUT=${1:-$HOME/goinfer-logs/s6-moe-$(date +%F)}
GIW=$HOME/models/qwen3.6-35b-a3b-int4.giw
VIS=$HOME/models/qwen3.6-35b-a3b-vision
[ -x "$BIN/serve-cuda" ] || { echo "FATAL: $BIN/serve-cuda is missing"; exit 2; }
for f in "$GIW" "$VIS/config.json" "$SRC/testdata/qwen35vl_preprocess_image.png"; do [ -e "$f" ] || { echo "FATAL: $f is missing"; exit 2; }; done
mkdir -p "$OUT"
{ echo "binary: $BIN/serve-cuda (rev $(cat "$BIN/serve-cuda.rev" 2>/dev/null))"; echo "started: $(date '+%F %T %Z')"; echo "gpu: $(nvidia-smi --query-gpu=name,driver_version,memory.used --format=csv,noheader)"
  echo "load: $(cat /proc/loadavg)"; df -h ~ | tail -1; } | tee "$OUT/provenance.txt"
cd "$SRC" || exit 2
GS3C_IMAGE=testdata/qwen35vl_preprocess_image.png GS3C_SETTLE=30 GS3C_EXTRA="--moe-cache-experts --vision $VIS --kv-sessions 1 -ctx 4096" \
  bash docs/measurements/multimodal-support-2026-10/run-gs3c-served.sh "$BIN/serve-cuda" "$OUT/served" cuda:auto,=cuda:cpu,cuda:cpu "$GIW" > "$OUT/served.log" 2>&1
rc=$?
grep -E "arm |decode path|prefill path|vision|IDENTICAL|differing|top-3|near-tie|exited|\(.*s\):" "$OUT/served.log" | cut -c1-220
for lab in cuda-towerauto cuda cuda2; do
  f=$(ls "$OUT"/served/gs3c-*"-$lab.log" 2>/dev/null | head -1)
  [ -n "$f" ] && grep -q "decode path: cuda-resident" "$f" || { echo "!! arm $lab did not decode cuda-resident (or its log is missing): the reading is void"; rc=1; }
done
echo "finished: $(date '+%F %T %Z') rc=$rc" | tee -a "$OUT/provenance.txt"
exit $rc
