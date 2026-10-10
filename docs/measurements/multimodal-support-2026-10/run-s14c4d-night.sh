#!/usr/bin/env bash
# G-S14c4d (docs/tasks/task-multimodal-support-2026-10.md, "G-S14c4d", registered before this code): Qwen3-ASR-0.6B with the assistant turn opened with the one token "language", against float32,
# on the 73 LibriSpeech dummy clips. The goinfer side only: the reference R is the 2026-10-09 transformers output (a pure function of the same audio and weights; refs.json hash checked below), and the
# Gemma record arm is not repeated.
#   1. goinfer   decoder TestQwen3ASRWER_arms with GOINFER_S14C4_PF=1: f32, int4, int4h8 (the unforced control) and f32pf, int4pf, int4h8pf     -> go.json
#   2. grade     scripts/s14c4d_grade.py: G-S14c4d-a, G-S14c4d-b, the first-token classes, and the byte comparison with 2026-10-09's unforced arms -> summary.txt
# Pinned in $BIN: decoder.test (realckpt, rev beside it), s14c4d_grade.py, hf-2026-10-09.json, go-2026-10-09.json, refs.sha256. Checkpoint from ~/models (local NVMe), never the archive. Correctness gate, not timed.
# Estimate ~15 min (the 2026-10-09 Go side was 510 s for three arms); queue at 30:
#   python3 scripts/night.py add s14c4d --est 30 --by "nobara session, S14 G-S14c4d" --doc docs/tasks/task-multimodal-support-2026-10.md -- bash <pinned>/run-s14c4d-night.sh
# S14C4D_DRY=1 checks the preconditions and prints the plan.
set -uo pipefail
BIN=${BIN:-$HOME/goinfer-bench/s14c4d}
OUT=${1:-$HOME/goinfer-logs/s14c4d-$(date +%F)}
DATA=$HOME/goinfer-bench/librispeech-dummy
Q3=$HOME/models/qwen3-asr-0.6b
fatal() { echo "FATAL: $*" >&2; exit 2; }
for f in decoder.test s14c4d_grade.py hf-2026-10-09.json go-2026-10-09.json refs.sha256 rev; do [ -e "$BIN/$f" ] || fatal "$BIN/$f is missing"; done
for f in "$Q3/model.safetensors" "$Q3/tokenizer.json" "$DATA/refs.json"; do [ -e "$f" ] || fatal "$f is missing"; done
case "$OUT$Q3" in *"/srv/models"*|*"/Volumes/"*) fatal "an archive path";; esac
[ "$(sha256sum "$DATA/refs.json" | cut -d' ' -f1)" = "$(cut -d' ' -f1 "$BIN/refs.sha256")" ] || fatal "refs.json is not the one the 2026-10-09 reference was made on: R cannot be reused"
mkdir -p "$OUT"; SUM="$OUT/summary.txt"; : > "$SUM"
{ echo "binaries: $BIN (rev $(cat "$BIN/rev"))"; echo "started: $(date '+%F %T %Z')"; echo "load: $(cat /proc/loadavg)"; free -g | sed -n 2p; } | tee "$OUT/provenance.txt"
[ -n "${S14C4D_DRY:-}" ] && { echo "DRY: preconditions hold; plan = goinfer six arms (GOINFER_S14C4_PF=1), grade"; exit 0; }
t0=$SECONDS
( while sleep 60; do echo "[$(date +%T)] ... goinfer running $((SECONDS - t0))s; last: $(tail -n1 "$OUT/goinfer.log" | cut -c1-120)"; done ) &
tick=$!
( cd "$BIN" && env GOINFER_HEAVY_TESTS=1 GOINFER_S14C4_PF=1 GOINFER_S14C4_DIR="$DATA" GOINFER_S14C4_OUT="$OUT/go.json" GOINFER_QWEN3ASR_DIR="$Q3" ./decoder.test -test.run '^TestQwen3ASRWER_arms$' -test.v -test.timeout 90m ) > "$OUT/goinfer.log" 2>&1
rc=$?; kill "$tick" 2>/dev/null; wait "$tick" 2>/dev/null
printf '%-10s rc=%-3s %5ss\n' goinfer "$rc" "$((SECONDS - t0))" | tee -a "$SUM"
if [ -e "$OUT/go.json" ]; then python3 "$BIN/s14c4d_grade.py" "$DATA/refs.json" "$BIN/hf-2026-10-09.json" "$OUT/go.json" "$BIN/go-2026-10-09.json" 2>&1 | tee -a "$SUM"
else echo "!! not graded: go.json is missing (see $OUT/goinfer.log)" | tee -a "$SUM"; fi
echo "finished: $(date '+%F %T %Z')" | tee -a "$OUT/provenance.txt"
exit 0
