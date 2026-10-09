#!/usr/bin/env bash
# Step (b') of the Gemma 4 31B work (docs/tasks/task-multimodal-support-2026-10.md, "Gemma 4 31B, step (b')", registered before its code): goinfer's CPU int8int8 and int4 arms against a layer-streaming Hugging Face float32
# reference on the 12 G-31a prompts, the plain-bf16 E4B as the scale.
#   0. controls   scripts/g31b_controls.py on the 31B-shaped tiny (streaming against ordinary, six planted defects red, determinism); a failed control aborts the job: no reading is made
#   1. E4B        goinfer arms (decoder TestG31b_goinferLogits) and Hugging Face float32 ordinary, along G-31a's E4B path (g31a/qc-e4b.json, pinned)
#   2. 31B        the int8int8 bundle (prequant, 33 GB, deleted after), goinfer arms along g31a/qc-31b.json, and the layer-streaming reference (59 GB read once)
#   3. grade      scripts/g31b_grade.py: agreement and KL per arm, the bootstrap, the registered rule
# Pinned in $BIN (main at the rev in $BIN/rev): decoder.test (realckpt), prequant, g31b_hf_stream.py, g31b_grade.py, g31b_controls.py. Checkpoints from ~/models (local NVMe), never the archive.
# Estimate ~110 min (streaming ~20, E4B ordinary ~5, goinfer teacher-forced arms ~45, the int8int8 bundle ~7, loads); queue at 150:
#   python3 scripts/night.py add g31b --est 150 --by "nobara session, 31B (b')" --doc docs/tasks/task-multimodal-support-2026-10.md -- bash docs/measurements/multimodal-support-2026-10/run-g31b-night.sh
# G31B_DRY=1 checks the preconditions and prints the plan. A step that fails is recorded and the later ones still run where they can; a missing output makes the grade refuse.
set -uo pipefail
SRC=$(cd "$(dirname "$0")/../../.." && pwd)
BIN=${BIN:-$HOME/goinfer-bench/g31b}
OUT=${1:-$HOME/goinfer-logs/g31b-$(date +%F)}
PY=$HOME/g4venv/bin/python
DIR31=$HOME/models/gemma-4-31B-it
DIRE4=$HOME/models/gemma-4-E4B-it
GIW4=$HOME/models/gemma-4-31B-it.int4.cpu-amd64.giw
GIW8=$HOME/models/gemma-4-31B-it.int8int8.cpu-amd64.giw
REC=$SRC/docs/measurements/multimodal-support-2026-10/g31a
fatal() { echo "FATAL: $*" >&2; exit 2; }
for f in decoder.test prequant g31b_hf_stream.py g31b_grade.py g31b_controls.py; do [ -e "$BIN/$f" ] || fatal "$BIN/$f is missing"; done
for f in "$DIR31/config.json" "$DIRE4/config.json" "$GIW4" "$REC/qc-31b.json" "$REC/qc-e4b.json" "$SRC/testdata/gemma4-dense-twogeom-tiny/config.json"; do [ -e "$f" ] || fatal "$f is missing"; done
[ -x "$PY" ] || fatal "$PY is missing"
for p in "$DIR31" "$DIRE4" "$OUT"; do case "$p" in /srv/models*|/Volumes/*) fatal "$p is the archive (CLAUDE.md)";; esac; done
sha=$(sha256sum "$REC/qc-31b.json" | cut -c1-16); [ "$sha" = "660afa59d8362374" ] || fatal "qc-31b.json is not the pinned G-31a file (sha256 $sha)"
sha=$(sha256sum "$REC/qc-e4b.json" | cut -c1-16); [ "$sha" = "4d21d193203c1d2d" ] || fatal "qc-e4b.json is not the pinned G-31a file (sha256 $sha)"
free_gb=$(df -BG --output=avail "$HOME/models" | tail -1 | tr -dc '0-9'); need=45; [ "$free_gb" -ge "$need" ] || fatal "only ${free_gb} GB free under ~/models, need $need (the int8int8 bundle is ~33 GB, deleted after)"
mkdir -p "$OUT"
SUM="$OUT/summary.txt"; : > "$SUM"
{ echo "binaries: $BIN (main $(cat "$BIN/rev" 2>/dev/null))"; echo "started: $(date '+%F %T %Z')"; echo "free: ${free_gb} GB; load: $(cat /proc/loadavg)"; free -g | sed -n 2p; } | tee "$OUT/provenance.txt"
[ -n "${G31B_DRY:-}" ] && { echo "DRY: preconditions hold; plan = controls, E4B (goinfer + HF ordinary), 31B (bundle, goinfer, HF streaming), grade"; exit 0; }
step() { local name=$1; shift; local t0=$SECONDS rc
  echo "[$(date +%T)] START $name"
  ( while sleep 60; do echo "[$(date +%T)] ... $name running $((SECONDS - t0))s; last: $(tail -n1 "$OUT/$name.log" | cut -c1-120)"; done ) &
  local tick=$!
  "$@" > "$OUT/$name.log" 2>&1; rc=$?
  kill "$tick" 2>/dev/null; wait "$tick" 2>/dev/null
  printf '%-18s rc=%-3s %5ss\n' "$name" "$rc" "$((SECONDS - t0))" | tee -a "$SUM"; return $rc; }
gotest() { ( cd "$SRC/decoder" && env GOINFER_HEAVY_TESTS=1 "$@" "$BIN/decoder.test" -test.run '^TestG31b_goinferLogits$' -test.v -test.timeout 150m ); }
hf() { "$PY" -I "$BIN/g31b_hf_stream.py" "$@"; }

step controls "$PY" -I "$BIN/g31b_controls.py" "$SRC/testdata/gemma4-dense-twogeom-tiny" || { echo "!! the controls did not hold: no reading is made" | tee -a "$SUM"; tail -12 "$OUT/controls.log" | tee -a "$SUM"; exit 1; }
grep -E "^control|ALL CONTROLS" "$OUT/controls.log" | tee -a "$SUM"

[ -e "$OUT/e4b.int4.f32" ] || step goinfer-e4b gotest GOINFER_G31B_DIR="$DIRE4" GOINFER_G31B_PATHS="$REC/qc-e4b.json" GOINFER_G31B_OUT="$OUT/e4b" || echo "!! the E4B goinfer arms failed" | tee -a "$SUM"
[ -e "$OUT/e4b.hf.f32" ] || step hf-e4b hf "$DIRE4" "$OUT/e4b.seqs.json" "$OUT/e4b.hf" ordinary || echo "!! the E4B Hugging Face reference failed" | tee -a "$SUM"

if [ -e "$GIW8" ]; then echo "exists: $GIW8" | tee -a "$SUM"
else step prequant-int8 "$BIN/prequant" -quant int8int8 -target cpu-amd64 -o "$GIW8" "$DIR31" || { echo "!! the int8int8 bundle failed; removing a partial file" | tee -a "$SUM"; rm -f "$GIW8"; }; fi
[ -e "$OUT/b31.int4.f32" ] || step goinfer-31b gotest GOINFER_G31B_DIR="$DIR31" GOINFER_G31B_PATHS="$REC/qc-31b.json" GOINFER_G31B_OUT="$OUT/b31" GOINFER_G31B_INT8="$GIW8" GOINFER_G31B_INT4="$GIW4" || echo "!! the 31B goinfer arms failed" | tee -a "$SUM"
[ -e "$OUT/b31.hf.f32" ] || step hf-31b hf "$DIR31" "$OUT/b31.seqs.json" "$OUT/b31.hf" stream || echo "!! the 31B streaming reference failed" | tee -a "$SUM"
rm -f "$GIW8" && echo "deleted the int8int8 bundle (the int4 bundle stays)" | tee -a "$SUM"

for f in b31.int8int8.f32 b31.int4.f32 b31.hf.f32 e4b.int8int8.f32 e4b.int4.f32 e4b.hf.f32; do [ -e "$OUT/$f" ] || { echo "!! $f is missing: no grade" | tee -a "$SUM"; exit 1; }; done
step grade "$PY" -I "$BIN/g31b_grade.py" "$OUT/b31" "$OUT/e4b" || echo "!! the grade failed" | tee -a "$SUM"
cat "$OUT/grade.log" | tee -a "$SUM"
echo "finished: $(date '+%F %T %Z')" | tee -a "$OUT/provenance.txt"
exit 0
