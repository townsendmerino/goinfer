#!/usr/bin/env bash
# G-S5b (docs/tasks/task-multimodal-support-2026-10.md, S5): Gemma 4 E2B on an audio prompt, goinfer's float32 CPU
# prefill against transformers' Gemma4ForConditionalGeneration in float32. Both need E2B in float32 (about 20 GB each,
# one after the other), so this runs on nobara, at night. Three steps (decoder/gemma4_audio_e2b_real_test.go's header):
# goinfer writes the prompt ids, HF scores them, goinfer scores them and grades, planted defects included.
#
# Pinned: $BIN/decoder-s5.test, built from this worktree (rev in decoder-s5.rev); E2B from ~/models/gemma-4-E2B-unq;
# transformers from ~/.venv-vl. Run from the worktree the binary was built in. Estimate: ~10 min; queued at 20.
#   python3 scripts/night.py add s5b-e2b-audio --est 20 --by "Claude (Mac), S5" --doc docs/tasks/task-multimodal-support-2026-10.md -- bash -c 'cd ~/wt/goinfer-s5 && bash docs/measurements/multimodal-support-2026-10/run-s5b-nobara.sh'
set -uo pipefail
SRC=$(cd "$(dirname "$0")/../../.." && pwd)
BIN=${BIN:-$HOME/goinfer-bench/s5}
OUT=${1:-$HOME/goinfer-bench/s5/s5b-$(date +%F)}
PY=${PY:-$HOME/.venv-vl/bin/python}
E2B=$HOME/models/gemma-4-E2B-unq
[ -x "$BIN/decoder-s5.test" ] || { echo "FATAL: $BIN/decoder-s5.test is missing"; exit 2; }
[ -d "$E2B" ] || { echo "FATAL: $E2B is missing"; exit 2; }
[ -x "$PY" ] || { echo "FATAL: $PY is missing"; exit 2; }
mkdir -p "$OUT"
{ echo "binary:   $BIN/decoder-s5.test (rev $(cat "$BIN/decoder-s5.rev" 2>/dev/null))"; echo "started:  $(date '+%F %T %Z')"
  "$PY" -c 'import transformers, torch; print("transformers", transformers.__version__, "torch", torch.__version__)'; } | tee "$OUT/provenance.txt"
cd "$SRC/decoder" || exit 2
run() { GOINFER_HEAVY_TESTS=1 GOINFER_GEMMA4_E2B="$E2B" GOINFER_S5B_OUT="$OUT" GOINFER_S5B_STEP="$1" \
  "$BIN/decoder-s5.test" -test.run '^TestGemma4E2BAudioFull$' -test.v -test.timeout 60m; }
echo "=== 1. ids $(date '+%T')"
run ids > "$OUT/1-ids.log" 2>&1 || { tail -5 "$OUT/1-ids.log"; exit 1; }
grep -E "G-S5b|^--- " "$OUT/1-ids.log"
echo "=== 2. HF reference $(date '+%T')"
( cd "$SRC" && "$PY" scripts/pin_gemma4_e2b_audio_full.py --model "$E2B" --wav testdata/embeddinggemma2-audio/mid.wav --out "$OUT" ) \
  > "$OUT/2-hf.log" 2>&1 || { tail -5 "$OUT/2-hf.log"; exit 1; }
tail -2 "$OUT/2-hf.log"
echo "=== 3. goinfer, graded $(date '+%T')"
run compare > "$OUT/3-compare.log" 2>&1
rc=$?
grep -E "G-S5b|^--- |_test.go:[0-9]+" "$OUT/3-compare.log"
grep -q -- "--- PASS: TestGemma4E2BAudioFull" "$OUT/3-compare.log" || { echo "!! G-S5b did not PASS (a failure or a skip)"; rc=1; }
echo "finished: $(date '+%F %T %Z') rc=$rc" | tee -a "$OUT/provenance.txt"
exit $rc
