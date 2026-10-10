#!/usr/bin/env bash
# G-S18g2 (docs/tasks/task-multimodal-support-2026-10.md, "G-S18g2", registered before this run): Gemma 3 4B's int8 Metal
# SigLIP tower against the f16 Metal tower, each measured against the exact CPU float32 tower. It replaces G-S18g, whose
# near-tie rule the f16 tower itself fails.
#   per image  metal.test TestGemma3TowerDump: the CPU, f16 Metal and int8 Metal towers' projected features (one file)
#   per unit   decoder.test TestGemma3TowerSensitivity: the int4 decoder on the CPU, teacher-forced along the CPU tower's
#              own greedy reply to its end of turn (32 tokens at most); mean KL and argmax agreement per arm; the two
#              instrument controls; noise arms at each tower's size (three seeds)
#   grade      gs18g2_grade.py: the paired excess KL and the agreement difference, a cluster bootstrap over images,
#              the registered bands
# 12 images x 2 prompts = 24 units. Pinned in $BIN: metal.test (-tags goinfer_testhooks), decoder.test (-tags realckpt),
# gs18g2_grade.py, rev; the tests run from the worktree $SRC at that rev (they read ../testdata). The checkpoint is
# ~/models/gemma-3-4b-it through its sidecar (local disk, never the archive). A correctness gate, not timed.
# Estimate ~35 min (a dump ~40 s per image, a unit ~65 s); queue at 60:
#   python3 scripts/night.py add gs18g2 --est 60 --by "Claude, S18 G-S18g2" --doc docs/tasks/task-multimodal-support-2026-10.md -- bash <this script>
# GS18G2_DRY=1 checks the preconditions and prints the plan.
set -uo pipefail
BIN=${BIN:-$HOME/goinfer-bench/gs18g2}
SRC=${SRC:-$BIN/wt}
OUT=${1:-$HOME/goinfer-logs/gs18g2-$(date +%F)}
DIR=$HOME/models/gemma-3-4b-it
IMAGES=(glm_ocr/table.png glm_ocr/formula.png gemma3_preprocess_image.png qwen25vl_preprocess_image.png
  qwen25vl_preprocess_image_resize.png qwen35vl_preprocess_image.png glm_ocr/invoice.png glm_ocr/invoices/inv01.png
  glm_ocr/invoices/inv02.png glm_ocr/invoices/inv03.png glm_ocr/invoices/inv04.png glm_ocr/invoices/inv05.png)
PROMPTS=("What does this image show? Answer briefly." "Describe this image in one sentence.")
fatal() { echo "FATAL: $*" >&2; exit 2; }
for f in metal.test decoder.test gs18g2_grade.py rev; do [ -e "$BIN/$f" ] || fatal "$BIN/$f is missing"; done
[ -d "$SRC/metal" ] && [ -d "$SRC/decoder" ] || fatal "no worktree at $SRC"
for i in "${IMAGES[@]}"; do [ -e "$SRC/testdata/$i" ] || fatal "$SRC/testdata/$i is missing"; done
for p in "$DIR/config.json" "$DIR.int4.metal.giw"; do [ -e "$p" ] || fatal "$p is missing (the decoder loads through the sidecar)"; done
case "$DIR$OUT" in */Volumes/*|*/srv/models*) fatal "the archive is not a read path (CLAUDE.md)";; esac
free_kb=$(df -k "$HOME" | tail -1 | awk '{print $4}'); [ "$free_kb" -ge 1500000 ] || fatal "under 1.5 GB free on the home volume"
units=$(( ${#IMAGES[@]} * ${#PROMPTS[@]} ))
[ -n "${GS18G2_DRY:-}" ] && { echo "DRY: preconditions hold; plan = ${#IMAGES[@]} images x ${#PROMPTS[@]} prompts = $units units, binaries at $(cat "$BIN/rev")"; exit 0; }
mkdir -p "$OUT"
: > "$OUT/units.jsonl"
{ echo "binaries: $BIN (rev $(cat "$BIN/rev"))"; echo "started: $(date '+%F %T %Z')"; sw_vers | tr '\n' ' '; echo
  pmset -g batt | head -1; sysctl -n vm.swapusage; uptime; } | tee "$OUT/provenance.txt"
t0=$SECONDS done_units=0 failed=0
for img in "${IMAGES[@]}"; do
  feats=$OUT/feats.json
  echo "[$(date +%T)] dump $img"
  ( cd "$SRC/metal" && GOINFER_HEAVY_TESTS=1 GOINFER_G3_FEATS="$feats" GOINFER_G3_IMAGE="$img" \
      "$BIN/metal.test" -test.run '^TestGemma3TowerDump$' -test.v -test.count=1 -test.timeout 15m ) >> "$OUT/dump.log" 2>&1
  if [ $? -ne 0 ] || [ ! -s "$feats" ]; then
    echo "!! the dump failed for $img (see $OUT/dump.log)"; failed=$((failed + 1)); rm -f "$feats"; continue
  fi
  grep "g3 dump\] $img" "$OUT/dump.log" | tail -2 | cut -c1-200 | sed 's/^/  /'
  for prompt in "${PROMPTS[@]}"; do
    ( cd "$SRC/decoder" && GOINFER_HEAVY_TESTS=1 GOINFER_G3_FEATS="$feats" GOINFER_G3_IMAGE="$img" GOINFER_G3_PROMPT="$prompt" \
        GOINFER_G3_OUT="$OUT/units.jsonl" GOINFER_GEMMA3_4B="$DIR" \
        "$BIN/decoder.test" -test.run '^TestGemma3TowerSensitivity$' -test.v -test.count=1 -test.timeout 20m ) >> "$OUT/units.log" 2>&1
    if [ $? -ne 0 ]; then echo "!! the unit failed: $img / $prompt (see $OUT/units.log)"; failed=$((failed + 1)); fi
    done_units=$((done_units + 1))
    echo "[$(date +%T)] unit $done_units/$units done, $((SECONDS - t0))s elapsed: $img / ${prompt:0:24}"
  done
  rm -f "$feats"
done
echo "[$(date +%T)] $done_units units run, $failed failure(s); grading"
python3 "$BIN/gs18g2_grade.py" "$OUT/units.jsonl" --expect-units "$units" | tee "$OUT/verdict.txt"
rc=${PIPESTATUS[0]}
echo "finished: $(date '+%F %T %Z') rc=$rc" | tee -a "$OUT/provenance.txt"
exit "$rc"
