#!/usr/bin/env bash
# RELEASING.md §C1: the T3 parity sweep for v0.21.0, as a night job on nobara-pc. The same invocation as v0.20.0's (go run ./cmd/gate parity,
# EMIT_MANIFEST=1, real checkpoints, 120m cell timeout), pinned to REV in its own worktree so pushes during the night do not change what it measures.
#
# A fresh worktree has none of the main checkout's gitignored fixtures (~70 GB under testdata/ and decoder/testdata/), and without them most tiny-fixture
# gates skip. Each ignored entry is symlinked in from the main checkout instead of copied.
#
# CHECKPOINTS NOT IN ~/models (read from the /srv/models archive through the registry's env override; an explicit env path always wins). This is a
# correctness sweep reading a checkpoint, not a timed measurement, so the archive rule (docs/benchmarks.md, "Model storage") does not apply; the loads are
# just slower (a 5400 rpm SMR disk). nobara's free disk (about 70 GB) cannot hold them locally:
#   qwen3.6-35b-a3b, qwen3.8-27b   (as in v0.20.0)
#   laguna-xs2 (63 GB), llama4-scout-gguf (37 GB)   moved to the archive 2026-10-02, AFTER v0.20.0's sweep, where both ran. Without these overrides their
#   required gates would skip and be reported as blockers.
# Deliberately NOT overridden: the four Nemotron checkpoints (nemotron-hf 17 GB, nemotron3nano-30b-bf16 59 GB, nemotron3nano-gguf 23 GB, and
# nemotron35lightning, which is not on the archive at all) and qwen3vl-2b-instruct (not on the archive). v0.20.0's sweep also had these five NOT FOUND, so their
# gates skipped then too; overriding the three that exist would run gates this release line has never run (first-run gates on a 59 GB bf16 model on the CPU),
# which is a different decision from re-validating what was validated. The sweep's preflight names them again.
#
# Pass rule: the sweep's own verdict line, "ALL REQUIRED GATES GREEN". Then, by hand in the morning: promote what it confirmed (gate ledger promote) and
# merge the PARITY_ROW lines (RELEASING.md §C1).
#
# DRY=1 stops after the worktree, the fixture links and the provenance block, before the sweep: the unattended-script check ("test it once on a fresh dir").
export GOINFER_QWEN35_REAL=${GOINFER_QWEN35_REAL:-/srv/models/qwen3.6-35b-a3b}
export GOINFER_QWEN38=${GOINFER_QWEN38:-/srv/models/qwen3.8-27b}
export GOINFER_LAGUNA_XS2=${GOINFER_LAGUNA_XS2:-/srv/models/laguna-xs2}
export GOINFER_LLAMA4_GGUF=${GOINFER_LLAMA4_GGUF:-/srv/models/llama4-scout-gguf/Llama-4-Scout-17B-16E-Instruct-Q2_K.gguf}
set -euo pipefail
REV=${REV:-7b50947a}
SRC=$HOME/mycode/goinfer
BASE=${BASE:-$HOME/goinfer-bench/release-v0.21.0}
WT=$BASE/wt
LOG=${LOG:-$HOME/goinfer-logs/parity-sweep-v021-$(date +%F)}
export PATH=/usr/local/go/bin:$PATH
for p in "$GOINFER_QWEN35_REAL" "$GOINFER_QWEN38" "$GOINFER_LAGUNA_XS2" "$GOINFER_LLAMA4_GGUF"; do
  [ -e "$p" ] || { echo "FATAL: $p is missing (an archive override); its required gates would skip"; exit 2; }
done
mkdir -p "$BASE" "$LOG"
cd "$SRC"
git fetch -q origin
if [ -d "$WT" ]; then git worktree remove --force "$WT"; fi
git worktree add -q --detach "$WT" "$REV"
n=0
for d in testdata decoder/testdata; do
  while IFS= read -r p; do
    p=${p%/}
    [ -e "$WT/$p" ] && continue
    if [ -d "$SRC/$p" ]; then
      # a real directory holding symlinks, not a symlink to the directory: an ignore rule written for a directory
      # (decoder/testdata/.laguna-code-*/) does not match a symlink, which would leave the worktree dirty
      mkdir -p "$WT/$p"
      for c in "$SRC/$p"/* "$SRC/$p"/.[!.]*; do [ -e "$c" ] && ln -s "$c" "$WT/$p/"; done
    else
      mkdir -p "$(dirname "$WT/$p")"
      ln -s "$SRC/$p" "$WT/$p"
    fi
    n=$((n + 1))
  done < <(git ls-files --others --ignored --exclude-standard --directory "$d")
done

cd "$WT"
{
  echo "rev:        $(git rev-parse HEAD)"
  echo "started:    $(date '+%F %T %Z')"
  echo "fixtures:   $n gitignored entries symlinked from $SRC"
  echo "tree:       $(git status --porcelain | wc -l) entries in git status (0 = clean)"
  echo "go:         $(go version)"
  echo "driver:     $(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null || echo none)"
  echo "archive:    GOINFER_QWEN35_REAL=$GOINFER_QWEN35_REAL GOINFER_QWEN38=$GOINFER_QWEN38 GOINFER_LAGUNA_XS2=$GOINFER_LAGUNA_XS2 GOINFER_LLAMA4_GGUF=$GOINFER_LLAMA4_GGUF"
} | tee "$LOG/provenance.txt"
[ "${DRY:-0}" = 1 ] && { echo "DRY=1: stopping before the sweep"; exit 0; }
GOWORK=off EMIT_MANIFEST=1 go run ./cmd/gate parity 2>&1 | tee "$LOG/parity-sweep.log"
echo "finished:   $(date '+%F %T %Z')" | tee -a "$LOG/provenance.txt"
