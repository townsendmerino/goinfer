#!/usr/bin/env bash
# RELEASING.md §C1: the T3 parity sweep for v0.20.0, as a night job on nobara-pc. Same invocation as
# v0.19.0's (go run ./cmd/gate parity, EMIT_MANIFEST=1, real checkpoints, 120m cell timeout), pinned
# to REV in its own worktree so pushes during the night do not change what it measures.
#
# A fresh worktree has none of the main checkout's gitignored fixtures (~70 GB under testdata/ and
# decoder/testdata/), and without them most tiny-fixture gates skip. Each ignored entry is symlinked
# in from the main checkout instead of copied.
#
# Run 1 (0ca36756, 2026-09-30 19:22-20:24) FAILED with 17 blockers: 15 required gates SKIPPED because their
# checkpoints had been moved from ~/models to the /srv/models archive, and three asset-only tests were stale (fixed in
# bcf50a49). The checkpoints were copied back (owner OK) except the two largest, which nobara's free disk cannot also
# hold; those two are read from the archive through the asset registry's env override (an explicit env path always
# wins). That is a correctness sweep reading a checkpoint, not a timed measurement, so the archive rule
# (docs/benchmarks.md, "Model storage") does not apply; it only makes those loads slower.
export GOINFER_QWEN35_REAL=${GOINFER_QWEN35_REAL:-/srv/models/qwen3.6-35b-a3b}
export GOINFER_QWEN38=${GOINFER_QWEN38:-/srv/models/qwen3.8-27b}
#
# Pass rule: the sweep's own verdict line, "ALL REQUIRED GATES GREEN". Then, by hand in the morning:
# promote what it confirmed (gate ledger promote) and merge the PARITY_ROW lines (RELEASING.md §C1).
set -euo pipefail
REV=${REV:-bcf50a49}
SRC=$HOME/mycode/goinfer
BASE=$HOME/goinfer-bench/release-v0.20.0
WT=$BASE/wt
LOG=$HOME/goinfer-logs/parity-sweep-v020-$(date +%F)
export PATH=/usr/local/go/bin:$PATH
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
  echo "archive:    GOINFER_QWEN35_REAL=$GOINFER_QWEN35_REAL GOINFER_QWEN38=$GOINFER_QWEN38"
} | tee "$LOG/provenance.txt"
GOWORK=off EMIT_MANIFEST=1 go run ./cmd/gate parity 2>&1 | tee "$LOG/parity-sweep.log"
echo "finished:   $(date '+%F %T %Z')" | tee -a "$LOG/provenance.txt"
