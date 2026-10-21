#!/usr/bin/env bash
# RELEASING.md §C1: the T3 parity sweep for v0.23.0, as a night job on nobara-pc: `go run ./cmd/gate parity` with EMIT_MANIFEST=1 and the real checkpoints, pinned to
# REV (REQUIRED: the release-candidate SHA, never a default) in its own detached worktree so pushes during the night do not change what it measures.
#
# CHECKPOINTS FROM ~/models ONLY (owner instruction 2026-10-11; CLAUDE.md "Models: stored in the archive, benchmarked from local disk"). v0.20.0 and v0.21.0 read
# qwen3.6-35b-a3b, qwen3.8-27b, laguna-xs2 and llama4-scout-gguf from the /srv/models archive through the registry's env overrides; this script sets none, so those
# four are looked up under ~/models, where none of them is today (they are 219 GB against ~205 GB free). The sweep's own preflight names every checkpoint it cannot find
# and its required gates that need one SKIP and are reported as BLOCKERS: the verdict cannot read "ALL REQUIRED GATES GREEN" until they are present or the owner scopes
# the release's sweep (v0.20.0 and v0.21.0's scoped re-runs are the precedent). The morning read is therefore: which blockers are "checkpoint absent" and which are numeric.
#
# A fresh worktree has none of the main checkout's gitignored fixtures (~70 GB under testdata/ and decoder/testdata/), and without them most tiny-fixture gates skip.
# scripts/worktree_fixtures.sh places each ignored entry (its header has the rule): small fixtures COPIED as real files, since tests that copy a fixture
# directory keep only regular files and drop a symlinked one; larger ones cloned (reflinks), symlinks into ~/models kept as links.
#
# Estimate 160 minutes: the v0.21.0 sweep ran 2026-10-05 20:05 to 22:42 (157 min) with its realckpt cell cut off at its 120-minute timeout; v0.20.0's same cell took about
# 1h46. Loading from NVMe instead of the archive's SMR disk is faster, and four absent checkpoints skip their gates, so this should come in under that; 160 is the upper bound
# to plan the night around (night.py refuses jobs over 360).
#   python3 scripts/night.py add parity-v023 --est 160 --by "<who>, v0.23.0 C1" --doc RELEASING.md -- env REV=<sha> bash docs/measurements/release-v0.23.0/run-parity-sweep.sh
#
# Pass rule: the sweep's own verdict line, "ALL REQUIRED GATES GREEN". Then, by hand in the morning: commit the log under docs/measurements/release-v0.23.0/, promote every gate
# the sweep confirmed (go run ./cmd/gate ledger promote --gate <gate> --value PASS --by <you>) and merge the PARITY_ROW lines (RELEASING.md §C1, third obligation).
#
# DRY=1 stops after the worktree, the fixtures and the provenance block, before the sweep: the unattended-script check ("test it once on a fresh dir").
set -euo pipefail
REV=${REV:?REV is required: the release-candidate SHA, never a default}
SRC=${SRC:-$HOME/mycode/goinfer}   # the main checkout, whose gitignored fixtures the worktree gets
BASE=${BASE:-$HOME/goinfer-bench/release-v0.23.0}
WT=$BASE/wt
LOG=${LOG:-$HOME/goinfer-logs/parity-sweep-v023-$(date +%F)}
export PATH=/usr/local/go/bin:$PATH
case "$HOME $BASE $LOG" in *"/srv/models"*|*"/Volumes/"*) echo "FATAL: an archive path"; exit 2;; esac
if env | grep -E '^GOINFER_[A-Z0-9_]*=/srv/models|^GOINFER_[A-Z0-9_]*=/Volumes/' >/dev/null; then echo "FATAL: an archive override is set in the environment: this sweep reads ~/models only"; env | grep -E '^GOINFER_[A-Z0-9_]*=(/srv/models|/Volumes/)'; exit 2; fi
mkdir -p "$BASE" "$LOG"
cd "$SRC"
git fetch -q origin
if [ -d "$WT" ]; then git worktree remove --force "$WT"; fi
git worktree add -q --detach "$WT" "$REV"
fixtures=$(bash "$SRC/scripts/worktree_fixtures.sh" "$SRC" "$WT") || { echo "FATAL: placing the fixtures: $fixtures"; exit 2; }

cd "$WT"
{
  echo "rev:        $(git rev-parse HEAD)"
  echo "started:    $(date '+%F %T %Z')"
  echo "fixtures:   $fixtures"
  echo "tree:       $(git status --porcelain | wc -l) entries in git status (0 = clean)"
  echo "go:         $(go version)"
  echo "driver:     $(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null || echo none)"
  echo "models:      $HOME/models only (no archive override); absent checkpoints skip their gates and are reported as blockers"
} | tee "$LOG/provenance.txt"
[ "${DRY:-0}" = 1 ] && { echo "DRY=1: stopping before the sweep"; exit 0; }
# EMIT_MANIFEST defaults to 0 here: emitting rewrites testdata/parity_manifest.json in the worktree, and the sweep's JSON record (-record) reads a modified tree as DIRTY, which
# can never count (gate release row 11). The measured PARITY_ROW lines are in the log either way; merge them by hand in the morning (RELEASING.md §C1) or re-run with EMIT_MANIFEST=1.
# The record lands in the main checkout's docs/measurements/parity-sweep/ as an untracked file, to commit in the morning.
RECORD=${RECORD:-$SRC/docs/measurements/parity-sweep}
set +e   # the sweep exits 1 on a blocker; the rest of this script still reads its record and writes the provenance
GOWORK=off EMIT_MANIFEST=${EMIT_MANIFEST:-0} go run ./cmd/gate parity -logdir "$LOG" -record "$RECORD" 2>&1 | tee "$LOG/parity-sweep.log"
rc=${PIPESTATUS[0]}
ls "$RECORD"/*"$(git rev-parse --short=8 HEAD)"*.json >/dev/null 2>&1 || { echo "!! the sweep wrote no record for $REV under $RECORD"; rc=1; }
echo "sweep rc=$rc (the sweep's own verdict line decides pass: ALL REQUIRED GATES GREEN)" | tee -a "$LOG/provenance.txt"
echo "finished:   $(date '+%F %T %Z')" | tee -a "$LOG/provenance.txt"
exit $rc
