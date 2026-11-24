#!/bin/bash
# Builds the two BenchmarkResidentDecode test binaries for run_ab.py: A from BASE_WT (the base commit), B from NEW_WT (this change),
# each with the serve-auto scratch init (docs/measurements/phi3-serve-gap-2026-11-22/serve_auto_scratch_test.go.txt) placed as
# backend/cuda/zz_gap_serve_auto_test.go for the build and removed after it. Run under the shared timing lock.
#   bash build_bins.sh BASE_WT NEW_WT OUT_DIR
set -euo pipefail
export PATH=/usr/local/go/bin:$HOME/go/bin:$PATH
BASE=$1 NEW=$2 OUT=$3
mkdir -p "$OUT"
SCRATCH="$NEW/docs/measurements/phi3-serve-gap-2026-11-22/serve_auto_scratch_test.go.txt"
build() { # $1 tree, $2 arm
	local dst="$1/backend/cuda/zz_gap_serve_auto_test.go"
	cp "$SCRATCH" "$dst"
	(cd "$1/backend/cuda" && go test -c -tags 'cuda goinfer_testhooks' -o "$OUT/cuda-$2-$(git -C "$1" rev-parse --short=9 HEAD).test" .)
	rm -f "$dst"
}
build "$BASE" A
build "$NEW" B
ls -l "$OUT"
