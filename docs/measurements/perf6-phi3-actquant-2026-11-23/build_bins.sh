#!/bin/bash
# Builds perf 6's two backend/cuda test binaries for run_ab.py and the logits check: A from BASE_WT (the base commit), B from NEW_WT
# (this change), each with two scratch tests placed for the build and removed after it: the serve-auto init
# (docs/measurements/phi3-serve-gap-2026-11-22/serve_auto_scratch_test.go.txt) and the logits hash (logits_scratch_test.go.txt here).
# Run under the shared timing lock.
#   bash build_bins.sh BASE_WT NEW_WT OUT_DIR
set -euo pipefail
export PATH=/usr/local/go/bin:$HOME/go/bin:$PATH
BASE=$1 NEW=$2 OUT=$3
mkdir -p "$OUT"
AUTO="$NEW/docs/measurements/phi3-serve-gap-2026-11-22/serve_auto_scratch_test.go.txt"
HASH="$NEW/docs/measurements/perf6-phi3-actquant-2026-11-23/logits_scratch_test.go.txt"
build() { # $1 tree, $2 arm
	local d="$1/backend/cuda"
	cp "$AUTO" "$d/zz_gap_serve_auto_test.go"
	cp "$HASH" "$d/zz_perf6_logits_test.go"
	(cd "$d" && go test -c -tags 'cuda goinfer_testhooks' -o "$OUT/cuda-$2-$(git -C "$1" rev-parse --short=9 HEAD).test" .)
	rm -f "$d/zz_gap_serve_auto_test.go" "$d/zz_perf6_logits_test.go"
}
build "$BASE" A
build "$NEW" B
ls -l "$OUT"
