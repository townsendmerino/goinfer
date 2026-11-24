#!/bin/bash
# perf 13 attempt 2 (B3): tier-0 tests under the shared lock, then build B3's binary, then the A/B against the attempt-1 base binary.
set -uo pipefail
export PATH=/usr/local/go/bin:$HOME/go/bin:$PATH
export TMPDIR=/home/francis/goinfer-bench/tmp/nobara-cuda-small
WT=/home/francis/goinfer-bench/cuda-small-wt
OUT=/home/francis/goinfer-bench/perf13-ab
cd "$WT/backend/cuda"
echo "$(date -Is) tests: waiting for the shared lock"
python3 ../../scripts/timing_lock.py run --shared --wait 7200 --label cuda-small-b3-tests -- \
	go test -tags 'cuda goinfer_testhooks' -count=3 -timeout 12m \
	-run 'TestCost_cudaDecodeExact$|TestLaunchToken_queuedRowsReadTheirOwnEmbedding|TestGraphsSafeGate|TestGemma4EModel_cudaResidentParity|TestGemma4Graphs_bitExact_emodel' -v . >"$OUT/b3-tests.log" 2>&1
rc=$?
grep -E "^(--- |ok|FAIL)" "$OUT/b3-tests.log" | sort | uniq -c
if [ $rc -ne 0 ] || grep -q -- "--- FAIL" "$OUT/b3-tests.log" || ! grep -q -- "--- PASS: TestLaunchToken_queuedRowsReadTheirOwnEmbedding" "$OUT/b3-tests.log"; then
	echo "$(date -Is) B3-TESTS-FAILED rc=$rc"
	exit 1
fi
echo "$(date -Is) tests passed; building B3"
SCRATCH="$WT/docs/measurements/phi3-serve-gap-2026-11-22/serve_auto_scratch_test.go.txt"
cp "$SCRATCH" zz_gap_serve_auto_test.go
python3 ../../scripts/timing_lock.py run --shared --wait 7200 --label cuda-small-b3-build -- \
	go test -c -tags 'cuda goinfer_testhooks' -o "$OUT/bin/cuda-B3-$(git -C "$WT" rev-parse --short=9 HEAD).test" .
brc=$?
rm -f zz_gap_serve_auto_test.go
[ $brc -eq 0 ] || { echo "$(date -Is) B3-BUILD-FAILED"; exit 1; }
mkdir -p "$OUT/run2"
echo "$(date -Is) A/B: waiting for the exclusive lock"
cd "$WT"
python3 scripts/timing_lock.py run --wait 7200 --label perf13-ab2 -- timeout 90m python3 -u docs/measurements/perf13-emb-upload-2026-11-23/run_ab.py \
	--a "$OUT/bin/cuda-A-26a73a8b0.test" --b "$OUT/bin/cuda-B3-$(git -C "$WT" rev-parse --short=9 HEAD).test" --out "$OUT/run2" >"$OUT/run2/run.log" 2>&1
echo "$(date -Is) A/B rc=$?"
tail -1 "$OUT/run2/run.log"
