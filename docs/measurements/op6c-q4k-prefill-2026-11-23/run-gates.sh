#!/usr/bin/env bash
# OP6-C gates 2 and 3 (README.md), unattended: touched.py under the shared lock, the test binary built at the named commit, then
# gate 2 (bit identity) and gate 3 (TTFT A B B A) under the exclusive lock, each waiting for the lock's current holder.
# Usage: bash run-gates.sh <worktree> <out dir>
set -uo pipefail
WT=$1 OUT=$2
mkdir -p "$OUT"
export PATH=/usr/local/go/bin:$HOME/go/bin:$PATH
export TMPDIR=${TMPDIR:-$HOME/goinfer-bench/tmp/nobara-op6c}
cd "$WT"
echo "commit $(git rev-parse HEAD) start $(date -Is)" | tee "$OUT/run.txt"
python3 scripts/touched.py --wait 10800 > "$OUT/touched.log" 2>&1
if ! grep -q '^VERDICT: PASS' "$OUT/touched.log"; then
	echo "touched.py not PASS; stopping $(date -Is)" | tee -a "$OUT/run.txt"
	exit 1
fi
cd backend/cuda
python3 ../../scripts/timing_lock.py run --shared --wait 10800 --label op6c-build -- \
	go test -c -tags 'cuda goinfer_testhooks' -o "$OUT/cuda.test" . >> "$OUT/run.txt" 2>&1 || { echo "build failed" | tee -a "$OUT/run.txt"; exit 1; }
for g in TestOP6C_q4kPassBitIdentity TestOP6C_q4kPassSpeed; do
	echo "$g start $(date -Is)" | tee -a "$OUT/run.txt"
	GOINFER_HEAVY_TESTS=1 python3 ../../scripts/timing_lock.py run --wait 10800 --label "op6c-$g" -- \
		"$OUT/cuda.test" -test.run "^$g\$" -test.v -test.timeout 40m > "$OUT/$g.log" 2>&1
	rc=$?
	echo "$g exit $rc end $(date -Is)" | tee -a "$OUT/run.txt"
	if [ $rc -ne 0 ]; then exit $rc; fi
done
