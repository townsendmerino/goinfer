#!/usr/bin/env bash
# Print "== TESTS: P passed, S skipped, F failed (top-level) ==" for a `go test -json` file, on a lane with no python on PATH (the Windows lanes):
# the same count scripts/ci_test_summary.py prints for the Linux shards. A number people can see, not a gate: exits 1 only when the file holds no result at all.
#   bash scripts/ci_skip_count.sh test.json
set -u
f=${1:?usage: ci_skip_count.sh test.json}
count() { grep -c "\"Action\":\"$1\",\"Package\":\"[^\"]*\",\"Test\":\"[^\"/]*\"" "$f" || true; }
p=$(count pass); s=$(count skip); x=$(count fail)
line="== TESTS: $p passed, $s skipped, $x failed (top-level) =="
echo "$line"
[ -n "${GITHUB_STEP_SUMMARY:-}" ] && echo "$line" >> "$GITHUB_STEP_SUMMARY"
[ $((p + s + x)) -gt 0 ] || { echo "::error::$f holds no test result: the capture recorded nothing"; exit 1; }
