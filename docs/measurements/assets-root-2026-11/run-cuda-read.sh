#!/usr/bin/env bash
# The first real run of the backend/cuda tests that the asset-registry fix un-skips. One go test process per model family
# (the heavy tier's rule), each with its own -timeout; the sum of the timeouts is the 2 h cap. Logs: reading/<group>.log.
HERE="$(cd "$(dirname "$0")" && pwd)"
# A timed run holds the exclusive timing lock: re-run this script under it unless an ancestor already holds it.
if [ -z "${GOINFER_TIMING_LOCK:-}" ]; then
  exec python3 "$HERE/../../../scripts/timing_lock.py" run --wait 600 --label assets-root-cuda-read -- bash "$HERE/run-cuda-read.sh"
fi
mkdir -p "$HERE/reading" || exit 1
OUT="$HERE/reading"
cd "$HERE/../../../backend/cuda" || exit 1
export GOINFER_HEAVY_TESTS=1 PATH=/usr/local/go/bin:$HOME/go/bin:$PATH
run() { # name timeout regex
  echo "=== $(date +%T) group $1 (timeout $2)"; go test -tags 'cuda goinfer_testhooks' -count=1 -v -timeout "$2" -run "$3" . > "$OUT/$1.log" 2>&1
  echo "=== $(date +%T) group $1 rc=$?"; grep -E '^\s*(--- |ok|FAIL|panic)' "$OUT/$1.log" | cut -c1-150
}
run mrope 5m 'TestRopeKVMRoPEBatched_'
run qwen35vl 10m 'TestQwen35VLReal_residentImagePrefillMatchesCPU'
run qwen25vl 10m 'TestQwen25VLResidentReal_gate|TestQwen25VLMRoPEPrefillResidentReal_gate'
run glmocr 15m 'TestGlmOcrCUDA'
run gptoss 15m 'TestGptOssResidentParityCUDA'
run cohere 25m 'TestCohereRealResidentParityCUDA'
run drafters 40m 'TestGenerateBlockSpec_production|TestBlockSpecStream|TestDefaultVerifyWidth_sweep|TestResidentDrafter_|TestDFlashLoop_|TestDrafterVsOff_perSuite|TestFlashDecodeBlockSpecLane|TestR14DrafterArgmax|TestBlockSpec_twoTurnsMatchPlain'
echo "=== $(date +%T) done"
