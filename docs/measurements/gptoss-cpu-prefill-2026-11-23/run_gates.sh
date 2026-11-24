#!/bin/bash
# gpt-oss batched CPU prefill: G3 (fidelity, shared lock) then G4 with G2 inside it (TTFT, exclusive lock), as registered in the README
# beside this file. BIN is a decoder test binary of the code commit (go test -c -tags goinfer_testhooks); it runs from the worktree's
# decoder/ directory, where the prompt set's relative paths resolve.
#   bash run_gates.sh BIN WORKTREE OUT
set -uo pipefail
BIN=$1 WT=$2 OUT=$3
export PATH=/usr/local/go/bin:$HOME/go/bin:$PATH
mkdir -p "$OUT/g3" "$OUT/g4"
cd "$WT/decoder"
export GPTOSS_PF_MODEL=$HOME/models/gpt-oss-20b-MXFP4.gguf GPTOSS_PF_GIT=$(basename "$BIN" .test) GPTOSS_PF_MIN_AVAIL_MIB=31107
echo "$(date -Is) G3 start"
GPTOSS_PF_MODE=fidelity GPTOSS_PF_OUT="$OUT/g3" python3 ../scripts/timing_lock.py run --shared --wait 7200 --label gptoss-pf-g3 -- \
	"$BIN" -test.run '^TestGptOssCPUPrefillGate$' -test.v -test.timeout 120m >"$OUT/g3/stdout.log" 2>&1
echo "$(date -Is) G3 rc=$?"
grep -E "FIDELITY pooled|--- (PASS|FAIL)" "$OUT/g3/stdout.log"
echo "$(date -Is) G4 start"
GPTOSS_PF_MODE=ttft GPTOSS_PF_OUT="$OUT/g4" python3 ../scripts/timing_lock.py run --wait 7200 --label gptoss-pf-g4 -- \
	"$BIN" -test.run '^TestGptOssCPUPrefillGate$' -test.v -test.timeout 150m >"$OUT/g4/stdout.log" 2>&1
echo "$(date -Is) G4 rc=$?"
grep -E -- "--- (PASS|FAIL)" "$OUT/g4/stdout.log"
echo "$(date -Is) GATES-DONE"
