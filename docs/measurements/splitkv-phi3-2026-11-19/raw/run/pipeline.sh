#!/bin/bash
# splitkv-phi3 2026-11-19: build the scratch identity binary, run the identity check, then the registered sweep.
export PATH=/usr/local/go/bin:$HOME/go/bin:$PATH TMPDIR=$HOME/goinfer-work/review-tmp/nobara-splitkv
WT=$HOME/mycode/splitkv-phi3; OUT=$HOME/goinfer-logs/splitkv-phi3-2026-11-19; W=$HOME/goinfer-work/splitkv-phi3
LOCK="python3 $WT/scripts/timing_lock.py"
cd $WT/backend/cuda || exit 1
echo "$(date) build identity binary"
$LOCK run --shared --wait 7200 --label splitkv-phi3-build -- go test -c -tags 'cuda goinfer_testhooks' -o $W/cuda-id.test . || { echo BUILD-FAILED; exit 1; }
echo "$(date) identity (exclusive)"
export GOINFER_BENCH_MODEL=$HOME/models/phi3-mini-4k-gguf/Phi-3-mini-4k-instruct-q4.gguf
export SPLITKV_SERVE_AUTO=1
$LOCK run --wait 7200 --label "splitkv-phi3 identity" -- bash -c "
  echo \"load1 \$(cut -d' ' -f1 /proc/loadavg) at \$(date -Iseconds)\"; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
  $W/cuda-id.test -test.run '^TestSplitKVPhi3Identity\$' -test.v -test.timeout 20m > $OUT/identity-3900.log 2>&1; echo identity-3900 rc=\$?
  SPLITKV_ID_SHALLOW=1 $W/cuda-id.test -test.run '^TestSplitKVPhi3Identity\$' -test.v -test.timeout 20m > $OUT/identity-128.log 2>&1; echo identity-128 rc=\$?"
grep -h 'launches\|bit-identical\|kv=' $OUT/identity-*.log
echo "$(date) sweep"
bash $WT/docs/measurements/splitkv-phi3-2026-11-19/run-sweep.sh $W/cuda-id.test $OUT
echo "$(date) PIPELINE-DONE"
