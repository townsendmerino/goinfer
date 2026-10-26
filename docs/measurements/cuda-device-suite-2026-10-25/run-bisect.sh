#!/bin/bash
# The red device suite's two tests (TestKEqVCopy_bitIdenticalToTheDoubleProjection, TestPrefillTails_bitIdenticalToTheRecordedBaseline)
# at #200's parent, at #200 and at the fix, each through the device suite's own runner (REV, TESTARGS). Run by night.py.
set -u
S=$HOME/mycode/goinfer/docs/measurements/cuda-device-suite-2026-10-18/run-cuda-device-suite.sh
T='-run ^(TestKEqVCopy_bitIdenticalToTheDoubleProjection|TestPrefillTails_bitIdenticalToTheRecordedBaseline)$'
rc=0
for rev in d0a64bcc6 c0acef3b0 2a39592bc; do
  echo "===== $rev $(date +%H:%M:%S)"
  WT=$HOME/goinfer-bench/cuda-suite-bisect-wt REV=$rev TESTARGS="$T" bash "$S" "$HOME/goinfer-logs/cuda-suite-bisect-2026-10-25/$rev" 2>&1 | grep -E "^rev:|passed|FAIL|VERDICT|DIFFERS|bit-identical" 
done
echo "DONE"
