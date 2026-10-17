#!/usr/bin/env bash
# First-position logprobs: restored (R) vs cold (K) at the pass size the server uses (512) and at 256, where the restore point 1792 is a pass boundary.
set -eo pipefail
H=$HOME/goinfer-bench/restore-check-2026-10-17; cd $H
A=cap-t02/bodies/req-002.json; C=cap-t01/bodies/req-001.json; B=cap-t01/bodies/req-002.json
arm() { local n=$1; local chunk=$2; local x=$3; shift 3
  echo "=== $(date +%T) arm $n chunk=$chunk"
  GOINFER_PREFILL_CHUNK=$chunk ./serve.sh $H/serve-$n.log $x & local SP=$!
  until grep -q 'goinfer serving' $H/serve-$n.log 2>/dev/null; do sleep 2; done
  python3 -I replay.py $H/res-$n.json "$@"
  kill $SP; wait $SP 2>/dev/null || true
}
arm L-R512  512 ""                 $A $C $B:lp
arm L-K512  512 "-kv-host-bytes 0" $B:lp
arm L-K512b 512 "-kv-host-bytes 0" $B:lp
arm L-R256  256 ""                 $A $C $B:lp
arm L-K256  256 "-kv-host-bytes 0" $B:lp
arm L-K256b 256 "-kv-host-bytes 0" $B:lp
echo "=== $(date +%T) done"
