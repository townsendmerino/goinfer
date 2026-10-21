#!/usr/bin/env bash
set -eo pipefail
H=$HOME/goinfer-bench/restore-check-2026-10-17; cd $H
A=cap-t02/bodies/req-002.json; C=cap-t01/bodies/req-001.json; B=$1
arm() { local n=$1; local x=$2; shift 2
  echo "=== $(date +%T) arm $n"
  ./serve.sh $H/serve-$n.log $x & local SP=$!
  until grep -q 'goinfer serving' $H/serve-$n.log 2>/dev/null; do sleep 2; done
  python3 -I replay.py $H/res-$n.json "$@"
  kill $SP; wait $SP 2>/dev/null || true
}
arm T5-R1 ""                 $A $C $B:gen256
arm T5-R2 ""                 $A $C $B:gen256
arm T5-K1 "-kv-host-bytes 0" $B:gen256
arm T5-K2 "-kv-host-bytes 0" $B:gen256
echo "=== $(date +%T) done"
