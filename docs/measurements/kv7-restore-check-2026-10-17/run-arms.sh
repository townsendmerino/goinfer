#!/usr/bin/env bash
# Restore-vs-cold check on the pilot's binary and model. Arms: R (A cold -> title C -> B restores), K1/K2 (B alone, -kv-host-bytes 0), R2 (restore again).
set -eo pipefail
H=$HOME/goinfer-bench/restore-check-2026-10-17; cd $H
A=cap-t02/bodies/req-002.json; C=cap-t01/bodies/req-001.json; B=cap-t01/bodies/req-002.json
arm() { # name, extra serve args, replay specs...
  local n=$1; local x=$2; shift 2
  echo "=== $(date +%T) arm $n"
  ./serve.sh $H/serve-$n.log $x & local SP=$!
  until grep -q 'goinfer serving' $H/serve-$n.log 2>/dev/null; do sleep 2; done
  python3 -I replay.py $H/res-$n.json "$@"
  kill $SP; wait $SP 2>/dev/null || true
}
arm R  ""                 $A $C $B:gen
arm K1 "-kv-host-bytes 0" $B:gen
arm K2 "-kv-host-bytes 0" $B:gen
arm R2 ""                 $A $C $B:gen
echo "=== $(date +%T) done"
