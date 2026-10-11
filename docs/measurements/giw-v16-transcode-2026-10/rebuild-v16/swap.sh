#!/bin/bash
# Put the v16 rebuilds in place under the existing names. Every old file is kept beside it as <name>.pre-v16 (with its
# marker); nothing is deleted. The two archive bundles are copied to the archive the same way and their byte counts
# compared.
set -uo pipefail
B=$HOME/goinfer-bench/v16-rebuild; M=$HOME/models; A=/srv/models
keep() { # keep <dir> <name>: rename the old file and its marker aside
  [ -e "$1/$2.pre-v16" ] && { echo "REFUSING: $1/$2.pre-v16 already exists"; return 1; }
  mv "$1/$2" "$1/$2.pre-v16" && { [ -e "$1/$2.verified" ] && mv "$1/$2.verified" "$1/$2.pre-v16.verified"; true; }
}
echo "started $(date '+%H:%M:%S')"
for n in gemma4-26b-int4.giw gemma4-26b-int4-v14st.metal.giw qwen3.6-35b-a3b-int4.giw mellum2.int4.giw; do
  [ -f "$B/new/$n" ] || { echo "MISSING rebuild: $n"; exit 1; }
  keep "$M" "$n" || exit 1
  mv "$B/new/$n" "$M/$n" && mv "$B/new/$n.verified" "$M/$n.verified"
  echo "~/models/$n: $(stat -c %s "$M/$n") bytes (old kept: $(stat -c %s "$M/$n.pre-v16") bytes)"
done
echo "== archive"
n=gemma4-26b-int4-v14st.metal.giw
keep "$A" "$n" || exit 1
rsync -a "$M/$n" "$M/$n.verified" "$A/" && echo "archive $n: $(stat -c %s "$A/$n") bytes, local $(stat -c %s "$M/$n") bytes $([ "$(stat -c %s "$A/$n")" = "$(stat -c %s "$M/$n")" ] && echo MATCH || echo MISMATCH)"
n=qwen3.6-35b-a3b-int4-v12.giw
keep "$A" "$n" || exit 1
want=$(stat -c %s "$B/new/$n")
rsync -a "$B/new/$n" "$B/new/$n.verified" "$A/" && echo "archive $n: $(stat -c %s "$A/$n") bytes, staged $want bytes $([ "$(stat -c %s "$A/$n")" = "$want" ] && echo MATCH || echo MISMATCH)"
[ "$(stat -c %s "$A/$n")" = "$want" ] && find "$B/new" -name "$n*" -delete && echo "staging copy of $n removed"
sync
echo "finished $(date '+%H:%M:%S'); free on ~: $(df -h $HOME | tail -1 | awk '{print $4}'); staging: $(ls $B/new | wc -l) files"
