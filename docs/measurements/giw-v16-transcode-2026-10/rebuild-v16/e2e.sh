#!/bin/bash
# End to end, old bundle on the old build against its rebuild on the new build: one greedy reply each, on the CPU.
B=$HOME/goinfer-bench/v16-rebuild; M=$HOME/models
echo "started $(date '+%H:%M:%S')"
p=18500
for n in gemma4-26b-int4.giw gemma4-26b-int4-v14st.metal.giw mellum2.int4.giw; do
  p=$((p+2))
  o=$("$B/reply.sh" "$B/serve-v15-linux" "$M/$n" $p "$B/e2e-old-$n.log")
  w=$("$B/reply.sh" "$B/serve-v16-linux" "$B/new/$n" $((p+1)) "$B/e2e-new-$n.log")
  echo "== $n"; echo "  old: $o"; echo "  new: $w"; [ "$o" = "$w" ] && [ -n "$o" ] && echo "  SAME reply" || echo "  DIFFERENT (or empty)"
done
echo "finished $(date '+%H:%M:%S')"
