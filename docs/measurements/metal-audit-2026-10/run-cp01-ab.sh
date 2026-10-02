#!/usr/bin/env bash
# Night job: the C-P01 A/B of docs/tasks/task-metal-audit-2026-10.md ("C-P01: built, pending its A/B"), pre-registered
# there on 2026-10-02 before any timed run. M26 at 8 slots, 128 timed tokens per process, one process at a time under
# scripts/swap_killwatch.sh:
#   old = metal-tagged-f56b40ec.test (the binary T1.8 ran: the heap scale cache)
#   new = metal-tagged-<NEW>.test   (the same probe at the C-P01 commit: scales staged from the mapping)
# One discarded warm-up (old), then 8 pairs in ABBA order. Every process holds at token 32 and this script removes the
# hold file on sight, so both arms pause alike, outside the timed window, and both print the Go heap in use there.
#
#   python3 scripts/night.py add metal-audit-cp01-ab --est 20 --priority 20 --by "Claude, Metal audit C-P01 A/B" \
#     --doc docs/tasks/task-metal-audit-2026-10.md -- bash ~/goinfer-bench/metal-audit-2026-10/run-cp01-ab.sh
# Logs: ~/goinfer-logs/metal-audit-2026-10/cp01-ab/; the graded summary is summary.txt there.
set -uo pipefail
OLD=f56b40ec
NEW=2838b7de
BASE=$HOME/goinfer-bench/metal-audit-2026-10
KW=$BASE/swap_killwatch.sh
LOG=$HOME/goinfer-logs/metal-audit-2026-10/cp01-ab
M26=$HOME/models/gemma4-26b-int4-v14st.metal.giw
mkdir -p "$LOG"
for r in "$OLD" "$NEW"; do [ -x "$BASE/metal-tagged-$r.test" ] || { echo "missing test binary for $r"; exit 1; }; done
[ -f "$KW" ] || { echo "missing kill-watch $KW"; exit 1; }
[ -f "$M26" ] || { echo "missing model $M26"; exit 1; }
cd "$BASE"
{
  for r in "$OLD" "$NEW"; do echo "binary:   $r sha256 $(shasum -a 256 "$BASE/metal-tagged-$r.test" | cut -c1-16)"; done
  echo "model:    $M26 ($(stat -f %z "$M26") bytes)"
  echo "started:  $(date '+%F %T %Z')"
  echo "machine:  $(sysctl -n machdep.cpu.brand_string), $(( $(sysctl -n hw.memsize) / 1073741824 )) GB, macOS $(sw_vers -productVersion)"
  echo "load:     $(uptime | sed 's/.*load averages*: //')"
  echo "swap:     $(sysctl -n vm.swapusage)"
} | tee "$LOG/provenance.txt"

# one <name> <rev>: one probe process. Returns 9 if the kill-watch fired.
one() {
  local name=$1 rev=$2 holdf=$LOG/$1.hold
  rm -f "$holdf"
  echo "== $name ($rev) — $(date '+%T')" | tee -a "$LOG/provenance.txt"
  env GOINFER_METAL_AUDIT_M26=1 GOINFER_AUDIT_MODEL="$M26" GOINFER_AUDIT_SLOTS=8 GOINFER_AUDIT_TOKENS=128 \
    GOINFER_AUDIT_HOLD="$holdf" "$BASE/metal-tagged-$rev.test" -test.v -test.count=1 -test.timeout 15m \
    -test.run '^TestAuditM26_pagedProbe$' > "$LOG/$name.log" 2>&1 &
  local pid=$!
  KILL_DELTA_MB=1024 TICK_MB=80 TICK_N=2 POLL_S=1 bash "$KW" "$pid" "$LOG/$name.killwatch.log" > /dev/null &
  local kw=$!
  for _ in $(seq 1 3000); do
    [ -s "$holdf" ] && { rm -f "$holdf"; break; }
    kill -0 "$pid" 2>/dev/null || break
    sleep 0.2
  done
  wait "$pid"
  local rc=$?
  wait "$kw" 2>/dev/null
  echo "$name rc=$rc; swap now $(sysctl -n vm.swapusage)" | tee -a "$LOG/provenance.txt"
  if grep -q 'KILLING' "$LOG/$name.killwatch.log" 2>/dev/null; then
    echo "$name: the kill-watch fired; the A/B stops here" | tee -a "$LOG/provenance.txt"
    return 9
  fi
  sleep 15
  return 0
}

one warmup-old "$OLD" || exit 0
i=0
for p in 1 2 3 4 5 6 7 8; do
  if [ $((p % 2)) = 1 ]; then order="$OLD $NEW"; else order="$NEW $OLD"; fi
  for r in $order; do
    arm=old; [ "$r" = "$NEW" ] && arm=new
    one "p$p-$arm" "$r" || exit 0
  done
done

python3 - "$LOG" <<'EOF' | tee "$LOG/summary.txt"
import re, statistics, sys, pathlib
d = pathlib.Path(sys.argv[1])
def read(name):
    t = (d / f"{name}.log").read_text()
    tok = re.search(r"RESULT 8 slots: ([0-9.]+) tok/s", t)
    heap = re.search(r"Go heap in use ([0-9.]+) MB", t)
    return (float(tok.group(1)) if tok else None, float(heap.group(1)) if heap else None)
ratios, heaps = [], {"old": [], "new": []}
for p in range(1, 9):
    o, n = read(f"p{p}-old"), read(f"p{p}-new")
    for arm, v in (("old", o), ("new", n)):
        if v[1] is not None:
            heaps[arm].append(v[1])
    if o[0] and n[0]:
        ratios.append(n[0] / o[0])
        print(f"pair {p}: old {o[0]:.2f} tok/s, new {n[0]:.2f} tok/s, new/old {n[0] / o[0]:.4f}; heap old {o[1]} MB, new {n[1]} MB")
    else:
        print(f"pair {p}: INCOMPLETE (old {o}, new {n})")
if len(ratios) < 8 or len(heaps["old"]) < 8 or len(heaps["new"]) < 8:
    print(f"VERDICT: INCOMPLETE ({len(ratios)} of 8 pairs); nothing is decided")
    sys.exit()
med = statistics.median(ratios)
drop = statistics.median(heaps["old"]) - statistics.median(heaps["new"])
below = sum(r < 1 for r in ratios)
print(f"median new/old {med:.4f} (min {min(ratios):.4f}, max {max(ratios):.4f}); {below} of 8 pairs below 1")
print(f"median heap at token 32: old {statistics.median(heaps['old']):.1f} MB, new {statistics.median(heaps['new']):.1f} MB, drop {drop:.1f} MB")
if drop < 1024:
    print("VERDICT: NOT SHIPPED — the heap dropped by less than 1.0 GB, so the saving did not happen")
elif med >= 0.97:
    print("VERDICT: SHIP (median >= 0.97 and the heap dropped by >= 1.0 GB)")
elif med >= 0.93:
    print("VERDICT: TO THE OWNER (0.93 <= median < 0.97)")
else:
    print("VERDICT: NOT SHIPPED AS IS (median < 0.93): pread the scale span with the nibbles and re-run")
EOF
echo "finished: $(date '+%F %T %Z')" | tee -a "$LOG/provenance.txt"
