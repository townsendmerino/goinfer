#!/usr/bin/env bash
# Night job: the C-P01 A/B of docs/tasks/task-metal-audit-2026-10.md ("C-P01: built, pending its A/B"), pre-registered
# there on 2026-10-02 before any timed run, and amended the same day, still before any, to three arms. M26 at 8 slots,
# 128 timed tokens per process, one process at a time under scripts/swap_killwatch.sh:
#   old   = metal-tagged-f56b40ec.test (the binary T1.8 ran: the heap scale cache)
#   copy  = metal-tagged-2838b7de.test (scales copied out of the mapping)
#   pread = metal-tagged-<PREAD>.test  (scales pread from the file with the nibbles; the arm that ships)
# One discarded warm-up (old), then 8 rounds of the three arms in a rotating order. Every process holds at token 32
# and this script removes the hold file on sight, so all arms pause alike, outside the timed window, and each prints
# the Go heap in use there.
#
#   python3 scripts/night.py add metal-audit-cp01-ab --est 30 --priority 20 --by "Claude, Metal audit C-P01 A/B" \
#     --doc docs/tasks/task-metal-audit-2026-10.md -- bash ~/goinfer-bench/metal-audit-2026-10/run-cp01-ab.sh
# Logs: ~/goinfer-logs/metal-audit-2026-10/cp01-ab/; the graded summary is summary.txt there.
set -uo pipefail
OLD=f56b40ec
COPY=2838b7de
PREAD=494eb05a
BASE=$HOME/goinfer-bench/metal-audit-2026-10
KW=$BASE/swap_killwatch.sh
LOG=$HOME/goinfer-logs/metal-audit-2026-10/cp01-ab
M26=$HOME/models/gemma4-26b-int4-v14st.metal.giw
mkdir -p "$LOG"
for r in "$OLD" "$COPY" "$PREAD"; do [ -x "$BASE/metal-tagged-$r.test" ] || { echo "missing test binary for $r"; exit 1; }; done
[ -f "$KW" ] || { echo "missing kill-watch $KW"; exit 1; }
[ -f "$M26" ] || { echo "missing model $M26"; exit 1; }
cd "$BASE"
{
  for r in "$OLD" "$COPY" "$PREAD"; do echo "binary:   $r sha256 $(shasum -a 256 "$BASE/metal-tagged-$r.test" | cut -c1-16)"; done
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

rev_of() { case $1 in old) echo "$OLD" ;; copy) echo "$COPY" ;; pread) echo "$PREAD" ;; esac; }

one warmup-old "$OLD" || exit 0
for p in 1 2 3 4 5 6 7 8; do
  case $((p % 3)) in
    1) order="old copy pread" ;;
    2) order="copy pread old" ;;
    0) order="pread old copy" ;;
  esac
  for arm in $order; do
    one "p$p-$arm" "$(rev_of "$arm")" || exit 0
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
arms = ("old", "copy", "pread")
ratios = {"copy": [], "pread": []}
heaps = {a: [] for a in arms}
for p in range(1, 9):
    v = {a: read(f"p{p}-{a}") for a in arms}
    for a in arms:
        if v[a][1] is not None:
            heaps[a].append(v[a][1])
    if all(v[a][0] for a in arms):
        for a in ("copy", "pread"):
            ratios[a].append(v[a][0] / v["old"][0])
        print(f"round {p}: " + ", ".join(f"{a} {v[a][0]:.2f} tok/s (heap {v[a][1]} MB)" for a in arms)
              + f"; copy/old {v['copy'][0] / v['old'][0]:.4f}, pread/old {v['pread'][0] / v['old'][0]:.4f}")
    else:
        print(f"round {p}: INCOMPLETE ({v})")
if any(len(ratios[a]) < 8 for a in ratios) or any(len(heaps[a]) < 8 for a in arms):
    print(f"VERDICT: INCOMPLETE ({len(ratios['pread'])} of 8 rounds); nothing is decided")
    sys.exit()
med = {a: statistics.median(ratios[a]) for a in ratios}
drop = {a: statistics.median(heaps["old"]) - statistics.median(heaps[a]) for a in ratios}
for a in ("copy", "pread"):
    r = ratios[a]
    print(f"{a}/old: median {med[a]:.4f} (min {min(r):.4f}, max {max(r):.4f}), {sum(x < 1 for x in r)} of 8 below 1; "
          f"heap at token 32 {statistics.median(heaps[a]):.1f} MB, {drop[a]:.1f} MB below old's {statistics.median(heaps['old']):.1f}")
m, h = med["pread"], drop["pread"]
if h < 1024:
    print("VERDICT (pread): NOT SHIPPED — the heap dropped by less than 1.0 GB, so the saving did not happen")
elif m >= 0.97:
    print("VERDICT (pread): SHIP (median >= 0.97 and the heap dropped by >= 1.0 GB)")
elif m >= 0.93:
    print("VERDICT (pread): TO THE OWNER (0.93 <= median < 0.97)")
else:
    print("VERDICT (pread): NOT SHIPPED AS IS (median < 0.93)")
print("copy/old is reported, not graded: it says what the pread step is worth")
EOF
echo "finished: $(date '+%F %T %Z')" | tee -a "$LOG/provenance.txt"
