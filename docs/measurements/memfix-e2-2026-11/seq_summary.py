"""Summarize seq-{before,after}-*.tsv: the card's drop over the pre-fit allocations (start free minus free after the last on-device
allocation before the fit-check reading), less the 2 MiB a process's first allocation takes (packing-model.md's card column),
and the count of on-device allocations. Usage: python3 -I seq_summary.py <dir>"""
import sys, glob, os
MIB = 1 << 20
rows = {}
for p in sorted(glob.glob(os.path.join(sys.argv[1], 'seq-*.tsv'))):
    arm, model = os.path.basename(p)[4:-4].split('-', 1)
    ev = [l.rstrip('\n').split('\t') for l in open(p)]
    start = int(ev[0][2])
    fit = [i for i, e in enumerate(ev) if e[0] == 'S' and e[1] == 'fit'][0]
    pre = [e for e in ev[:fit] if e[0] == 'A' and e[3] == '1']
    drop = start - int(pre[-1][5])
    dense = sum(int(e[2]) for e in pre if e[4] == 'dense')
    ndense = sum(1 for e in pre if e[4] == 'dense')
    rows.setdefault(model, {})[arm] = (drop / MIB - 2, len(pre), ndense, dense / MIB)
print(f"{'model':45} {'card before':>12} {'card after':>11} {'saved':>8} {'allocs b/a':>11} {'dense allocs b/a':>17}")
for m, r in rows.items():
    b, a = r.get('before'), r.get('after')
    if not (b and a):
        continue
    print(f"{m:45} {b[0]:12.2f} {a[0]:11.2f} {b[0]-a[0]:8.2f} {b[1]:5d}/{a[1]:<5d} {b[2]:8d}/{a[2]:<8d}")
