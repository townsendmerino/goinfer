#!/usr/bin/env python3
"""Grade cells.tsv against README.md's bar. A cell is void if, at its start, load1 > 8.0, MemAvailable < 5 GiB, a foreign build/test
process other than gopls was running, a CUDA process other than kwin_wayland held the card, or its tok/s is not a number. A pair is
(c1 A, c2 B) or (c4 A, c3 B) of one block; a re-run of block k is logged as block 100+k and its pairs replace k's void ones."""
import statistics, sys

def void_reason(r):
    if float(r["load1"]) > 8.0:
        return f"load1 {r['load1']}"
    if int(r["memavail_mib"]) < 5 * 1024:
        return f"MemAvailable {r['memavail_mib']} MiB"
    foreign = [f for f in r["foreign"].split(",") if f not in ("-", "") and not f.startswith("gopls:")]
    if foreign:
        return "foreign " + ",".join(foreign)
    apps = [a for a in r["gpu_apps"].split(";") if a.strip() and "kwin_wayland" not in a]
    if apps:
        return "gpu " + ";".join(apps)
    try:
        float(r["tok_s"])
    except ValueError:
        return r["tok_s"]
    return ""

rows = [dict(zip(h, l.rstrip("\n").split("\t"))) for h in [None] for l in []]
with open(sys.argv[1]) as f:
    head = f.readline().rstrip("\n").split("\t")
    rows = [dict(zip(head, l.rstrip("\n").split("\t"))) for l in f if l.strip()]
cells = {}
print("void cells:")
for r in rows:
    why = void_reason(r)
    if why:
        print(f"  d{r['depth']} b{r['block']} c{r['cell']} {r['arm']} {r['start']}: {why} (tok/s {r['tok_s']})")
    cells[(int(r["depth"]), int(r["block"]), int(r["cell"]))] = (r, why)
res = {}
for depth in sorted({k[0] for k in cells}):
    pairs = {}
    for (d, b, c), (r, why) in sorted(cells.items()):
        if d != depth or c not in (1, 4):
            continue
        partner = 2 if c == 1 else 3
        pr = cells.get((d, b, partner))
        if pr is None or why or pr[1]:
            continue
        a, bb = float(r["tok_s"]), float(pr[0]["tok_s"])
        pairs[(b % 100, c)] = (a, bb, b)
    ratios = [bb / a for a, bb, _ in pairs.values()]
    print(f"\ndepth {depth}: {len(ratios)} valid pairs")
    for (blk, c), (a, bb, src) in sorted(pairs.items()):
        print(f"  block {blk}{' (re-run %d)' % src if src >= 100 else ''} pair {'A1B1' if c == 1 else 'B2A2'}: A {a:.1f} B {bb:.1f} B/A {bb/a:.4f}")
    if ratios:
        med = statistics.median(ratios)
        faster = sum(x > 1 for x in ratios)
        print(f"  median B/A {med:.4f}; B faster on {faster}/{len(ratios)}; min {min(ratios):.4f} max {max(ratios):.4f}")
        res[depth] = (med, faster, len(ratios))
if 3900 in res and 128 in res:
    (m39, f39, n39), (m128, _, n128) = res[3900], res[128]
    if n39 < 10 or n128 < 10:
        verdict = f"INCOMPLETE ({n128} pairs at 128, {n39} at 3900; the bar needs 10 each)"
    elif m128 < 0.98:
        verdict = "KILL (128 median under 0.98)"
    elif f39 >= 9 and m39 >= 1.05:
        verdict = "WIN"
    elif f39 >= 7 and m39 >= 1.00:
        verdict = "AMBIGUOUS (parked)"
    else:
        verdict = "KILL"
    print(f"\nverdict: {verdict}")
