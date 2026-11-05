#!/usr/bin/env python3
"""Explains TestPlanSlack_realSizeDiff's golden change group by group. Reads the priced side's plan lines the test dumps with -planslack.out, at the last commit where
its golden held (937cfe8fb) and at f5261f4b, and writes groups.tsv beside it.

  python3 explain.py            # in this directory

Claim tested: the only change between the two is that dense pricing grew by a constant per model (norms, biases and small f32 tables, which the plan had not summed);
every plan term but `dense` is unchanged, and every decision boundary moved by exactly that constant.
"""
import collections, gzip, os, re, sys
H = os.path.dirname(os.path.abspath(__file__))
LINE = re.compile(r'^(D3 .*? extra=\d+) \|\| free (\d+): (\S+) ctx (\d+) slots (\d+) dense (\d+) slack (\d+) expertUsed (\d+) kv (\d+) extra (\d+) free (\d+) need (\d+) \| (.*)$')
FIXED = (7630553088, 7630553088 + 17 * 2**20)  # nobara's cuda probe, with and without the desktop's context
def load(name):
    g = collections.defaultdict(dict)
    for l in gzip.open(f"{H}/{name}", "rt"):
        m = LINE.match(l.rstrip("\n"))
        if not m:
            sys.exit("unparsed line: " + l[:100])
        g[m[1]][int(m[2])] = dict(pl=m[3], ctx=int(m[4]), slots=int(m[5]), dense=int(m[6]), slack=int(m[7]), exp=int(m[8]), kv=int(m[9]), extra=int(m[10]), need=int(m[12]))
    return g
A, B = load("after-937cfe8fb.txt.gz"), load("after-f5261f4b.txt.gz")
assert set(A) == set(B), "the two dumps name different groups"
model = lambda g: g.split()[1].split(".gguf")[0]
dec = lambda c: (c["pl"], c["ctx"], c["slots"])
delta = {}
for g in A:
    da, db = {c["dense"] for c in A[g].values()}, {c["dense"] for c in B[g].values()}
    assert len(da) == 1 and len(db) == 1, g
    d = db.pop() - da.pop()
    assert delta.setdefault(model(g), d) == d, f"{g}: the dense delta differs within a model"
moved_other = [g for g in A for t in ("slack", "exp", "extra") if {c[t] for c in A[g].values()} != {c[t] for c in B[g].values()}]
def bounds(cells):  # a true change point: the reading f and its neighbour f-1 both dumped, deciding differently
    return {f: (dec(cells[f - 1]), dec(cells[f])) for f in cells if f - 1 in cells and dec(cells[f - 1]) != dec(cells[f])}
rows, total, matched, fixed_changed = [], 0, 0, []
for g in sorted(A):
    d = delta[model(g)]
    ba, bb = bounds(A[g]), bounds(B[g])
    ok = sum(1 for f, x in ba.items() if bb.get(f + d) == x)
    total, matched = total + len(ba), matched + ok
    one_to_one = ok == len(ba) == len(bb)
    ch = []
    for f in FIXED:
        if f in A[g] and f in B[g] and dec(A[g][f]) != dec(B[g][f]):
            ch.append(f"free {f}: {dec(A[g][f])[:2]} -> {dec(B[g][f])[:2]}")
            fixed_changed.append((g, ch[-1]))
    rows.append((g, d, len(ba), ok, "yes" if one_to_one else "NO", len(ch), "; ".join(ch)))
with open(f"{H}/groups.tsv", "w") as f:
    f.write("group\tdense_delta_bytes\tchange_points_old\tmatched_at_plus_delta\tone_to_one\tnobara_readings_decided_differently\tdetail\n")
    for r in rows:
        f.write("\t".join(map(str, r)) + "\n")
print("dense delta per model, bytes:", delta)
print("groups where slack, expertUsed or extra moved:", sorted(set(moved_other)) or "none")
print(f"change points in the old run: {total}; found in the new run at exactly +dense delta with the same decision on both sides: {matched}")
print(f"groups whose change points correspond one to one: {sum(r[4] == 'yes' for r in rows)} of {len(rows)}")
print(f"nobara's two readings: {len(fixed_changed)} of {2 * len(A)} cells decide differently")
for g, c in fixed_changed:
    print("   ", g, "|", c)
sys.exit(0 if matched == total and not moved_other and all(r[4] == "yes" for r in rows) else 1)
