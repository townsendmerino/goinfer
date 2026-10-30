import json, sys, gzip
def load(p):
    op = gzip.open if p.endswith(".gz") else open
    el, pk = {}, None
    for l in op(p, "rt"):
        try: e = json.loads(l)
        except ValueError: continue
        t = e.get("Test", "")
        if t and "/" not in t and e.get("Action") in ("pass","fail","skip"): el[t] = e.get("Elapsed", 0)
        if not t and e.get("Action") in ("pass","fail"): pk = (e["Action"], e.get("Elapsed"))
    return el, pk
a, pa = load(sys.argv[1]); b, pb = load(sys.argv[2])
print("A pkg", pa, len(a), "tests", f"{sum(a.values()):.0f}s;  B pkg", pb, len(b), "tests", f"{sum(b.values()):.0f}s")
new = {t: v for t, v in b.items() if t not in a}
gone = {t: v for t, v in a.items() if t not in b}
common = [t for t in b if t in a]
print(f"new in B: {len(new)} tests, {sum(new.values()):.0f}s; gone: {len(gone)}, {sum(gone.values()):.0f}s; common grew {sum(b[t]-a[t] for t in common):+.0f}s")
for t, v in sorted(new.items(), key=lambda x: -x[1])[:12]: print(f"   new {t} {v:.1f}")
for t in sorted(common, key=lambda t: -(b[t]-a[t]))[:8]: print(f"   grew {t} {a[t]:.1f} -> {b[t]:.1f}")
