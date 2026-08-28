#!/usr/bin/env python3
"""G33 Tier 1: replay the routing trace through a per-layer LRU and classify the misses.

STEP 1 IS A VALIDATION GATE, NOT A FORMALITY. If this replay does not reproduce the measured
hits/misses, the model of the cache is wrong and every number after it is meaningless.
"""
import json, sys, collections

def replay(decisions, slots):
    """Per-layer LRU of `slots` experts. Returns (hits, misses, miss_records)."""
    lru = collections.defaultdict(collections.OrderedDict)  # layer -> {expert: None}, LRU order
    hits = misses = 0
    recs = []
    # per-layer history of when each expert was last used, in that layer's own decision count
    lastseen = collections.defaultdict(dict)
    step = collections.Counter()
    for d in decisions:
        L, idx = d["layer"], d["idx"]
        step[L] += 1
        c = lru[L]
        for e in idx:
            if e in c:
                c.move_to_end(e); hits += 1
            else:
                misses += 1
                prev = lastseen[L].get(e)
                recs.append({"layer": L, "expert": e, "step": step[L],
                             "age": (step[L] - prev) if prev is not None else None})
                c[e] = None
                if len(c) > slots:
                    c.popitem(last=False)
        for e in idx:
            lastseen[L][e] = step[L]
    return hits, misses, recs

def main(path, exp_hits=None, exp_misses=None):
    t = json.load(open(path))
    dec, slots = t["decisions"], t["slots"]
    layers = sorted({d["layer"] for d in dec})
    print(f"trace: {len(dec)} decisions, {len(layers)} MoE layers, topK={t['topK']}, "
          f"slots={slots}, tokens={t['tokens']}")
    print(f"decisions/layer: {len(dec)/len(layers):.1f}")
    h, m, recs = replay(dec, slots)
    print(f"\nREPLAY: {h} hits / {m} misses = {h/(h+m)*100:.1f}%")
    if exp_hits:
        ok = (h == exp_hits and m == exp_misses)
        print(f"MEASURED: {exp_hits} hits / {exp_misses} misses = {exp_hits/(exp_hits+exp_misses)*100:.1f}%")
        print(f"VALIDATION GATE: {'PASS — the cache model reproduces hardware' if ok else 'FAIL — model is wrong, STOP'}")
        if not ok:
            d = abs(h-exp_hits)
            print(f"  off by {d} ({d/exp_hits*100:.2f}%) — do not trust anything below")
    # COLD vs EVICTED
    cold = [r for r in recs if r["age"] is None]
    ev   = [r for r in recs if r["age"] is not None]
    print(f"\nMISS SPLIT: cold {len(cold)} ({len(cold)/len(recs)*100:.1f}%) | "
          f"evicted {len(ev)} ({len(ev)/len(recs)*100:.1f}%)")
    if ev:
        ages = sorted(r["age"] for r in ev)
        import statistics
        print(f"  evicted age (decisions at that layer since last use): "
              f"median {statistics.median(ages):.0f}, p90 {ages[int(len(ages)*0.9)]}, max {max(ages)}")
        for k in (2, 4, 8, 16, 32):
            n = sum(1 for a in ages if a <= k)
            print(f"    recoverable if the cache remembered {k:2d} more steps: {n} ({n/len(recs)*100:.1f}% of misses)")

if __name__ == "__main__":
    main(sys.argv[1], *(int(x) for x in sys.argv[2:]))
