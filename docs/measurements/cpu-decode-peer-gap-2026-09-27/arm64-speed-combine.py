"""L1 gate 6 (arm64 CPU speed): combine pass 1 and pass 2 by the rule pre-registered in run-arm64-speed-pass2.sh.

Pass 1 labels are literal (goinfer = NEW 5c85f7c0, goinfer_old = OLD 3cd62e6d). Pass 2 swapped the binaries, so its
"goinfer" is OLD and its "goinfer_old" is NEW.
"""
import json, math, sys

def cells(path):
    out = {}
    for line in open(path):
        line = line.strip()
        if line.startswith("{"):
            r = json.loads(line) if not line.startswith('{"cells"') else None
            if r and "engine" in r:
                out[(r["engine"], r["model"])] = r
    return out

def load(path):
    try:
        d = json.load(open(path))
    except json.JSONDecodeError:
        return cells(path)
    rows = d if isinstance(d, list) else d.get("cells", d.get("results", []))
    return {(r["engine"], r["model"]): r for r in rows if isinstance(r, dict) and "engine" in r}

p1, p2 = load(sys.argv[1]), load(sys.argv[2])
print("| model | pass 1 new / old (new first) | pass 2 new / old (old first) | pass 1 new ÷ old | pass 2 new ÷ old "
      "| **combined (geo-mean)** | Ollama p1 / p2 | new ÷ Ollama (mean) | old ÷ Ollama (mean) | verdict |")
print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
for m in ("0.5B", "1.5B", "7B"):
    n1, o1 = p1[("goinfer", m)]["mean"], p1[("goinfer_old", m)]["mean"]
    o2, n2 = p2[("goinfer", m)]["mean"], p2[("goinfer_old", m)]["mean"]
    ol1, ol2 = p1[("ollama", m)]["mean"], p2[("ollama", m)]["mean"]
    r1, r2 = n1 / o1, n2 / o2
    comb = math.sqrt(r1 * r2)
    if (r1 >= 1.03 and r2 <= 0.97) or (r1 <= 0.97 and r2 >= 1.03):
        verdict = "UNRESOLVED BY DRIFT"
    elif comb < 1.00 and r1 < 1.00 and r2 < 1.00:
        verdict = "slower in both passes: NEON-widen follow-up flagged"
    else:
        verdict = "not slower"
    nol = math.sqrt((n1 / ol1) * (n2 / ol2))
    ool = math.sqrt((o1 / ol1) * (o2 / ol2))
    print(f"| {m} | {n1} / {o1} | {n2} / {o2} | {r1:.3f}× | {r2:.3f}× | **{comb:.3f}×** | {ol1} / {ol2} | {nol:.3f}× "
          f"| {ool:.3f}× | {verdict} |")
for tag, p in (("pass 1", p1), ("pass 2", p2)):
    bad = [(k, r.get("error"), r.get("counts", {}).get("token_gate", {}).get("verdict")) for k, r in p.items()
           if r.get("error") or r.get("counts", {}).get("token_gate", {}).get("verdict") != "ok"]
    print(f"\n{tag}: {len(p)} cells; errors / non-ok token gates: {bad or 'none'}")
