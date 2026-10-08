#!/usr/bin/env python3
"""Grade the vetted-checkpoint peer cells (docs/measurements/peer-vetted-2026-10-07-nobara-pc.md and -macbook.md):
`python3 grade.py <results dir>`.

The 2026-09-29 sweep's rules (../peer-sweep-2026-09-29/grade.py), unchanged:
- r = goinfer / Ollama per pair (run i against run i, 3 pairs);
- AHEAD (all > 1.03), LEVEL (all in 0.97-1.03), BEHIND (all < 0.97), AMBIGUOUS-HIGH / -LOW otherwise, decided by every pair;
- the 5% spread cap;
- the harness's proportional token gate;
- a GPU cell whose decode path is the CPU's is VOID.

Two additions, both registered before the run:
1. A cell the harness refused (no runs) is VOID with the harness's own reason. A swap-voided arm lands here: the harness
   runs with BENCH_SWAP_VOID_MB=0, so any swap growth over the cell voids that engine's arm.
2. A goinfer arm that is valid beside a VOID Ollama arm is reported "GOINFER-ALONE" with its median. claims.json then
   carries peer: null and no ratio.
"""
import ast, json, os, statistics as st, sys

B = sys.argv[1] if len(sys.argv) > 1 else "."
LO, HI = 0.97, 1.03


def outcome(rs, capped):
    if capped:
        return "AMBIGUOUS-LOW" if any(r < LO for r in rs) else "AMBIGUOUS-HIGH"
    if all(r > HI for r in rs):
        return "AHEAD"
    if all(LO <= r <= HI for r in rs):
        return "LEVEL"
    if all(r < LO for r in rs):
        return "BEHIND"
    return "AMBIGUOUS-LOW" if any(r < LO for r in rs) else "AMBIGUOUS-HIGH"


def spread(xs):
    return (max(xs) - min(xs)) / st.mean(xs)


def counts(c):
    v = c.get("counts") or {}
    return v if isinstance(v, dict) else ast.literal_eval(v)


def arm_ok(c):
    if not c or not c.get("runs"):
        return False, (c or {}).get("error") or "not run"
    k = counts(c)
    gate = (k.get("token_gate") or {}).get("verdict")
    if gate not in ("ok", "short"):
        return False, f"token gate {gate}"
    if (k.get("swap") or {}).get("swap_growth_mb", 0) > 0:
        return False, f"swap grew +{k['swap']['swap_growth_mb']} MB"
    dp = k.get("decode_path")
    if c["engine"] == "goinfer" and c["backend"] != "cpu" and dp and dp.startswith("cpu"):
        return False, f"decode path {dp}"
    return True, None


def grade(path):
    recs = json.load(open(path))
    prov, cells = recs[0], recs[1:]
    by = {(c["engine"], c["backend"], c["model"], c["depth"], c["config"]): c for c in cells}
    keys = sorted({k[1:] for k in by})
    for be, m, d, cfg in keys:
        g, p = by.get(("goinfer", be, m, d, cfg)), by.get(("ollama", be, m, d, cfg))
        gok, gwhy = arm_ok(g)
        pok, pwhy = arm_ok(p)
        row = dict(file=os.path.basename(path), backend=be, model=m, depth=d, config=cfg,
                   goinfer=[round(x, 1) for x in (g or {}).get("runs") or []],
                   ollama=[round(x, 1) for x in (p or {}).get("runs") or []],
                   decode_path=counts(g).get("decode_path") if g else None,
                   swap_mb=((counts(g).get("swap") or {}).get("swap_growth_mb") if g else None,
                            (counts(p).get("swap") or {}).get("swap_growth_mb") if p else None),
                   ollama_version=(prov.get("peer") or {}).get("version"),
                   ctx_pin=prov.get("ctx_pin"))
        if not gok:
            row.update(outcome="VOID", why=f"goinfer: {gwhy}")
        elif not pok:
            row.update(outcome="GOINFER-ALONE", why=f"Ollama: {pwhy}", median_goinfer=round(st.median(g["runs"]), 1))
        else:
            rs = [a / b for a, b in zip(g["runs"], p["runs"])]
            capped = spread(g["runs"]) > 0.05 or spread(p["runs"]) > 0.05
            row.update(pairs=[round(r, 3) for r in rs], median_r=round(st.median(rs), 3),
                       median_goinfer=round(st.median(g["runs"]), 1), median_ollama=round(st.median(p["runs"]), 1),
                       spread_g=round(100 * spread(g["runs"]), 1), spread_p=round(100 * spread(p["runs"]), 1),
                       outcome=outcome(rs, capped), capped=capped)
        print(json.dumps(row))


if __name__ == "__main__":
    for name in sorted(os.listdir(B)):
        if name.endswith(".json") and not name.startswith("plan"):
            try:
                json.load(open(os.path.join(B, name)))[0]
            except Exception:
                continue
            print(f"## {name}")
            grade(os.path.join(B, name))
