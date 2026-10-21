#!/usr/bin/env python3
"""goinfer against mlx-lm on the Mac's Metal cells, by docs/measurements/peer-notrunc-2026-10-17.md. The registered grader
(../grade.py) pairs goinfer with Ollama and llama.cpp only. This applies its own outcome and spread rules, imported
unchanged, to the mlx pairs: pair i = goinfer run i / mlx run i, the same bands, the same 5% spread cap and the same token
gate. Each row names the mlx server path its cell ran (mlx_path; inferred from the seed where the record predates the
field). Run: python3 grade_mlx.py [g-metal-decode.json]"""
import json, os, statistics as st, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import grade  # noqa: E402  the registered grader, unchanged

path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "g-metal-decode.json")
recs = json.load(open(path))
cells = [c for c in recs[1:] if c.get("runs")]
by = {(c["engine"], c["model"], c["config"]): c for c in cells}
for (eng, m, cfg), g in sorted(by.items(), key=lambda kv: kv[0][1:]):
    p = by.get(("mlx", m, cfg))
    if eng != "goinfer" or not p:
        continue
    rs = [a / b for a, b in zip(g["runs"], p["runs"])]
    capped = grade.spread(g["runs"]) > 0.05 or grade.spread(p["runs"]) > 0.05
    gates = (grade.counts(g).get("token_gate", {}).get("verdict"), grade.counts(p).get("token_gate", {}).get("verdict"))
    ok = all(v in ("ok", "short") for v in gates)
    path_ = p.get("mlx_path") or ("single-request" if (p.get("sent") or {}).get("seed") is not None else "batch")
    print(json.dumps(dict(model=m, config=cfg, peer="mlx", mlx_path=path_, pairs=[round(r, 3) for r in rs],
                          median=round(st.median(rs), 3), goinfer=[round(x, 1) for x in g["runs"]],
                          peer_runs=[round(x, 1) for x in p["runs"]], outcome="VOID" if not ok else grade.outcome(rs, capped),
                          capped=capped, token_gates=gates)))
