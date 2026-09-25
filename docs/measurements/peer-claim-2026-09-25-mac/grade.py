#!/usr/bin/env python3
"""Grade peer-claim cells g/i (decode) and h (prefill) — `python3 grade.py [dir]`, by docs/measurements/peer-claim-2026-09-25.md Part 1.

Decode: pair i = goinfer run i vs peer run i; r = goinfer / peer (r > 1: goinfer faster).
Prefill: pair i = request i; r = peer TTFT / goinfer TTFT.
Outcome from ALL pairs: AHEAD (all > 1.03), LEVEL (all in [0.97, 1.03]), AMBIGUOUS-HIGH (straddle 1.03,
none < 0.97), AMBIGUOUS-LOW (straddle 0.97), BEHIND (all < 0.97). Spread cap: if either engine's
run-to-run spread (max - min) / mean > 5%, the outcome is capped at AMBIGUOUS, HIGH or LOW by the same
straddle rule (LOW if any pair < 0.97). Validity: every cell returned >= 95% of the requested tokens.
"""
import ast, json, os, statistics as st, sys

# The results files sit beside this script; pass another directory to grade a different copy.
B = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
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
    if any(r < LO for r in rs):
        return "AMBIGUOUS-LOW"
    return "AMBIGUOUS-HIGH"


def spread(xs):
    return (max(xs) - min(xs)) / st.mean(xs)


def decode(path, requested):
    cells = [c for c in json.load(open(path))[1:] if c.get("runs")]
    by = {(c["engine"], c["model"], c["depth"]): c for c in cells}
    rows = []
    for (eng, m, d), g in sorted(by.items(), key=lambda kv: (kv[0][1], kv[0][2])):
        if eng != "goinfer":
            continue
        for peer in ("ollama", "llamacpp"):
            p = by.get((peer, m, d))
            if not p:
                continue
            rs = [a / b for a, b in zip(g["runs"], p["runs"])]
            sg, sp = spread(g["runs"]), spread(p["runs"])
            capped = sg > 0.05 or sp > 0.05
            tok_ok = all(json_counts(c)["tokens"] >= 0.95 * requested for c in (g, p))
            rows.append(dict(model=m, depth=d, peer=peer, pairs=[round(r, 3) for r in rs],
                             median=round(st.median(rs), 3), n=len(rs),
                             goinfer=[round(x, 1) for x in g["runs"]], peer_runs=[round(x, 1) for x in p["runs"]],
                             spread_g=round(100 * sg, 1), spread_p=round(100 * sp, 1),
                             outcome=("VOID" if not tok_ok else outcome(rs, capped)), capped=capped,
                             tokens=(json_counts(g)["tokens"], json_counts(p)["tokens"]),
                             load=(g["machine"]["loadavg"][0], p["machine"]["loadavg"][0])))
    return rows


def json_counts(c):
    v = c["counts"]
    return v if isinstance(v, dict) else ast.literal_eval(v)  # the harness stores a repr'd dict


def prefill(path):
    rows = []
    for c in json.load(open(path))["cells"]:
        g, p = c["goinfer"]["ttft_ms_all"], c["ollama"]["ttft_ms_all"]
        rs = [b / a for a, b in zip(g, p)]  # peer TTFT / goinfer TTFT
        sg, sp = spread(g), spread(p)
        capped = sg > 0.05 or sp > 0.05
        rows.append(dict(model=c["model"], K=c["depth"], peer="ollama", pairs=[round(r, 3) for r in rs],
                         median=round(st.median(rs), 3), n=len(rs), goinfer_ms=g, peer_ms=p,
                         goinfer_exact_ms_median=c["goinfer_exact"]["ttft_ms_median"],
                         spread_g=round(100 * sg, 1), spread_p=round(100 * sp, 1),
                         outcome=outcome(rs, capped), capped=capped))
    return rows


if __name__ == "__main__":
    req = 3 * 8 * 64
    for name in ("g-metal-decode.json", "i-cpu-decode.json"):
        print(f"## {name}")
        for r in decode(os.path.join(B, name), req):
            print(json.dumps(r))
    print("## h-metal-prefill.json")
    for r in prefill(os.path.join(B, "h-metal-prefill.json")):
        print(json.dumps(r))
