#!/usr/bin/env python3
"""Grade the nobara peer re-anchor (README.md beside this file): `python3 grade.py <results dir> [<results dir> ...]`.

A copy of ../peer-sweep-2026-09-29/grade.py whose bands, pairing, spread cap, token gate and decode-path gate are unchanged.
Three registered differences, all about which arms are graded, none about a band:
  1. It reads this sweep's files (a-dense, a-depth8000, b-controls, c-26b, f-sampled, d-prefill, e-cpu, v-cpu) from each dir given.
  2. An arm the harness voided under the void rule (counts.swap.swap_void, or a heavy-load start it could not get past) is VOID with
     that reason, as are arms with no runs; a goinfer arm beside a missing or VOID peer arm is printed VOID (peer), never paired.
  3. Each row carries the cell's start machine state (loadavg, available memory) as the record of the void rule's readings.

Original header (09-29):

A copy of ../peer-claim-2026-09-25/grade.py with exactly the two changes that pre-registration names:
  1. validity is the harness's proportional token gate (counts.token_gate.verdict, per engine), not 95% of REQ;
  2. a GPU cell whose recorded decode path is the CPU's is VOID (and a cell the harness refused has no runs: MISSING).
Everything below that is unchanged from the 09-25 grader.

Original header:

The rules are the Mac grader's (../peer-claim-2026-09-25-mac/grade.py), unchanged. This copy differs only in keying
cells by (backend, model, depth, config), since f-sampled.json holds three configs per model and a-e-dense.json two
backends.

Decode: pair i = goinfer run i vs peer run i; r = goinfer / peer (r > 1: goinfer faster).
Prefill: pair i = request i; r = peer TTFT / goinfer TTFT.
Outcome from ALL pairs: AHEAD (all > 1.03), LEVEL (all in [0.97, 1.03]), AMBIGUOUS-HIGH (straddle 1.03,
none < 0.97), AMBIGUOUS-LOW (straddle 0.97), BEHIND (all < 0.97). Spread cap: if either engine's
run-to-run spread (max - min) / mean > 5%, the outcome is capped at AMBIGUOUS, HIGH or LOW by the same
straddle rule (LOW if any pair < 0.97). Validity: every cell returned >= 95% of the requested tokens.
"""
import ast, json, os, statistics as st, sys

DIRS = sys.argv[1:] or [os.path.dirname(os.path.abspath(__file__))]
LO, HI = 0.97, 1.03
REQ = 3 * 8 * 64  # BENCH_RUNS=3 × 8 completions × 64 tokens


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


def counts(c):
    v = c["counts"]
    return v if isinstance(v, dict) else ast.literal_eval(v)  # the harness stores a repr'd dict


def _mstate(c):
    m = c.get("machine") or {}
    la = m.get("loadavg")
    return dict(loadavg=la[0] if isinstance(la, list) and la else la, available_mb=m.get("available_mb"))


def decode(path):
    recs = json.load(open(path))
    prov, cells = recs[0], [c for c in recs[1:] if c.get("runs")]
    by = {(c["engine"], c["backend"], c["model"], c["depth"], c["config"]): c for c in cells}
    for c in recs[1:]:
        if not c.get("runs"):
            print(json.dumps(dict(file=os.path.basename(path), MISSING=True, engine=c.get("engine"), backend=c.get("backend"),
                                  model=c.get("model"), depth=c.get("depth"), config=c.get("config"), error=c.get("error"))))
    rows = []
    for (eng, be, m, d, cfg), g in sorted(by.items(), key=lambda kv: kv[0][1:]):
        if eng != "goinfer":
            continue
        for peer in ("ollama", "llamacpp"):
            p = by.get((peer, be, m, d, cfg))
            if not p:
                if any(c.get("engine") == peer for c in recs[1:]):
                    rows.append(dict(file=os.path.basename(path), backend=be, model=m, depth=d, config=cfg, peer=peer,
                                     outcome="VOID", why=f"{peer}: no runs recorded"))
                continue
            rs = [a / b for a, b in zip(g["runs"], p["runs"])]
            sg, sp = spread(g["runs"]), spread(p["runs"])
            capped = sg > 0.05 or sp > 0.05
            toks = (counts(g)["tokens"], counts(p)["tokens"])
            gates = (counts(g).get("token_gate", {}).get("verdict"), counts(p).get("token_gate", {}).get("verdict"))
            dpath = counts(g).get("decode_path")
            on_cpu = be != "cpu" and bool(dpath) and dpath.startswith("cpu")
            sv = [(counts(x).get("swap") or {}).get("swap_void") for x in (g, p)]
            ok = all(v in ("ok", "short") for v in gates) and not on_cpu and not any(sv)
            rows.append(dict(file=os.path.basename(path), backend=be, model=m, depth=d, config=cfg, peer=peer,
                             pairs=[round(r, 3) for r in rs], median=round(st.median(rs), 3), n=len(rs),
                             goinfer=[round(x, 1) for x in g["runs"]], peer_runs=[round(x, 1) for x in p["runs"]],
                             spread_g=round(100 * sg, 1), spread_p=round(100 * sp, 1),
                             outcome=("VOID" if not ok else outcome(rs, capped)), capped=capped, tokens=toks,
                             token_gates=gates, decode_path=dpath,
                             driver=(prov.get("gpu") or {}).get("driver"), ctx_pin=prov.get("ctx_pin"),
                             swap_void=sv, machine_g=_mstate(g), machine_p=_mstate(p)))
    return rows


def prefill(path):
    rows = []
    for c in json.load(open(path))["cells"]:
        g, p = c["goinfer"]["ttft_ms_all"], c["ollama"]["ttft_ms_all"]
        rs = [b / a for a, b in zip(g, p)]  # peer TTFT / goinfer TTFT
        sg, sp = spread(g), spread(p)
        capped = sg > 0.05 or sp > 0.05
        rows.append(dict(file=os.path.basename(path), model=c["model"], K=c["depth"], peer="ollama", pairs=[round(r, 3) for r in rs],
                         median=round(st.median(rs), 3), n=len(rs), goinfer_ms=g, peer_ms=p,
                         goinfer_exact_ms_median=c.get("goinfer_exact", {}).get("ttft_ms_median"),
                         spread_g=round(100 * sg, 1), spread_p=round(100 * sp, 1),
                         outcome=outcome(rs, capped), capped=capped))
    return rows


if __name__ == "__main__":
    for B in DIRS:
        for name in ("a-dense.json", "a-depth8000.json", "b-controls.json", "c-26b.json", "f-sampled.json", "e-cpu.json", "v-cpu.json", "v2-g20.json"):
            if not os.path.exists(os.path.join(B, name)):
                continue
            print(f"## {os.path.join(os.path.basename(B), name)}")
            for r in decode(os.path.join(B, name)):
                print(json.dumps(r))
        pf = os.path.join(B, "d-prefill.json")
        if os.path.exists(pf):
            print(f"## {os.path.join(os.path.basename(B), 'd-prefill.json')}")
            for r in prefill(pf):
                print(json.dumps(r))
