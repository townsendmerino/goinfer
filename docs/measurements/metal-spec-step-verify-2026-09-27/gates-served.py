#!/usr/bin/env python3
"""Metal spec verify on the step kernels: the served gates (docs/tasks/task-concurrency-2026-09.md MC4, registered in
8bb80b1f). Usage: gates-served.py DIR (copy-15b.json chat-15b.json [copy-7b.json chat-7b.json]). Keys <arm>_<round>,
arm in plain / specold / specnew; one client."""
import json, os, statistics, sys


def cells(path):
    R = json.load(open(path))["results"]
    return {k: (next(iter(v.values())) if isinstance(v, dict) and "aggregate_tok_s" not in v and v else v) for k, v in R.items() if v}


def replies(cell):
    out = []
    for c in cell["per_client"]:
        for t in (c["turns"] if isinstance(c, dict) and "turns" in c else c):
            out.append(t.get("content_sha"))
    return out


def grade(path, name):
    C = cells(path)
    same = {"specold": 0, "specnew": 0}
    total = 0
    ratio = {"specold": [], "specnew": []}
    for r in (1, 2, 3):
        p = C.get(f"plain_{r}")
        if not p:
            continue
        rp = replies(p)
        for arm in ("specold", "specnew"):
            s = C.get(f"{arm}_{r}")
            if not s:
                continue
            rs = replies(s)
            same[arm] += sum(a == b for a, b in zip(rp, rs))
            ratio[arm].append(s["aggregate_tok_s"] / p["aggregate_tok_s"])
        total += len(rp)
        print(f"  {name} round {r}: plain {p['aggregate_tok_s']:.2f} tok/s; " + "; ".join(
            f"{arm} {C[f'{arm}_{r}']['aggregate_tok_s']:.2f} ({ratio[arm][-1]:.3f}x)" for arm in ("specold", "specnew") if f"{arm}_{r}" in C))
    med = {arm: statistics.median(v) if v else float("nan") for arm, v in ratio.items()}
    print(f"  {name}: spec-new / plain median {med['specnew']:.3f}x, spec-old / plain {med['specold']:.3f}x; "
          f"replies equal to plain: spec-new {same['specnew']}/{total}, spec-old {same['specold']}/{total}")
    return med, same["specnew"] == total and total > 0


def main(d):
    ok = True
    res = {}
    for w in ("copy", "chat"):
        for tag in ("15b", "7b"):
            p = os.path.join(d, f"{w}-{tag}.json")
            if os.path.exists(p):
                res[(w, tag)] = grade(p, f"{w} {tag}")
                if tag == "15b":
                    ok &= res[(w, tag)][1]
    print(f"GATE 1 every reply equals plain's (1.5B, hard): {'PASS' if ok else 'FAIL'}")
    if ("copy", "15b") in res:
        c = res[("copy", "15b")][0]["specnew"]
        print(f"GATE 2 copy 1.5B spec-new / plain {c:.3f}x (>= 1.25 ships; 1.03-1.25 owner): {'PASS' if c >= 1.25 else 'owner' if c >= 1.03 else 'FAIL'}")
    if ("chat", "15b") in res:
        h = res[("chat", "15b")][0]["specnew"]
        print(f"GATE 3 chat 1.5B spec-new / plain {h:.3f}x (>= 0.97, hard): {'PASS' if h >= 0.97 else 'FAIL'}")


if __name__ == "__main__":
    main(sys.argv[1])
