#!/usr/bin/env python3
"""E-P06's served grade (docs/tasks/task-metal-audit-2026-10.md, "E-P06: built, pending its grade"; pre-registered there
before any graded run). Usage: gates-ep06.py DIR, reading copy-<m>.json and chat-<m>.json for m in 15b, 05b, 7b. Keys
<arm>_<round>, arm in plain / specold / specnew, one client, greedy. specnew is --spec ngram with the model's own verify
cost curve (calibrateVerifyCost), specold the same with the shipped constant; plain is the do-nothing arm."""
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
    same, total, new_old, new_plain = 0, 0, [], []
    for r in range(1, 6):
        p, o, n = C.get(f"plain_{r}"), C.get(f"specold_{r}"), C.get(f"specnew_{r}")
        if not (p and o and n):
            if p or o or n:
                print(f"  {name} round {r}: missing an arm")
            continue
        rp = replies(p)
        for s in (o, n):
            same += sum(a == b for a, b in zip(rp, replies(s)))
            total += len(rp)
        new_old.append(n["aggregate_tok_s"] / o["aggregate_tok_s"])
        new_plain.append(n["aggregate_tok_s"] / p["aggregate_tok_s"])
        print(f"  {name} round {r}: plain {p['aggregate_tok_s']:.2f}, spec-old {o['aggregate_tok_s']:.2f}, spec-new "
              f"{n['aggregate_tok_s']:.2f} tok/s; new/old {new_old[-1]:.3f}x, new/plain {new_plain[-1]:.3f}x")
    med = lambda v: statistics.median(v) if v else float("nan")
    print(f"  {name}: new/old median {med(new_old):.3f}x (range {min(new_old, default=float('nan')):.3f}-"
          f"{max(new_old, default=float('nan')):.3f}), new/plain median {med(new_plain):.3f}x; replies equal to plain {same}/{total}")
    return med(new_old), total > 0 and same == total


def main(d):
    res, ident = {}, True
    for tag in ("15b", "05b", "7b"):
        for w in ("copy", "chat"):
            p = os.path.join(d, f"{w}-{tag}.json")
            if os.path.exists(p):
                res[(w, tag)], ok = grade(p, f"{w} {tag}")
                ident &= ok
    print(f"PRECONDITION every spec reply equals plain's, every model and workload: {'PASS' if ident else 'FAIL (kills)'}")
    h, c = res.get(("chat", "15b"), float("nan")), res.get(("copy", "15b"), float("nan"))
    verdict = "SHIPS" if ident and h >= 1.02 and c >= 0.98 else "KILLED"
    print(f"E-P06 METRIC chat 1.5B spec-new/spec-old {h:.3f}x (>= 1.02), copy 1.5B {c:.3f}x (>= 0.98): {verdict}")


if __name__ == "__main__":
    main(sys.argv[1])
