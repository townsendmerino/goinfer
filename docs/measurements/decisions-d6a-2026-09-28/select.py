#!/usr/bin/env python3
"""D6a's deterministic samples (docs/tasks/task-constrained-confidence.md, D6a pre-registration).

From SargeDev/jev-distill-corpus-v3 @ fc99c6357a9f89f7512c4a987314352addead049
(calibration.jsonl sha256 c5e232a0f8efdf9fafc4a829ad40917a6b3ace6ce4bc276b6210c947fb068155,
 ood.jsonl sha256 70d0f01742c2d7bbae8546794ce2e12d45ec86eedbd48e7afd31922b4f885781):

  calib-sample.jsonl: 1,500 calibration rows, yuri_v1 excluded (placeholder [0.5, 0.5] targets), stratified by kind
                      in the proportions of what is left (noul 4,363 / choice 3,418 / score 3,173).
  eval-sample.jsonl:  OOD rows (all openjev_v2, gold): 1,000 noul, 1,000 choice, and all 72 score.

Within a stratum, the rows with the lowest sha256(id) are taken. The corpus's row shape is decide's input shape.

usage: select.py <data dir> <out dir> [calib total, default 1500] [eval rows per noul/choice stratum, default 1000]

The optional sizes are the 2026-09-28 amendment (task doc, "D6a amendment"): 600 and 400. Rows are still the lowest
sha256(id) within each stratum, so an amended sample is the head of the original one, stratum by stratum.
"""
import hashlib, json, os, sys


def key(r):
    return hashlib.sha256(r["id"].encode()).hexdigest()


def take(rows, n):
    return sorted(rows, key=key)[:n]


def main(data, out, ncal=1500, nev=1000):
    cal = [json.loads(l) for l in open(os.path.join(data, "calibration.jsonl"))]
    ood = [json.loads(l) for l in open(os.path.join(data, "ood.jsonl"))]
    cal = [r for r in cal if r["source"] != "yuri_v1"]
    by = {k: [r for r in cal if r["kind"] == k] for k in ("noul", "choice", "score")}
    total = sum(len(v) for v in by.values())
    n = {k: round(ncal * len(v) / total) for k, v in by.items()}
    n["noul"] += ncal - sum(n.values())  # rounding remainder
    calib = [r for k in ("noul", "choice", "score") for r in take(by[k], n[k])]
    ev = {k: [r for r in ood if r["kind"] == k] for k in ("noul", "choice", "score")}
    evs = take(ev["noul"], nev) + take(ev["choice"], nev) + ev["score"]
    os.makedirs(out, exist_ok=True)
    for name, rows in (("calib-sample.jsonl", calib), ("eval-sample.jsonl", evs)):
        with open(os.path.join(out, name), "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("calib", {k: n[k] for k in n}, "eval", {k: sum(r["kind"] == k for r in evs) for k in ("noul", "choice", "score")},
          "ood split proportions", {k: len(v) for k, v in ev.items()})


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], *(int(a) for a in sys.argv[3:5]))
