#!/usr/bin/env python3
"""Grader for the D13 multi-question addendum (decisions-d13-clef-multiquestion-2026-10-03.md). The rule is the record's section 2; this applies it.

    grade_mq.py --ref probs_f32.jsonl --f32 rows_f32.jsonl --int8 rows_int8int8.jsonl
"""
import argparse, json, math

def rows(p):
    return {r["id"]: r for r in (json.loads(l) for l in open(p) if l.strip())}

def kl(p, q):
    return sum(a * math.log(a / max(b, 1e-300)) for a, b in zip(p, q) if a > 0)

def am(x):
    return max(range(len(x)), key=lambda i: x[i])

def compare(ref, arm):
    out, bad = [], []
    for rid, r in ref.items():
        a = arm.get(rid)
        if a is None:
            bad.append(f"{rid}: no row")
            continue
        if a["n_tokens"] != r["n_tokens"]:
            bad.append(f"{rid}: {a['n_tokens']} tokens, reference {r['n_tokens']}")
        for pos, (rq, aq) in enumerate(zip(r["questions"], a["questions"]), 1):
            if rq["id"] != aq["id"] or rq["option_ids"] != aq["option_ids"]:
                bad.append(f"{rid} q{pos}: ids/options differ")
            out.append(dict(rec=rid, pos=pos, qid=rq["id"], dp=max(abs(x - y) for x, y in zip(rq["probs"], aq["probs"])),
                            kl=kl(rq["probs"], aq["probs"]), same=am(rq["probs"]) == am(aq["probs"]),
                            margin=sorted(rq["probs"], reverse=True)[0] - (sorted(rq["probs"], reverse=True)[1] if len(rq["probs"]) > 1 else 0)))
        if len(r["questions"]) != len(a["questions"]):
            bad.append(f"{rid}: {len(a['questions'])} answers for {len(r['questions'])} questions")
    return out, bad

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True); ap.add_argument("--f32"); ap.add_argument("--int8")
    a = ap.parse_args()
    ref = rows(a.ref)
    print(f"reference: {len(ref)} records, {sum(len(r['questions']) for r in ref.values())} questions")
    verdicts = {}
    for name, p in (("f32", a.f32), ("int8int8", a.int8)):
        if not p:
            continue
        cs, bad = compare(ref, rows(p))
        n = len(cs)
        mdp, mkl = max(c["dp"] for c in cs), sum(c["kl"] for c in cs) / n
        top1 = sum(c["same"] for c in cs)
        print(f"\n## {name}: {n} questions in {len(ref)} records")
        for b in bad:
            print("  STRUCTURAL:", b)
        print(f"  max |dP| {mdp:.3g}; mean KL {mkl:.5f}; top-1 agreement {top1}/{n}")
        for pos in range(1, 6):
            xs = [c for c in cs if c["pos"] == pos]
            if xs:
                print(f"    position {pos} ({xs[0]['qid']}): mean KL {sum(c['kl'] for c in xs)/len(xs):.5f}, max |dP| {max(c['dp'] for c in xs):.3g}, top-1 {sum(c['same'] for c in xs)}/{len(xs)}")
        for c in cs:
            if not c["same"]:
                print(f"    top-1 flip: {c['rec']} q{c['pos']} ({c['qid']}), reference top-two margin {c['margin']:.3f}")
        if name == "f32":
            v = "FAIL" if bad or top1 < n or mdp > 1e-3 else ("PASS" if mdp <= 1e-4 else "AMBIGUOUS")
            verdicts[name] = f"{v} (max |dP| {mdp:.3g} vs 1e-4 / 1e-3; top-1 {top1}/{n}; structural problems {len(bad)})"
        else:
            worst_pos = max((sum(c['kl'] for c in cs if c['pos'] == p) / max(1, sum(1 for c in cs if c['pos'] == p))) for p in range(1, 6))
            v = "FAIL" if bad or mkl > 0.06 or worst_pos > 0.06 else ("PASS" if mkl <= 0.03 else "AMBIGUOUS")
            verdicts[name] = f"{v} (mean KL {mkl:.4f} vs 0.03 / 0.06; worst position mean KL {worst_pos:.4f} vs 0.06; top-1 {top1}/{n} reported, no bar; structural problems {len(bad)})"
    print("\nVERDICTS")
    for k, v in verdicts.items():
        print(f"  {k:9s} {v}")

if __name__ == "__main__":
    main()
