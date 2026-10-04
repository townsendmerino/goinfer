#!/usr/bin/env python3
"""Joint versus single (characterization, no pass/fail bar): decisions-d13-clef-multiquestion-2026-10-03.md section 4.

    grade_joint_single.py --joint probs_f32.jsonl --single probs_single_f32.jsonl

For every question of every five-question record: the probabilities when it is asked with the other four (the existing joint reference run) against the same question asked
alone (the single run). Reported: KL(single || joint), max |dP|, top-1 agreement, by position and by kind, and each flip with the single run's top-two margin.
"""
import argparse, json, math

def rows(p):
    return {r["id"]: r for r in (json.loads(l) for l in open(p) if l.strip())}

def kl(p, q):
    return sum(a * math.log(a / max(b, 1e-300)) for a, b in zip(p, q) if a > 0)

def am(x):
    return max(range(len(x)), key=lambda i: x[i])

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--joint", required=True); ap.add_argument("--single", required=True)
    a = ap.parse_args()
    joint, single = rows(a.joint), rows(a.single)
    cs = []
    for rid, r in joint.items():
        for pos, q in enumerate(r["questions"], 1):
            s = single.get(f"{rid}::{q['id']}")
            if s is None:
                print(f"MISSING single row for {rid}::{q['id']}"); continue
            sq = s["questions"][0]
            if sq["id"] != q["id"] or sq["option_ids"] != q["option_ids"]:
                print(f"STRUCTURAL: {rid}::{q['id']}: options {sq['option_ids']} vs {q['option_ids']}")
            ps = sorted(sq["probs"], reverse=True)
            cs.append(dict(rec=rid, pos=pos, qid=q["id"], kl=kl(sq["probs"], q["probs"]), dp=max(abs(x - y) for x, y in zip(sq["probs"], q["probs"])),
                           same=am(sq["probs"]) == am(q["probs"]), margin=ps[0] - (ps[1] if len(ps) > 1 else 0), ntok_joint=r["n_tokens"], ntok_single=s["n_tokens"]))
    n = len(cs)
    print(f"{n} questions; mean KL(single||joint) {sum(c['kl'] for c in cs)/n:.5f}, median {sorted(c['kl'] for c in cs)[n//2]:.5f}, max {max(c['kl'] for c in cs):.5f}; "
          f"max |dP| {max(c['dp'] for c in cs):.4f}; top-1 agreement {sum(c['same'] for c in cs)}/{n}")
    print("\nby position (the question's place in the joint record):")
    for pos in range(1, 6):
        xs = [c for c in cs if c["pos"] == pos]
        if xs:
            print(f"  {pos} {xs[0]['qid']:8s} mean KL {sum(c['kl'] for c in xs)/len(xs):.5f}  max |dP| {max(c['dp'] for c in xs):.4f}  top-1 {sum(c['same'] for c in xs)}/{len(xs)}")
    print("\nby K (state length):")
    for K in (256, 1024):
        xs = [c for c in cs if f"-K{K}-" in c["rec"]]
        if xs:
            print(f"  K={K:4d} mean KL {sum(c['kl'] for c in xs)/len(xs):.5f}  max |dP| {max(c['dp'] for c in xs):.4f}  top-1 {sum(c['same'] for c in xs)}/{len(xs)}  (n={len(xs)})")
    print("\nper question:")
    for c in cs:
        print(f"  {c['rec']:14s} {c['qid']:8s} KL {c['kl']:.5f}  max|dP| {c['dp']:.4f}  {'same' if c['same'] else 'FLIP'}  (tokens joint {c['ntok_joint']}, single {c['ntok_single']})")
    for c in cs:
        if not c["same"]:
            print(f"top-1 flip: {c['rec']} {c['qid']}: the single run's top-two margin {c['margin']:.3f}")

if __name__ == "__main__":
    main()
