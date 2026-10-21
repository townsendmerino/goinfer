#!/usr/bin/env python3
"""compare_lp.py DIR A B... — first-position top-20 logprobs of res-A.json against res-B.json: the largest |difference| over the tokens both list."""
import json, sys
d = sys.argv[1]
def top(n):
    c = json.load(open(f"{d}/res-{n}.json"))[-1]["logprobs"]["content"][0]
    return c["token"], c["logprob"], {t["token"]: t["logprob"] for t in c["top_logprobs"]}
a = sys.argv[2]
for b in sys.argv[3:]:
    A, B = top(a)[2], top(b)[2]; sh = set(A) & set(B)
    print(f"{a} vs {b}: shared {len(sh)}/20, max|dlogprob| {max(abs(A[k]-B[k]) for k in sh):.3e}, top-20 bit-identical {A == B}")
