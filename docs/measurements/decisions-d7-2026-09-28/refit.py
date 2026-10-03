#!/usr/bin/env python3
"""D7: name the wrong input behind the two OFF PROJECTION five-question cells (decisions-d7-2026-09-28.md, section 4.4 asks for this before a number is quoted).

Recomputes, from the measured requests, the prefill rate (usage input tokens / time, decision arms, which generate nothing), the per-generated-token cost of the
constrained arms (time left after prefill at that rate, over the output tokens), the output length of the five-field answer, and the prompt length of the one-pass
arm against the state-only length the projection assumed. Run: python3 docs/measurements/decisions-d7-2026-09-28/refit.py [raw.jsonl]
"""
import json, os, statistics as st, sys
p = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "run-2026-10-02", "raw.jsonl")
rows = [json.loads(l) for l in open(p) if '"arm"' in l]
by = {}
for r in rows:
    by.setdefault(r["arm"], []).append(r)
mean = lambda arm, K, f: st.mean(f(r) for r in by[arm] if r["K"] == K)
print("prefill rate (decision arms generate nothing): input tokens / time")
for arm in ("decision", "decision5"):
    print("  " + arm.ljust(10) + "  ".join(f"K={K}: {mean(arm,K,lambda r:r['in'])/mean(arm,K,lambda r:r['t']):.0f} tok/s" for K in (32, 256, 1024, 4096) if any(r['K'] == K for r in by[arm])))
print("five decisions against five times one decision (is the premise 'each question pays a full prefill' right?)")
for K in (256, 1024, 4096):
    print(f"  K={K}: 5 x single {5*mean('decision',K,lambda r:r['t']):.2f} s, measured decision x5 {mean('decision5',K,lambda r:r['t']):.2f} s")
x = [r["in"] for r in by["decision"]]; y = [r["t"] for r in by["decision"]]
mx, my = st.mean(x), st.mean(y); b = sum((a - mx) * (c - my) for a, c in zip(x, y)) / sum((a - mx) ** 2 for a in x); c0 = my - b * mx
print("constrained arms: time after prefill (at the decision arms' fitted rate), per generated token; the projection assumed 16.5 ms")
for arm in ("schema", "tool", "schema5"):
    per = [((r["t"] - (b * r["in"] + c0)) / r["out"], r["out"]) for r in by[arm] if r["out"] > 0]
    print(f"  {arm:8} output tokens mean {st.mean(p[1] for p in per):.1f}; {1000*st.mean(p[0] for p in per):.1f} ms per generated token")
print("one-pass five-field arm: prompt length and output length against the projection (state tokens only; 23-26 output tokens)")
for K in (256, 1024, 4096):
    print(f"  K={K}: schema5 input {mean('schema5',K,lambda r:r['in']):.0f} tokens (state {mean('schema5',K,lambda r:r['state_tokens']):.0f}), output {mean('schema5',K,lambda r:r['out']):.1f} tokens, {mean('schema5',K,lambda r:r['t']):.2f} s")
