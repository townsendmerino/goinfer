#!/usr/bin/env python3
"""Grades gate.json against the rules registered in README.md here; prints one line per check and a VERDICT line.
Exit 0 PASS, 1 FAIL, 3 PARKED, 4 Q1-ONLY, 2 VOID.

  python3 grade.py <gate.json>
"""
import json, sys

TTFT_PASS, TTFT_PARK, TTFT_SLACK_S = 1.5, 2.5, 0.25


def first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return f"token {i}: {x} vs {y}"
    return "none" if len(a) == len(b) else f"length {len(a)} vs {len(b)}"


def main():
    rec = json.load(open(sys.argv[1]))
    out = {"fail": [], "void": [], "park": [], "q1only": []}
    lines = []

    def check(ok, kind, name, detail):
        lines.append(f"{'ok  ' if ok else kind.upper():4} {name}: {detail}")
        if not ok:
            out[kind].append(name)

    A = {r["r"]: r for r in rec["A"] if isinstance(r["r"], int)}
    ctrl = [r for r in rec["A"] if not isinstance(r["r"], int)][0]
    B = {r["r"]: r for r in rec["B"]}
    cold = {r["r"]: r for r in rec["cold"]}
    reused = lambda r: (A[r]["reused"] or 0, B[r]["usage"].get("prefill_reused_tokens") or 0)

    # R3: the instrument. Every cold request reused nothing.
    colds = [("control", ctrl["reused"] or 0)] + [(f"cold r{r}", cold[r]["usage"].get("prefill_reused_tokens") or 0) for r in cold]
    check(all(v == 0 for _, v in colds), "void", "R3 cold requests reused nothing", ", ".join(f"{k} {v}" for k, v in colds))
    # Reported, not graded: the history is scripted, so a streamed reply that differs from the non-streamed one changes no later request.
    for r in A:
        same = A[r]["reply"] == B[r]["reply"]
        lines.append(f"{'note' if not same else 'ok  '} r{r} streamed reply {'=' if same else '!='} non-streamed reply"
                     + ("" if same else f" (first difference at character {first_diff(A[r]['reply'], B[r]['reply'])})"))
    # I1: a restored question equals a cold prefill of the same request: the whole served reply (reasoning, content and the calls' JSON) and
    # the same number of completion tokens, and every logprob where the route serves them (the tools route serves none).
    for r in (4, 6):
        w, c = B[r], cold[r]
        ok = (w["reply"] == c["reply"] and bool(w["reply"]) and w["tokens"] == c["tokens"]
              and w["usage"].get("completion_tokens") == c["usage"].get("completion_tokens"))
        check(ok, "fail", f"I1 r{r} warm = cold", f"{w['usage'].get('completion_tokens')} completion tokens, {len(w['tokens'])} logprobs; "
              f"reply first difference at {first_diff(w['reply'], c['reply'])}")
    # R1: the second question reuses a kept state on both servers (no other held point is at or before its divergence).
    ra, rb = reused(4)
    check(ra > 0 and rb > 0, "fail", "R1 r4 (Q2) reused a kept state", f"streamed {ra} of {A[4]['prompt_tokens']}, logprob {rb}")
    # R2: the third question reuses more than the second did: the kept point moved to Q2, past Q1's whole loop.
    sa, sb = reused(6)
    check(sa > ra and sb > rb, "q1only", "R2 r6 (Q3) reused more than r4", f"r6 streamed {sa}, logprob {sb}; r4 {ra}, {rb}")
    # T1 (B35's rule): a graded question's time to first token tracks the rows it still prefills.
    for r in (4, 6):
        rows = A[r]["prompt_tokens"] - (A[r]["reused"] or 0)
        pred = ctrl["ttft_s"] * rows / ctrl["prompt_tokens"]
        ratio = (A[r]["ttft_s"] - TTFT_SLACK_S) / pred if pred > 0 else float("inf")
        scaled_cold = ctrl["ttft_s"] * A[r]["prompt_tokens"] / ctrl["prompt_tokens"]
        detail = (f"ttft {A[r]['ttft_s']:.2f}s for {rows} rows prefilled (predicted {pred:.2f}s, ratio {ratio:.2f}); "
                  f"{A[r]['ttft_s'] / scaled_cold:.3f} of cold at its length")
        if ratio <= TTFT_PASS:
            lines.append(f"ok   T1 r{r}: {detail}")
        elif ratio <= TTFT_PARK:
            lines.append(f"PARK T1 r{r}: {detail}")
            out["park"].append(f"T1 r{r}")
        else:
            check(False, "fail", f"T1 r{r}", detail)
    print("\n".join(lines))
    for verdict, code, key in (("VOID", 2, "void"), ("FAIL", 1, "fail"), ("PARKED", 3, "park"), ("Q1-ONLY", 4, "q1only")):
        if out[key]:
            print(f"VERDICT {verdict} ({', '.join(out[key])})")
            return code
    print("VERDICT PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
