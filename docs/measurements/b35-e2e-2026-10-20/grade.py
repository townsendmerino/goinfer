#!/usr/bin/env python3
"""Grades gate.json against the rules registered in README.md here, and prints one line per check and a VERDICT line.
Exit 0 on PASS, 1 on FAIL, 3 on PARKED.

  python3 grade.py <gate.json>
"""
import json, sys

TTFT_PASS, TTFT_PARK, TTFT_SLACK_S = 1.5, 2.5, 0.25


def main():
    rec = json.load(open(sys.argv[1]))
    fails, parks, lines = [], [], []

    def check(ok, name, detail):
        lines.append(f"{'ok  ' if ok else 'FAIL'} {name}: {detail}")
        if not ok:
            fails.append(name)

    if not rec["sizes"]:
        print("VERDICT FAIL (no sizes recorded)")
        return 1
    for row in rec["sizes"]:
        n = row["size"]
        B = {r["turn"]: r for r in row["B"]}
        cold = {r["turn"]: r for r in row["cold"]}
        base = {r["turn"]: r for r in row["base"]}
        A = [r for r in row["A"] if isinstance(r["turn"], int)]
        ctrl = [r for r in row["A"] if not isinstance(r["turn"], int)][0]

        # I1: the warm path (a restored state) equals a cold prefill of the same request, token for token and logprob for logprob.
        for t in (2, 3):
            check(B[t]["tokens"] == cold[t]["tokens"] and B[t]["tokens"], f"{n} I1 turn {t} warm = cold",
                  f"{len(B[t]['tokens'])} tokens, first difference at {first_diff(B[t]['tokens'], cold[t]['tokens'])}")
        # I2: B35 leaves cold output as the base build gives it.
        for t, mine in ((1, B[1]), (2, cold[2]), (3, cold[3])):
            check(mine["tokens"] == base[t]["tokens"] and mine["tokens"], f"{n} I2 turn {t} cold B35 = cold base",
                  f"first difference at {first_diff(mine['tokens'], base[t]['tokens'])}")
        # I3: the streamed run and the logprob run gave the same replies (greedy; else the record does not describe one conversation).
        for r in A:
            check(r["reply"] == B[r["turn"]]["reply"], f"{n} I3 turn {r['turn']} streamed reply = logprob reply", repr(r["reply"][:40]))
        # R1: turns 2 and 3 reused a prefix on both B35 servers; every cold request reused nothing.
        for t in (2, 3):
            ra, rb = A[t - 1]["reused"] or 0, B[t]["usage"].get("prefill_reused_tokens") or 0
            check(ra > 0 and rb > 0, f"{n} R1 turn {t} reused", f"streamed {ra} of {A[t - 1]['prompt_tokens']}, logprob {rb}")
        colds = [("control", ctrl["reused"] or 0), ("B turn 1", B[1]["usage"].get("prefill_reused_tokens") or 0)]
        colds += [(f"cold turn {t}", cold[t]["usage"].get("prefill_reused_tokens") or 0) for t in cold]
        colds += [(f"base turn {t}", base[t]["usage"].get("prefill_reused_tokens") or 0) for t in base]
        check(all(v == 0 for _, v in colds), f"{n} R1 cold requests reused nothing", ", ".join(f"{k} {v}" for k, v in colds))
        # T1: a warm turn's time to first token tracks the rows it still prefills: at most TTFT_PASS x the cold control's time scaled to
        # those rows, plus TTFT_SLACK_S. Between that and TTFT_PARK x: parked for Francis. Past it: FAIL.
        for t in (2, 3):
            r = A[t - 1]
            rows = r["prompt_tokens"] - (r["reused"] or 0)
            pred = ctrl["ttft_s"] * rows / ctrl["prompt_tokens"]
            ratio = (r["ttft_s"] - TTFT_SLACK_S) / pred if pred > 0 else float("inf")
            detail = (f"ttft {r['ttft_s']:.2f}s for {rows} rows prefilled; the cold control {ctrl['ttft_s']:.2f}s for {ctrl['prompt_tokens']} "
                      f"(predicted {pred:.2f}s, ratio {ratio:.2f}; {r['ttft_s'] / ctrl['ttft_s']:.3f} of cold)")
            if ratio <= TTFT_PASS:
                lines.append(f"ok   {n} T1 turn {t}: {detail}")
            elif ratio <= TTFT_PARK:
                lines.append(f"PARK {n} T1 turn {t}: {detail}")
                parks.append(f"{n} T1 turn {t}")
            else:
                check(False, f"{n} T1 turn {t}", detail)
    print("\n".join(lines))
    verdict = "FAIL" if fails else "PARKED" if parks else "PASS"
    print(f"VERDICT {verdict}" + (f" ({', '.join(fails or parks)})" if fails or parks else ""))
    return {"PASS": 0, "FAIL": 1, "PARKED": 3}[verdict]


def first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return f"token {i}: {x} vs {y}"
    return "none" if len(a) == len(b) else f"length {len(a)} vs {len(b)}"


if __name__ == "__main__":
    sys.exit(main())
