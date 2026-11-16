#!/usr/bin/env python3
"""KV8-overlap's T-O (README.md here): the T2 conversation on a small hybrid, main against the change, REPS repetitions of A B B A (2 pairs each), graded on the registered rule.

  python3 ab_pairs.py --main <bin> --change <bin> --model <path> --out <dir> [--reps 5] [--ctx 12288]

Reuses served_msgboundary.py's cell(). A pair is an A cell and an adjacent B cell: (A1, B1) and (B2, A2) of each repetition. Exit 0 PASS, 1 FAIL, 2 VOID, 3 PARKED.
"""
import argparse, importlib.util, json, os, statistics, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("sm", os.path.join(HERE, "../kv8-msgboundary-2026-11/served_msgboundary.py"))
sm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sm)
h = sm.h


def run_cell(label, which, binary, a):
    rec, ts = sm.cell(label, which, binary, a.model, a.out)
    if rec["void"]:
        print(f"VOID CELL {label}: re-run once, in order"); time.sleep(60)
        rec, ts = sm.cell(label + "r", which, binary, a.model, a.out)
    return rec, ts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--main", required=True)
    p.add_argument("--change", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--ctx", type=int, default=12288)
    a = p.parse_args()
    h.CTX = a.ctx
    os.makedirs(a.out, exist_ok=True)
    cells, pairs, replies_ok, notes = [], [], True, []
    for rep in range(a.reps):
        order = [("A", "main", a.main), ("B", "change", a.change), ("B", "change", a.change), ("A", "main", a.main)]
        got = []
        for i, (arm, which, binary) in enumerate(order):
            rec, ts = run_cell(f"{rep + 1}{'abba'[i]}", which, binary, a)
            if rec["void"]:
                print("VERDICT: VOID"); return 2
            cells.append((rec, ts)); got.append((arm, rec, ts))
        for x, y in ((got[0], got[1]), (got[2], got[3])):  # (A, B) in order of adjacency
            A, B = (x, y) if x[0] == "A" else (y, x)
            by = lambda ts, r: next(t for t in ts if t["req"] == r)
            pr = {r: by(B[2], r)["ttft_s"] / by(A[2], r)["ttft_s"] for r in ("r1", "r2", "r3", "r4")}
            for r in ("r1", "r2", "r3", "r4"):
                if by(B[2], r)["reply"] != by(A[2], r)["reply"]:
                    replies_ok = False; notes.append(f"rep {rep + 1}: {r}'s reply differs")
                if by(B[2], r)["reused"] != by(A[2], r)["reused"]:
                    notes.append(f"rep {rep + 1}: {r} reuses {by(B[2], r)['reused']} against main's {by(A[2], r)['reused']}")
            cost = lambda rec: sum(c["ms"] for c in rec["snapshot_cost"])
            pairs.append({"rep": rep + 1, "ratios": pr, "snap_ms_main": cost(A[1]), "snap_ms_change": cost(B[1])})
            print(f"[{time.strftime('%H:%M:%S')}] pair {len(pairs)}: r1 x{pr['r1']:.4f} r2 x{pr['r2']:.4f} r3 x{pr['r3']:.4f} r4 x{pr['r4']:.4f}; snapshot copy ms main {cost(A[1]):.0f} change {cost(B[1]):.0f}", flush=True)
    with open(os.path.join(a.out, "pairs.jsonl"), "w") as f:
        for pr in pairs:
            f.write(json.dumps(pr) + "\n")
        for rec, ts in cells:
            f.write(json.dumps(rec) + "\n")
            for t in ts:
                f.write(json.dumps({k: v for k, v in t.items() if k != "reply"} | {"reply_len": len(t["reply"])}) + "\n")
    faster = sum(1 for pr in pairs if pr["ratios"]["r1"] < 1)
    r3med = statistics.median(pr["ratios"]["r3"] for pr in pairs)
    r1med = statistics.median(pr["ratios"]["r1"] for pr in pairs)
    swap = max(rec.get("peak_swap_mb", 0) for rec, _ in cells)
    hwm = max(rec.get("peak_hwm_mb", 0) for rec, _ in cells)
    print(f"r1 faster in {faster} of {len(pairs)} pairs (median ratio {r1med:.4f}); r3 median ratio {r3med:.4f}; replies equal {replies_ok}; peak VmSwap {swap} MB, peak VmHWM {hwm} MB")
    for n in notes:
        print("  note:", n)
    n = len(pairs)
    if not replies_ok or faster < round(0.7 * n) or r3med > 1.05 or swap > h.SWAP_MB:
        verdict = "FAIL"
    elif faster >= round(0.9 * n) and r3med <= 1.02:
        verdict = "PASS"
    else:
        verdict = "PARKED"
    print("VERDICT:", verdict)
    return {"PASS": 0, "FAIL": 1, "PARKED": 3}[verdict]


if __name__ == "__main__":
    sys.exit(main())
