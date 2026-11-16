#!/usr/bin/env python3
"""G-M2 (README.md here): one served cell of served_msgboundary.py's conversation on a Mamba-2 hybrid, graded on counts.

  python3 served_mamba2_status_quo.py --serve <bin> --model <path> --out <dir>

Exit 0 PASS, 1 FAIL, 2 VOID (a cell started above the load bar).
"""
import argparse, importlib.util, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("sm", os.path.join(HERE, "../kv8-msgboundary-2026-11/served_msgboundary.py"))
sm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sm)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--serve", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--ctx", type=int, default=12288)
    a = p.parse_args()
    sm.h.CTX = a.ctx
    os.makedirs(a.out, exist_ok=True)
    rec, turns = sm.cell("S1", "main", a.serve, a.model, a.out)
    with open(os.path.join(a.out, "turns.jsonl"), "w") as f:
        for t in turns:
            f.write(json.dumps(t) + "\n")
    json.dump(rec, open(os.path.join(a.out, "cell.json"), "w"), indent=1)
    if rec["void"]:
        print("VERDICT: VOID (load over the bar)")
        return 2
    by = {t["req"]: t for t in turns}
    checks = {
        "no snapshot taken": not rec["snapshots_taken"],
        "r3 reuses 0 rows": (by["r3"]["reused"] or 0) == 0,
        "r2 reuses at least one row": (by["r2"]["reused"] or 0) > 0,
        "VmSwap at most 512 MB": rec["peak_swap_mb"] <= 512,
    }
    for k, v in checks.items():
        print(f"  {k}: {v}")
    ok = all(checks.values())
    print("VERDICT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
