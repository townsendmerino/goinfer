#!/usr/bin/env python3
"""Apply PREREGISTERED.md's decision rule to the two arms' probe results.

    python3 ab_verdict.py A.json B.json     # exit 0 ADOPT, 3 PARK, 4 REJECT, 2 void (an arm is empty or incomplete)
    python3 ab_verdict.py --selftest

C = correct calls over the 21 tool-prompt replies, E = loops answered over them, K = control prompts answered in prose (of 3).
ADOPT if B.E >= A.E and B.C >= A.C and B.K >= A.K. REJECT if B.E <= A.E - 3 or B.K <= A.K - 2. Otherwise PARK. (ADOPT is tested first,
so a result that is both "no worse on E" and "worse on K by 2" is REJECT only if it fails ADOPT — it does, K being lower.)
"""
import json, sys

def stats(rows):
    tool = [r for r in rows if r["prompt"] != "control"]
    ctl = [r for r in rows if r["prompt"] == "control"]
    return {"n": len(rows), "C": sum(1 for r in tool if r["ok"]), "E": sum(1 for r in tool if r.get("loop") == "answered"),
            "K": sum(1 for r in ctl if r["ok"]), "tool": len(tool), "ctl": len(ctl)}

def decide(a, b):
    if b["E"] >= a["E"] and b["C"] >= a["C"] and b["K"] >= a["K"]:
        return "ADOPT"
    if b["E"] <= a["E"] - 3 or b["K"] <= a["K"] - 2:
        return "REJECT"
    return "PARK"  # including every result the numbers do not land in

def selftest():
    A = {"C": 21, "E": 19, "K": 3}
    cases = [
        (dict(C=21, E=19, K=3), "ADOPT"), (dict(C=21, E=21, K=3), "ADOPT"),
        (dict(C=21, E=18, K=3), "PARK"), (dict(C=20, E=17, K=3), "PARK"),
        (dict(C=21, E=16, K=3), "REJECT"), (dict(C=21, E=19, K=1), "REJECT"),
        (dict(C=20, E=19, K=3), "PARK"),   # E not worse but a call lost: not ADOPT, not REJECT -> PARK
        (dict(C=21, E=19, K=2), "PARK"),   # one control lost: not ADOPT, not REJECT -> PARK
    ]
    bad = [(b, want, decide(A, b)) for b, want in cases if decide(A, b) != want]
    print("selftest:", "ok" if not bad else bad)
    return 0 if not bad else 1

def main():
    if sys.argv[1:] == ["--selftest"]:
        sys.exit(selftest())
    a, b = (stats(json.load(open(p))) for p in sys.argv[1:3])
    print("arm A (goinfer's own order):", a)
    print("arm B (the template's order):", b)
    for name, s in (("A", a), ("B", b)):
        if s["tool"] != 21 or s["ctl"] != 3:
            print("VOID: arm %s is incomplete (%d tool replies, %d control; want 21 and 3)" % (name, s["tool"], s["ctl"]))
            sys.exit(2)
    v = decide(a, b)
    print("VERDICT:", v)
    sys.exit({"ADOPT": 0, "PARK": 3, "REJECT": 4}[v])

if __name__ == "__main__":
    main()
