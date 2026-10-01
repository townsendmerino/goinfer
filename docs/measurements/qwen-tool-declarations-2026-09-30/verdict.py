#!/usr/bin/env python3
"""Apply PREREGISTERED.md's rule to the declarations A/B.

    python3 verdict.py DIR           # DIR holds decl-<model>-A.json / -B.json and their .log.serve files
    python3 verdict.py --selftest

Per model: C = correct calls (of 21), E = loops answered (of 21), K = control answered in prose (of 3).
REJECT if on ANY model B.C <= A.C-3 or B.E <= A.E-3 or B.K <= A.K-2.  ADOPT if on EVERY model B >= A on C, E and K.  Otherwise PARK.
VOID (not a result) if an arm is incomplete, or its load log does not say what the arm is (B: "tool declarations: the template's bytes"; A: not).
Exit status: 0 ADOPT, 3 PARK, 4 REJECT, 2 VOID.
"""
import json, os, sys

MODELS = ["coder05", "q25-7b", "q35-08b", "q35-9b"]
MARK = "tool declarations: the template's bytes"

def stats(rows):
    tool = [r for r in rows if r["prompt"] != "control"]
    ctl = [r for r in rows if r["prompt"] == "control"]
    return {"C": sum(1 for r in tool if r["ok"]), "E": sum(1 for r in tool if r.get("loop") == "answered"),
            "K": sum(1 for r in ctl if r["ok"]), "tool": len(tool), "ctl": len(ctl)}

def decide(pairs):
    """pairs: list of (A, B) stat dicts."""
    if any(b["C"] <= a["C"] - 3 or b["E"] <= a["E"] - 3 or b["K"] <= a["K"] - 2 for a, b in pairs):
        return "REJECT"
    if all(b["C"] >= a["C"] and b["E"] >= a["E"] and b["K"] >= a["K"] for a, b in pairs):
        return "ADOPT"
    return "PARK"

def selftest():
    def S(c, e, k): return {"C": c, "E": e, "K": k}
    cases = [
        ([(S(14, 9, 3), S(14, 9, 3)), (S(21, 20, 3), S(21, 21, 3))], "ADOPT"),
        ([(S(14, 9, 3), S(15, 11, 3))], "ADOPT"),
        ([(S(14, 9, 3), S(13, 9, 3))], "PARK"),                                  # one call lost: not ADOPT, not REJECT
        ([(S(14, 9, 3), S(14, 9, 2))], "PARK"),                                  # one control lost
        ([(S(14, 9, 3), S(11, 9, 3))], "REJECT"),                                # three calls lost
        ([(S(21, 20, 3), S(21, 17, 3))], "REJECT"),                              # three loops lost
        ([(S(21, 20, 3), S(21, 20, 1))], "REJECT"),                              # two controls lost
        ([(S(14, 9, 3), S(16, 12, 3)), (S(21, 20, 3), S(20, 20, 3))], "PARK"),   # a gain on one model does not buy a loss on another
        ([(S(14, 9, 3), S(16, 12, 3)), (S(21, 20, 3), S(21, 16, 3))], "REJECT"),
    ]
    bad = [(p, w, decide(p)) for p, w in cases if decide(p) != w]
    print("selftest:", "ok" if not bad else bad)
    return 0 if not bad else 1

def main():
    if sys.argv[1:] == ["--selftest"]:
        sys.exit(selftest())
    d = sys.argv[1]
    pairs, void = [], []
    for m in MODELS:
        arms = {}
        for arm in "AB":
            jp = os.path.join(d, "decl-%s-%s.json" % (m, arm)); sp = os.path.join(d, "decl-%s-%s.log.serve" % (m, arm))
            try:
                st = stats(json.load(open(jp)))
                log = open(sp).read()
            except Exception as e:
                void.append("%s arm %s: missing or unreadable (%s)" % (m, arm, e)); continue
            if st["tool"] != 21 or st["ctl"] != 3:
                void.append("%s arm %s: incomplete (%d tool replies, %d control)" % (m, arm, st["tool"], st["ctl"]))
            if (MARK in log) != (arm == "B"):
                void.append("%s arm %s: load log %s the declarations marker" % (m, arm, "has" if arm == "A" else "lacks"))
            arms[arm] = st
        if len(arms) == 2:
            pairs.append((arms["A"], arms["B"]))
            print("%-8s A %s | B %s" % (m, {k: arms["A"][k] for k in "CEK"}, {k: arms["B"][k] for k in "CEK"}))
    if void or len(pairs) != len(MODELS):
        print("VOID:", *void, sep="\n  ")
        sys.exit(2)
    v = decide(pairs)
    print("VERDICT:", v)
    sys.exit({"ADOPT": 0, "PARK": 3, "REJECT": 4}[v])

if __name__ == "__main__":
    main()
