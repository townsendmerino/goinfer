#!/usr/bin/env python3
"""Checks of gate_i1v2.py: each registered class on a synthetic run, a control run reported rather than voided (amendment 1),
and the wire compaction detector on I3's recorded runs (amendment 1). No model, no GPU.

  python3 test_gate_i1v2.py
"""
import glob, gzip, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gate_i1v2 as g  # noqa: E402


def rec(arm, reqs, seq, exit=0, passed=True, agent="codex", comps=None, err=False, decoy=False, ctx=15616):
    pr = [dict(path="/v1/responses", status=400 if ovf else 200, prompt_tokens=None if ovf else size, completion_tokens=comp,
               tools=tools, overflow=True if ovf else None, overflow_tokens=size if ovf else None) for size, ovf, comp, tools in reqs]
    return dict(id="x", agent=agent, task="chain", labels=dict(ctx=str(ctx), arm=arm, config="c"), exit=exit, passed=passed,
                ended_on_error=err, compactions=comps if comps is not None else sum(1 for e in seq if e[0] == "compact"),
                wire=dict(per_request=pr, n_halted=0, n_5xx=0), chain=dict(seq=seq, last=23, decoy_content=decoy), void_halted=None)


def t(size, ovf=False, comp=300, tools=9):
    return (size, ovf, comp, tools)


grow = [t(5669 + 1000 * i) for i in range(10)]                        # history grows a file a step, to 14,669
reads = [["read", i] for i in range(10)]
reads2 = [["read", i] for i in range(10, 24)]
after = [t(7000 + 1000 * i) for i in range(9)]                        # compacted to about 7,000, grows again
ok = rec("graded", grow + [t(15700, True, 0)] + after + [t(15650, True, 0)] + [t(7200 + 1000 * i) for i in range(5)],
         reads + [["compact"]] + reads2[:8] + [["compact"]] + reads2[8:])
two = rec("graded", grow + [t(15700, True, 0)] + after + [t(15650, True, 0), t(8000)], reads + [["compact"], ["compact"]] + reads2, comps=2)
CASES = {
    "PASS": ok,
    "VOID never reached": rec("graded", grow, reads, passed=False),
    "VOID agent first": rec("graded", grow[:6] + [t(7000)] + grow[6:] + [t(15700, True, 0)], reads + [["compact"]]),
    "VOID incurable": rec("graded", grow[:3] + [t(17000, True, 0)], reads[:3]),
    "VOID off-chain": rec("graded", grow + [t(15700, True, 0)], reads, decoy=True),
    "FAIL ended on overflow": rec("graded", grow + [t(15700, True, 0)], reads + [["compact"]], exit=1, passed=False, err=True),
    "FAIL two compactions": two,
    # amendment 1: a control that compacts by itself and never meets an overflow reports "agent first", not "never reached"
    "REPORT agent first": rec("control", grow + [t(7000)] + after, reads + [["compact"]] + reads2[:8]),
    "REPORT goinfer overflow first": rec("control", grow + [t(15700, True, 0)] + after, reads + [["compact"]] + reads2[:8]),
    "REPORT neither": rec("control", grow, reads, passed=False),
}
for want, r in CASES.items():
    v, d = g.grade(r)
    got = v if v != "REPORT" else f"REPORT {d['first']}"
    assert got.split()[0] == want.split()[0] and (v != "REPORT" or got == want), (want, v, d)
    print(f"ok  {want:32} -> {v} {d.get('why') or d.get('first', '')}")

# the cross-check: the same PASS run with a silent agent (no own compaction events) is contested, and the fallback grades it
silent = json.loads(json.dumps(ok)); silent["chain"]["seq"] = [e for e in silent["chain"]["seq"] if e[0] != "compact"]; silent["compactions"] = 0
v, d = g.grade(silent)
print(f"ok  silent agent, own count         -> {v} {d.get('contested', '')}")
v2, d2 = g.grade(silent, fallback=True)
print(f"ok  silent agent, wire fallback     -> {v2} wire compactions {d2['compactions_wire']}")
assert d2["compactions_wire"] == 2 and v2 == "PASS"
# the two counts disagree on the verdict: the agent reports one compaction, the wire shows two in a row with no read between
c = rec("graded", grow + [t(15700, True, 0), t(7000), t(6200)] + [t(7200 + 1000 * i) for i in range(1, 15)] + [t(15650, True, 0), t(7000)],
        reads + [["compact"]] + reads2 + [["compact"]], comps=2)
v, d = g.grade(c)
print(f"ok  counts disagree                 -> {v}, contested: {d.get('contested')}")
assert v == "PASS" and d.get("contested")
assert "contested" in g.verdicts([(c, v, d), (c, v, d)])[0]

# amendment 1's detector on I3's recorded runs: opencode reported no compaction of its own in any of them; the wire finds one
# wherever the history reset to near the fixed part, and none where a summarisation request left the history in place.
EXPECT = {("q3-4b-16k", "write", 1): 1, ("q3-4b-16k", "write", 2): 1, ("q3-4b-16k", "fix", 1): 1,
          ("q3-4b-11k-opencode-ovf", "edit", 2): 1, ("q3-4b-11k-opencode-ovf", "edit", 3): 1,
          ("q3-4b-11k-opencode-ovf", "fix", 1): 1, ("q3-4b-11k-opencode-ovf", "fix", 2): 4, ("q3-4b-11k-opencode-ovf", "fix", 3): 1}
n = 0
for f in sorted(glob.glob(os.path.join(HERE, "..", "i3-night-2026-10-13", "job-*", "records.jsonl.gz"))):
    for line in gzip.open(f, "rt"):
        r = json.loads(line)
        if r["agent"] != "opencode":
            continue
        key = (r["labels"]["config"], r["task"], r["rep"])
        got = len(g.wire_compactions(r))
        assert got == EXPECT.get(key, 0), (key, got)
        assert r["compactions"] == 0
        n += 1
assert n == 29, n
print(f"ok  wire detector on I3's {n} recorded opencode runs: {sum(EXPECT.values())} compactions where the history reset, none elsewhere")
print("all checks hold")
