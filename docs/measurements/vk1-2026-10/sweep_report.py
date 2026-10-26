#!/usr/bin/env python3
"""sweep_report.py: which aikit bumps in VK0's window changed a kernel fingerprint, and what that would have saved.

    python3 docs/measurements/vk1-2026-10/sweep_report.py SWEEP.jsonl.gz VK0-STEPS.jsonl.gz

SWEEP is aikit_sweep.py's output; VK0-STEPS is docs/measurements/vk0-2026-10/steps-<date>.jsonl.gz. A bump "changed" a
configuration when any kernel's fingerprint differs between its two versions there, or when either version does not
build the suite (not comparable, counted as changed). A commit's re-stalings are "saved" only when its sole trigger is
the aikit pin and the bump changed no fingerprint in any configuration.
"""
import collections
import gzip
import json
import sys


def load(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        return [json.loads(line) for line in f]


def main(sweep_path, steps_path):
    fp = {}
    configs = []
    for r in load(sweep_path):
        if r["config"] not in configs:
            configs.append(r["config"])
        fp[(r["aikit"], r["config"])] = r["report"]["kernels"] if r["ok"] else None
    built = collections.defaultdict(list)
    for (v, c), k in fp.items():
        if k is not None:
            built[c].append(v)
    print("versions that build the suite, per configuration:")
    for c in configs:
        print(f"  {c}: {len(built[c])} of {sum(1 for (v, cc) in fp if cc == c)}")

    def diff(a, b, c):
        ka, kb = fp.get((a, c)), fp.get((b, c))
        if ka is None or kb is None:
            return None
        return sorted(k for k in set(ka) | set(kb) if ka.get(k) != kb.get(k))

    steps = load(steps_path)
    trans = {}
    for s in steps:
        for t in s["triggers"] or []:
            if t.startswith("aikit:"):
                a, b = t[len("aikit:"):].split("->")
                trans.setdefault((a, b), []).append(s)
    print()
    print("aikit transition | per configuration: kernels whose fingerprint changed | commits | family re-stalings | sole-trigger re-stalings")
    tot = collections.Counter()
    for (a, b), ss in sorted(trans.items(), key=lambda kv: min(s["date"] for s in kv[1])):
        per = {c: diff(a, b, c) for c in configs}
        comparable = all(d is not None for d in per.values())
        unchanged = comparable and all(not d for d in per.values())
        fam = sum(len(s["stale_today"]) for s in ss)
        sole = [s for s in ss if all(t.startswith("aikit:") for t in s["triggers"])]
        sole_fam = sum(len(s["stale_today"]) for s in sole)
        desc = "; ".join(f"{c}: " + ("not comparable" if d is None else (", ".join(d) if d else "none")) for c, d in per.items())
        print(f"  {a} -> {b} | {desc} | {len(ss)} | {fam} | {sole_fam}")
        tot["transitions"] += 1
        tot["commits"] += len(ss)
        tot["fam"] += fam
        tot["sole_commits"] += len(sole)
        tot["sole_fam"] += sole_fam
        if comparable:
            tot["comparable"] += 1
            tot["comparable_sole_fam"] += sole_fam
            tot["comparable_sole_commits"] += len(sole)
        if unchanged:
            tot["unchanged"] += 1
            tot["saved_commits"] += len(sole)
            tot["saved_fam"] += sole_fam
    print()
    print(f"transitions: {tot['transitions']}; comparable in every configuration: {tot['comparable']}; "
          f"no fingerprint changed in any configuration: {tot['unchanged']}")
    print(f"re-staling commits with an aikit trigger: {tot['commits']} ({tot['fam']} family re-stalings); "
          f"aikit the sole trigger: {tot['sole_commits']} ({tot['sole_fam']})")
    print(f"of the sole-trigger commits on comparable transitions ({tot['comparable_sole_commits']}, "
          f"{tot['comparable_sole_fam']} re-stalings), a kernel-fingerprint key would have re-staled nothing at "
          f"{tot['saved_commits']} commits, {tot['saved_fam']} family re-stalings")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
