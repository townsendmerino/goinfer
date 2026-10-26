#!/usr/bin/env python3
"""summarize.py: the VK0 tables from replay.go's per-commit records.

    python3 docs/measurements/vk0-2026-10/summarize.py docs/measurements/vk0-2026-10/steps-<date>.jsonl.gz

Reads one JSON object per commit (replay.go's `step`) and prints the counts in vk0-2026-10.md. No git, no network.
"""
import collections
import gzip
import json
import re
import sys

UNCHANGED_RE = re.compile(r"output unchanged|bit-identical|bit identical|weights unchanged|non-numeric", re.I)


def load(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        return [json.loads(line) for line in f]


def L(x):
    return x or []


def trigger_class(t):
    """'set:core:decoder/model.go' -> 'core decoder/model.go'; 'own:decoder/x.go' -> 'own'; 'aikit:a->b' -> 'aikit'."""
    k = t.split(":")
    if k[0] == "set":
        return k[1] + " " + k[2]
    return k[0]


def main(path):
    S = load(path)
    out = []
    p = out.append
    merges = [s for s in S if s["parents"] > 1]
    dates = sorted(s["date"] for s in S)
    p(f"window: {dates[0]} .. {dates[-1]} (committer dates), {len(S)} commits ({len(merges)} merges), last {S[-1]['commit'][:8]}")
    errs = [s for s in S if L(s.get("errors"))]
    p(f"commits with a family that could not be hashed: {len(errs)}")

    # The replay against history: at every commit that wrote a deps_hash, does the committed hash equal the replay of
    # the function of the day for every validated family?
    wrote = [s for s in S if L(s["refreshed"])]
    clean = [s for s in wrote if not L(s["committed_stale_of_day"])]
    p("")
    p("== replay vs history ==")
    p(f"commits that wrote a deps_hash: {len(wrote)}; of them, every validated family's committed hash equals the replay: {len(clean)}")
    for s in wrote:
        if L(s["committed_stale_of_day"]):
            p(f"  differs: {s['commit'][:8]} {s['date'][:10]} P{s['parents']} {len(s['committed_stale_of_day'])} families: {s['subject'][:90]}")
    carried = [s for s in S if L(s["committed_stale_of_day"])]
    p(f"commits whose tree fails TestParityManifest_fresh (a validated family stale as committed): {len(carried)} of {len(S)}")

    def block(title, key):
        R = [s for s in S if L(s[key])]
        fam = sum(len(s[key]) for s in R)
        p("")
        p(f"== {title} ==")
        p(f"commits that re-stale at least one family: {len(R)} ({sum(s['parents'] > 1 for s in R)} merges); family re-stalings: {fam}")
        hist = collections.Counter(len(s[key]) for s in R)
        p("families per re-staling commit: " + ", ".join(f"{n}:{c}" for n, c in sorted(hist.items())))
        return R, fam

    R, fam = block("today's function (Go tokens), every commit", "stale_today")
    Rd, famd = block("the function of the day (raw bytes before 5ff89161)", "stale_of_day")
    p(f"re-stalings the token hash removed in the window: {famd - fam} family re-stalings, {len(Rd) - len(R)} commits")

    # Triggers, today's function. A commit's classes are the set of its triggers' classes.
    p("")
    p("== what triggered each re-staling commit (today's function) ==")
    by_file = collections.Counter()
    by_file_fam = collections.Counter()
    sole = collections.Counter()
    sole_fam = collections.Counter()
    for s in R:
        classes = set()
        for t, n in (s["trigger_families"] or {}).items():
            c = trigger_class(t)
            classes.add(c)
            by_file[c] += 1
        for c in classes:
            by_file_fam[c] += len(s["stale_today"])
        key = " + ".join(sorted(classes)) if classes else "(merge: a combination neither parent held)"
        if len(classes) == 1 or not classes:
            sole[key] += 1
            sole_fam[key] += len(s["stale_today"])
        else:
            sole["(several inputs)"] += 1
            sole_fam["(several inputs)"] += len(s["stale_today"])
    p("input | commits it is among the triggers of | family re-stalings in those commits")
    for c, n in by_file.most_common():
        p(f"  {c} | {n} | {by_file_fam[c]}")
    p("sole trigger | commits | family re-stalings")
    for c, n in sole.most_common():
        p(f"  {c} | {n} | {sole_fam[c]}")

    # aikit
    p("")
    p("== aikit pin changes ==")
    ak = [s for s in R if any(t.startswith("aikit:") for t in s["triggers"] or [])]
    ak_nm = [s for s in ak if s["parents"] == 1]
    p(f"re-staling commits with an aikit trigger: {len(ak)} ({len(ak_nm)} non-merge); family re-stalings: {sum(len(s['stale_today']) for s in ak)}")
    versions = sorted({t.split(":", 1)[1] for s in ak for t in s["triggers"] if t.startswith("aikit:")})
    p(f"distinct aikit transitions: {len(versions)}")
    for s in ak_nm:
        tr = [t.split(":", 1)[1] for t in s["triggers"] if t.startswith("aikit:")]
        flag = " [subject says unchanged]" if UNCHANGED_RE.search(s["subject"]) else ""
        p(f"  {s['commit'][:8]} {s['date'][:10]} {','.join(tr)} {len(s['stale_today'])} fam{flag}: {s['subject'][:80]}")

    # Did a golden move?
    p("")
    p("== did a forward golden move? ==")
    gm = [s for s in S if s["parents"] == 1 and any(g.startswith("M ") for g in L(s["goldens"]))]
    p(f"non-merge commits that modify an existing forward golden: {len(gm)}")
    for s in gm:
        mods = [g[2:] for g in s["goldens"] if g.startswith("M ")]
        p(f"  {s['commit'][:8]} {s['date'][:10]} re-stales {len(L(s['stale_today']))}: {', '.join(mods)[:110]} | {s['subject'][:70]}")
    rg = [s for s in R if s in gm]
    p(f"re-staling commits that also modify a forward golden: {len(rg)}; their family re-stalings: {sum(len(s['stale_today']) for s in rg)}")
    ga = [s for s in R if s["parents"] == 1 and any(g.startswith("A ") for g in L(s["goldens"])) and s not in gm]
    p(f"re-staling commits that only add a new golden (a new feature's reference, no existing golden moved): {len(ga)}")
    mm = sum(len(L(s["metrics_moved"])) for s in wrote)
    p(f"family deps_hash writes: {sum(len(s['refreshed']) for s in wrote)}; of them with metrics, validated_at, date or status changed too: {mm}")
    for s in wrote:
        if L(s["metrics_moved"]):
            p(f"  {s['commit'][:8]} {s['date'][:10]} {s['metrics_moved']}: {s['subject'][:90]}")

    # The before number.
    nomove = [s for s in R if s not in gm and not L(s["metrics_moved"])]
    p("")
    p("== the before number ==")
    p(f"family re-stalings by today's key: {fam} in {len(R)} commits")
    p(f"  of them in commits that moved no existing forward golden and changed no family's recorded metrics: "
      f"{sum(len(s['stale_today']) for s in nomove)} in {len(nomove)} commits")
    p(f"  in commits that did move a golden or a recorded metric: {fam - sum(len(s['stale_today']) for s in nomove)} in {len(R) - len(nomove)} commits")
    p(f"commits that wrote a deps_hash (the refreshes the key made someone run): {len(wrote)}")
    print("\n".join(out))


if __name__ == "__main__":
    main(sys.argv[1])
