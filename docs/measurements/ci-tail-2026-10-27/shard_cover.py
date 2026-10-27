#!/usr/bin/env python3
"""shard_cover.py: PW2a's gates G1 to G3 (docs/tasks/task-subsystems-2026-10.md §12.2).

Reads .github/workflows/ci.yml at two git refs, takes every `go test` line in the decoder shards and test-rest, and maps
each top-level test of ./decoder and ./internal/sampling to the steps that select it (-run, -skip, the package list).

  G1: on both sides every test is selected by exactly one -race step; both sides select the same tests; only the named
      movers change step; the no-race sampler-gates step selects the same tests on both sides.
  G2: every test that ran under -race before runs under -race after. G1's exactly-one rule on the after side already
      implies it; it is printed on its own so a red run names the cause.
  G3: the set of job ids and job names is identical.

Usage (from the repository root, go on PATH):
  python3 docs/measurements/ci-tail-2026-10-27/shard_cover.py BEFORE_REF AFTER_REF [MOVER ...]
Exit 0 when all three pass. The test lists come from `go test -list` on the working tree, once: the gates compare two
ci.yml files over the same tests.
"""

import re
import shlex
import subprocess
import sys

import yaml

PKGS = {"./decoder/": "decoder", "./internal/sampling/": "sampling"}
JOBS = ("test-decoder-a", "test-decoder-b", "test-decoder-c", "test-decoder-d", "test-decoder-tail", "test-rest")


def list_tests(pkg):
    out = subprocess.run(["go", "test", "-list", ".", "-tags", "goinfer_testhooks", pkg],
                         check=True, capture_output=True, text=True).stdout
    names = [l for l in out.splitlines() if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", l)]
    return [n for n in names if not n.startswith("Benchmark")]


def workflow(ref):
    return yaml.safe_load(subprocess.run(["git", "show", f"{ref}:.github/workflows/ci.yml"],
                                         check=True, capture_output=True, text=True).stdout)


def go_test_lines(wf):
    """(job, step name, argv) for each `go test` that runs tests (not `go test -list`) in the shard jobs."""
    for job in JOBS:
        for st in wf["jobs"][job]["steps"]:
            for line in st.get("run", "").splitlines():
                line = line.strip()
                if not line.startswith("go test ") or " -list " in line:
                    continue
                argv = shlex.split(line.split(">")[0])
                yield job, st["name"], argv


def selector(argv):
    """race, run regex, skip regex, the packages of ours it covers."""
    race, run, skip, pkgs = False, None, None, set()
    flat = []
    for a in argv[2:]:  # -run=X reads as -run X; a -race=<bool> form is refused rather than guessed
        assert not a.startswith("-race="), a
        flat += a.split("=", 1) if a.split("=", 1)[0] in ("-run", "-skip", "-timeout", "-tags") else [a]
    argv = argv[:2] + flat
    i = 2
    while i < len(argv):
        a = argv[i]
        if a == "-race":
            race = True
        elif a in ("-run", "-skip", "-timeout", "-tags"):
            if a == "-run":
                run = argv[i + 1]
            if a == "-skip":
                skip = argv[i + 1]
            i += 1
        elif a in PKGS:
            pkgs.add(PKGS[a])
        i += 1
    return race, run, skip, pkgs


def top(pattern):
    # go test applies the first '/'-separated element to top-level names; these patterns have no '/'.
    assert pattern is None or "/" not in pattern, pattern
    return pattern


def cover(wf, tests):
    """test (pkg, name) -> list of (job, step, race)."""
    m = {t: [] for t in tests}
    for job, step, argv in go_test_lines(wf):
        race, run, skip, pkgs = selector(argv)
        run, skip = top(run), top(skip)
        for (pkg, name) in tests:
            if pkg not in pkgs:
                continue
            if run is not None and not re.search(run, name):
                continue
            if skip is not None and re.search(skip, name):
                continue
            m[(pkg, name)].append((job, step, race))
    return m


def main(argv):
    before_ref, after_ref, movers = argv[1], argv[2], set(argv[3:])
    tests = [(p, n) for path, p in PKGS.items() for n in list_tests(path)]
    print(f"tests listed: {len(tests)} ({sum(p == 'decoder' for p, _ in tests)} decoder, "
          f"{sum(p == 'sampling' for p, _ in tests)} sampling)")
    wb, wa = workflow(before_ref), workflow(after_ref)
    cb, ca = cover(wb, tests), cover(wa, tests)
    ok = True

    def race_steps(c, t):
        return [s for s in c[t] if s[2]]

    for side, c in (("before", cb), ("after", ca)):
        bad = [t for t in tests if len(race_steps(c, t)) != 1]
        n = sum(1 for t in tests if race_steps(c, t))
        print(f"G1 {side}: {n} of {len(tests)} tests selected by a -race step; {len(bad)} not by exactly one")
        for t in bad[:20]:
            print(f"   {t}: {race_steps(c, t)}")
        ok &= not bad
    sel_b = {t for t in tests if cb[t]}
    sel_a = {t for t in tests if ca[t]}
    print(f"G1 selected before {len(sel_b)}, after {len(sel_a)}, equal {sel_b == sel_a}")
    ok &= sel_b == sel_a
    moved = sorted(t for t in tests if race_steps(cb, t) != race_steps(ca, t))
    print(f"G1 tests whose -race step changed: {len(moved)}")
    for t in moved:
        print(f"   {t[0]}.{t[1]}: {[s[:2] for s in race_steps(cb, t)]} -> {[s[:2] for s in race_steps(ca, t)]}")
    ok &= {n for _, n in moved} == movers
    norace_b = sorted(t for t in tests if any(not s[2] for s in cb[t]))
    norace_a = sorted(t for t in tests if any(not s[2] for s in ca[t]))
    print(f"G1 no-race step selects {len(norace_b)} before, {len(norace_a)} after, equal {norace_b == norace_a}")
    ok &= norace_b == norace_a

    lost = [t for t in tests if race_steps(cb, t) and not race_steps(ca, t)]
    print(f"G2 tests that lost -race: {len(lost)}")
    ok &= not lost

    jb = {(k, v.get("name", k)) for k, v in wb["jobs"].items()}
    ja = {(k, v.get("name", k)) for k, v in wa["jobs"].items()}
    print(f"G3 jobs before {len(jb)}, after {len(ja)}, identical {jb == ja}")
    ok &= jb == ja

    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
