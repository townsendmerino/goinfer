#!/usr/bin/env python3
"""Gate AX1-T: compare `go test -v` logs of the moved packages, aikit at the freeze tag against goinfer, by test name.

    ax1_compare.py <baseline.log> <goinfer.log>

Pass: every test that reads --- PASS in the baseline reads --- PASS in goinfer, and no test reads --- FAIL in goinfer.
Prints each skip with its reason, on both sides, so a skip is never read as a pass.
"""
import re
import sys

RESULT = re.compile(r"^\s*--- (PASS|FAIL|SKIP): (\S+)")


def parse(path):
    """Returns ({test: result}, {test: first file:line message logged by it})."""
    out, reasons, cur = {}, {}, None
    for line in open(path, errors="replace"):
        m = RESULT.match(line)
        run = re.match(r"^=== (?:RUN|CONT|PAUSE)\s+(\S+)", line)
        if m:
            out[m.group(2)] = m.group(1)
            cur = m.group(2)
        elif run:
            cur = run.group(1)
        elif cur and re.match(r"^\s+\S+\.go:\d+: ", line) and cur not in reasons:
            reasons[cur] = line.strip()[:200]
    return out, reasons


def main():
    base, base_why = parse(sys.argv[1])
    new, new_why = parse(sys.argv[2])
    bad = 0
    for name, r in sorted(base.items()):
        if r == "PASS" and new.get(name) != "PASS":
            print(f"LOST PASS: {name} (goinfer: {new.get(name, 'absent')})")
            bad += 1
    for name, r in sorted(new.items()):
        if r == "FAIL":
            print(f"FAIL in goinfer: {name}")
            bad += 1
    gained = sorted(n for n, r in new.items() if r == "PASS" and base.get(n) != "PASS")
    for n in gained:
        print(f"newly passing: {n} (baseline: {base.get(n, 'absent')})")
    for side, res, why in (("baseline", base, base_why), ("goinfer", new, new_why)):
        for n in sorted(k for k, r in res.items() if r == "SKIP"):
            print(f"skip [{side}] {n}: {why.get(n, '')}")
    npass = sum(1 for r in new.values() if r == "PASS")
    print(f"baseline PASS {sum(1 for r in base.values() if r == 'PASS')}, goinfer PASS {npass}, "
          f"regressions {bad}: {'PASS' if bad == 0 else 'FAIL'}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
