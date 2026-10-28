#!/usr/bin/env python3
"""summary_conditions.py: PW2b's gate B1 (docs/tasks/task-subsystems-2026-10.md §12.4).

Loads .github/workflows/ci.yml at BASE_REF and from the working tree, and passes iff they differ only in an `id` added
to a step and an `if: always()` that became `always() && steps.<id>.outcome != 'skipped'`, and every `always()` summary
that reads a `go test -json` file names the step that writes that file.

Usage (from the repository root): python3 docs/measurements/ci-tail-2026-10-27/summary_conditions.py BASE_REF
"""

import re
import subprocess
import sys

import yaml

SUMMARY = re.compile(r"ci_test_summary\.py (\S+\.json)")
CONDITION = re.compile(r"always\(\) && steps\.(\w+)\.outcome != 'skipped'")


def main(base):
    a = yaml.safe_load(subprocess.run(["git", "show", f"{base}:.github/workflows/ci.yml"],
                                      check=True, capture_output=True, text=True).stdout)
    b = yaml.safe_load(open(".github/workflows/ci.yml"))
    ok, changes = True, []
    if a.keys() != b.keys() or a["jobs"].keys() != b["jobs"].keys():
        print("the top level or the job list differs")
        return 1
    for k in a:
        if k != "jobs" and a[k] != b[k]:
            ok = False
            print("top-level key differs:", k)
    for jn in a["jobs"]:
        ja, jb = a["jobs"][jn], b["jobs"][jn]
        if {k: v for k, v in ja.items() if k != "steps"} != {k: v for k, v in jb.items() if k != "steps"}:
            ok = False
            print("job keys differ:", jn)
        sa, sb = ja.get("steps", []), jb.get("steps", [])
        if len(sa) != len(sb):
            ok = False
            print("step count differs:", jn)
            continue
        for x, y in zip(sa, sb):
            diff = {k for k in set(x) | set(y) if x.get(k) != y.get(k)}
            if not diff:
                continue
            if diff == {"id"} and "id" not in x:
                changes.append((jn, y["name"], "id: " + y["id"]))
            elif diff == {"if"} and x.get("if") == "always()" and CONDITION.fullmatch(y["if"]):
                changes.append((jn, y["name"], "if: " + y["if"]))
            else:
                ok = False
                print("unexpected difference:", jn, y.get("name"), sorted(diff))
        writers = {s["id"]: s for s in sb if s.get("id")}
        for s in sb:
            m = SUMMARY.search(s.get("run", ""))
            if not m or "always()" not in str(s.get("if", "")):
                continue
            c = CONDITION.fullmatch(str(s["if"]))
            if not c or c.group(1) not in writers or ("> " + m.group(1)) not in writers[c.group(1)].get("run", ""):
                ok = False
                print("summary not tied to the step that writes its file:", jn, s["name"])
    for c in changes:
        print(*c, sep=" | ")
    print(f"{len(changes)} keys changed")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
