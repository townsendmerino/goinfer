#!/usr/bin/env python3
"""gate_ledger.py — the record of gate results a human has confirmed (B14).

WHY THIS EXISTS. `parity_sweep.sh` classified every gate as pass / fail / cannot-evaluate. A gate
reporting FAIL on its FIRST EXECUTION is asserting a delta it has no second point to compute: there
is no prior result to differ from. Attributing that failure to whatever enabled the gate is the one
hypothesis ruled out by construction — the change that made a failure visible is the change that
provably did not cause it.

So there is a fourth outcome, `first-run`, and it needs a record of which gates have a confirmed
prior result. That record is this ledger.

THE LEDGER IS THE "SOMEONE DECIDED IT IS CORRECT" RECORD. A gate enters it when a person promotes an
observed value to a baseline — never by the sweep observing itself. Auto-promotion would turn "never
checked" into "expected" in one silent step, which is how a wrong golden gets pinned for a year.

FIVE REQUIRED FIELDS. An entry missing any of them is a note, not a confirmation:

    gate            what the entry is about
    value           the result a person declared correct — the baseline itself
    promoted_by     a confirmation is an ACT BY A PERSON; no author, nobody behind it
    date            absolute
    commit          the state of the tree they were looking at when they said "this is correct"

THE SOURCE KEY. `source_sha256` is the hash of the gate's own test function body — NOT of its name
and NOT of its file. That distinction is the whole point:

  * renaming a gate drops its entry to stale, so the gate reverts to `first-run`. SAFE and loud.
  * a gate KEEPING its name while its assertion changes is NOT safe: the ledger would compare a
    confirmed value against different semantics and report pass — a green built on a confirmation of
    something else, with nothing in the name, the count, or the result to reveal it.

Hashing the function body means editing the assertion invalidates the confirmation, while
reformatting elsewhere in the file does not. Same staleness the citation lint's content keys catch,
same remedy.
"""

import argparse
import datetime
import hashlib
import json
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
LEDGER = ROOT / "testdata/gate_ledger.json"
REQUIRED = ("gate", "value", "promoted_by", "date", "commit")


def load():
    if not LEDGER.exists():
        return {"entries": []}
    return json.loads(LEDGER.read_text())


def save(d):
    LEDGER.write_text(json.dumps(d, indent=2, sort_keys=True) + "\n")


def func_source(name):
    """The gate's own function body, as bytes, or None if it cannot be found.

    Scans for `func <name>(` and takes lines through the closing brace at column 0. Deliberately
    dumb: a parser that can be wrong in subtle ways is worse here than one that fails loudly, and a
    None result is handled as "cannot key this gate" rather than silently as "unchanged".
    """
    pat = re.compile(r"^func " + re.escape(name) + r"\(")
    for f in sorted(ROOT.rglob("*_test.go")):
        if "/testdata/" in str(f):
            continue
        lines = f.read_text(errors="replace").splitlines(keepends=True)
        for i, ln in enumerate(lines):
            if pat.match(ln):
                out = [ln]
                for nxt in lines[i + 1:]:
                    out.append(nxt)
                    if nxt.startswith("}"):
                        return "".join(out)
                return "".join(out)
    return None


def source_key(name):
    src = func_source(name)
    if src is None:
        return None
    return hashlib.sha256(src.encode()).hexdigest()[:16]


def head():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def find(d, gate):
    for e in d["entries"]:
        if e.get("gate") == gate:
            return e
    return None


def cmd_promote(a):
    d = load()
    if find(d, a.gate) and not a.force:
        print(f"gate_ledger: {a.gate} already confirmed — pass --force to re-promote", file=sys.stderr)
        return 1
    key = source_key(a.gate)
    if key is None:
        print(f"gate_ledger: cannot find `func {a.gate}(` in any _test.go — refusing to record a "
              f"confirmation that cannot be keyed to source", file=sys.stderr)
        return 1
    d["entries"] = [e for e in d["entries"] if e.get("gate") != a.gate]
    d["entries"].append({
        "gate": a.gate, "value": a.value, "promoted_by": a.by,
        "date": a.date or datetime.date.today().isoformat(),
        "commit": a.commit or head(), "source_sha256": key,
        "note": a.note or "",
    })
    d["entries"].sort(key=lambda e: e["gate"])
    save(d)
    print(f"gate_ledger: confirmed {a.gate} = {a.value} (by {a.by}, source key {key})")
    return 0


def cmd_classify(a):
    """Print exactly one word for the sweep to consume."""
    if not LEDGER.exists():
        # NO LEDGER AT ALL means "we have no idea", NOT "nothing has ever run". If an absent ledger
        # produced FIRST-RUN, then merely deleting this file -- or shipping the feature before
        # anyone seeds it -- would make EVERY failing gate non-blocking. That is a safety regression
        # disguised as a new outcome. The mechanism stays INERT until the ledger is deliberately
        # created, and inert means the old behaviour: a failure blocks.
        print("CONFIRMED")
        return 0
    d = load()
    e = find(d, a.gate)
    key = source_key(a.gate)
    if e is None and key is None:
        # NO ENTRY AND NO SOURCE. Not a first-run -- a first-run is a gate that RAN and produced a
        # result we have no baseline for. This is a name we cannot even locate: a typo, a deleted
        # test, or a gate in a file this scan does not reach. Granting it first-run amnesty would
        # hand a free pass to exactly the cases nobody can inspect, so it is UNKNOWN and the caller
        # should treat it as blocking. Caught by the end-to-end mutation check, not by design.
        print("UNKNOWN-GATE")
        return 0
    if e is None:
        print("FIRST-RUN")
        return 0
    if key is not None and key != e.get("source_sha256"):
        print("SOURCE-CHANGED")
        return 0
    print("CONFIRMED")
    return 0


def cmd_reconcile(a):
    """The three checks, reported together. Never exits non-zero: this informs, it does not block."""
    d = load()
    gates = [g for g in (a.gates or "").split(",") if g]
    known = {e["gate"] for e in d["entries"]}
    print(f"  gate ledger: {len(d['entries'])} confirmed entr(ies) — testdata/gate_ledger.json")

    incomplete = [e["gate"] for e in d["entries"] if any(not e.get(f) for f in REQUIRED)]
    if incomplete:
        print(f"    INCOMPLETE ({len(incomplete)}): missing a required field — a note, not a "
              f"confirmation: {', '.join(sorted(incomplete))}")

    if gates:
        firstrun = sorted(set(gates) - known)
        if firstrun:
            print(f"    FIRST-RUN ({len(firstrun)}): no confirmed prior result, so a failure here "
                  f"has no delta to compute — reported, never a blocker")
            for g in firstrun:
                print(f"      {g}")

    stale = sorted(known - set(gates)) if gates else []
    if stale:
        print(f"    STALE ({len(stale)}): ledger entry with no matching gate — reported and IGNORED. "
              f"A removed or renamed gate is ordinary; a ledger that blocks on its own leftovers "
              f"trains people to delete entries")
        for g in stale:
            print(f"      {g}")

    changed = []
    for e in d["entries"]:
        k = source_key(e["gate"])
        if k is not None and k != e.get("source_sha256"):
            changed.append((e["gate"], e.get("commit", "?")))
    if changed:
        print(f"    CONFIRMED BEFORE THE GATE LAST CHANGED ({len(changed)}) — WARNING, does not block.")
        print(f"      The assertion moved since the confirming commit, so the recorded value may be "
              f"a confirmation of different semantics.")
        for g, c in changed:
            print(f"      {g}  (confirmed at {c})")
    return 0


def cmd_seed(a):
    """Bulk-seed from a sweep log's PASSing gates, marked as such.

    BULK-SEEDED IS NOT THE SAME AS CONFIRMED, and the ledger says so per entry. Without a seed every
    gate is first-run on day one and nothing blocks — but pretending a bulk import is 48 individual
    human judgements would be exactly the false-confirmation this ledger exists to prevent. So it is
    recorded honestly and can be upgraded one gate at a time by `promote`.
    """
    log = pathlib.Path(a.log).read_text(errors="replace")
    passed = sorted(set(re.findall(r"^--- PASS: (\S+) \(", log, re.M)))
    gates = [g for g in (a.gates or "").split(",") if g]
    if gates:
        passed = [g for g in passed if g in gates]
    d = load()
    known = {e["gate"] for e in d["entries"]}
    added = 0
    for g in passed:
        if g in known:
            continue
        key = source_key(g)
        if key is None:
            continue
        d["entries"].append({
            "gate": g, "value": "PASS", "promoted_by": a.by,
            "date": a.date or datetime.date.today().isoformat(),
            "commit": a.commit or head(), "source_sha256": key,
            "note": "BULK-SEEDED from a sweep log, not an individual judgement — upgrade with "
                    "`gate_ledger.py promote` when a person actually checks this gate's value",
        })
        added += 1
    d["entries"].sort(key=lambda e: e["gate"])
    save(d)
    print(f"gate_ledger: seeded {added} gate(s) as PASS (bulk, by {a.by}); ledger now "
          f"{len(d['entries'])} entr(ies)")
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("promote", help="record a human confirmation of a gate's value")
    pr.add_argument("--gate", required=True)
    pr.add_argument("--value", required=True)
    pr.add_argument("--by", required=True)
    pr.add_argument("--date")
    pr.add_argument("--commit")
    pr.add_argument("--note")
    pr.add_argument("--force", action="store_true")
    pr.set_defaults(fn=cmd_promote)

    cl = sub.add_parser("classify", help="CONFIRMED | FIRST-RUN | SOURCE-CHANGED for one gate")
    cl.add_argument("--gate", required=True)
    cl.set_defaults(fn=cmd_classify)

    rc = sub.add_parser("reconcile", help="the three checks, reported next to the counts")
    rc.add_argument("--gates", help="comma-separated list of gates the sweep checked")
    rc.set_defaults(fn=cmd_reconcile)

    sd = sub.add_parser("seed", help="bulk-seed PASSing gates from a sweep log")
    sd.add_argument("--log", required=True)
    sd.add_argument("--by", required=True)
    sd.add_argument("--gates")
    sd.add_argument("--date")
    sd.add_argument("--commit")
    sd.set_defaults(fn=cmd_seed)

    a = p.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
