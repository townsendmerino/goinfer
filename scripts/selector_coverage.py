#!/usr/bin/env python3
"""selector_coverage.py — tests that EXIST versus tests any selector actually RUNS.

Three coverage gaps this campaign found were plumbing, not authorship. Every one of them was a test
that existed, passed when invoked, and was simply never selected:

  - the three int8int8 goldens      — skipped on GOINFER_HEAVY_TESTS being unset
  - gpt_oss's int8 golden           — behind //go:build realckpt, AND a missing checkpoint
  - eleven GGUF quant-format gates  — outside the goldens selector's regexp

Each was found by someone asking a different question and noticing in passing. This asks the question
directly: enumerate what exists, enumerate what the selectors reach, and print the difference.

DESIGN, from what the other censuses learned:

  - DERIVE BOTH SIDES. The selectors are read out of refresh_parity_hashes.sh's GOLDEN_RE and
    parity_sweep.sh's GATES block, not restated here. A second copy of either would drift, and
    drifting copies are the defect this repo keeps finding.
  - SEPARATE THE REASONS. never-selected, build-tag-excluded, env-gated and asset-blocked have
    different remedies and different costs — collapsing them into one "uncovered" number is what
    made the int8int8 rows look as expensive as authoring new fixtures when they cost one env var.
  - ERR TOWARD FLAGGING. The env detection matches any GOINFER_* read in the file, so it flags
    TestInt4_forwardParity for GOINFER_INT4_GOLDEN_UPDATE — which gates REGENERATION, not the test.
    That false positive is left in deliberately: a census that under-reports is the failure mode
    that produced all three gaps above, and a reader can dismiss a flagged line in seconds where a
    missing one costs weeks.
  - PRINT THE DIFFERENCE, NOT A VERDICT. A test can be selected and still vacuous, so a green here
    means "nothing became unreachable since a person last looked", not "coverage is adequate".
"""

import collections
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SCAN_DIRS = ["decoder"]  # cuda/ and metal/ are device-gated suites with their own runners

TESTFN = re.compile(r"^func (Test[A-Za-z0-9_]+)\(", re.M)
BUILDTAG = re.compile(r"^//go:build (.+)$", re.M)
HEAVY = re.compile(r'Getenv\("(GOINFER_HEAVY_TESTS|GOINFER_[A-Z0-9_]*)"\)')
ASSETSKIP = re.compile(r"t\.Skipf?\([^)]*\b(no |not found|missing|regenerate|checkpoint|gguf)", re.I)


def golden_re():
    sh = (ROOT / "scripts" / "refresh_parity_hashes.sh").read_text()
    m = re.search(r"GOLDEN_RE='([^']+)'", sh)
    if not m:
        sys.stderr.write("selector_coverage: GOLDEN_RE not found — cannot derive the goldens selector\n")
        sys.exit(1)
    return re.compile(m.group(1))


def sweep_tests():
    sh = (ROOT / "scripts" / "parity_sweep.sh").read_text()
    m = re.search(r"^GATES=\((.*?)^\)", sh, re.S | re.M)
    if not m:
        sys.stderr.write("selector_coverage: GATES block not found — cannot derive the sweep selector\n")
        sys.exit(1)
    return {t for _, t in re.findall(r'"\s*([^|"]+?)\s*\|\s*([A-Za-z0-9_]+)\s*"', m.group(1))}


def main() -> int:
    sel_g, sel_s = golden_re(), sweep_tests()
    rows = []
    for d in SCAN_DIRS:
        for f in sorted((ROOT / d).glob("*_test.go")):
            txt = f.read_text()
            tag = BUILDTAG.search(txt)
            tag = tag.group(1).strip() if tag else ""
            envs = sorted(set(HEAVY.findall(txt)))
            asset = bool(ASSETSKIP.search(txt))
            for name in TESTFN.findall(txt):
                rows.append({
                    "test": name, "file": f"{d}/{f.name}", "tag": tag,
                    "envs": envs, "asset": asset,
                    "sel": bool(sel_g.search(name)) or name in sel_s,
                })

    if not rows:
        sys.stderr.write("selector_coverage: found ZERO tests — the scanner is broken, not the tree empty\n")
        return 1

    selected = [r for r in rows if r["sel"]]
    unselected = [r for r in rows if not r["sel"]]

    print(f"  SELECTOR COVERAGE — {len(rows)} tests in {'/'.join(SCAN_DIRS)}/")
    print(f"    reached by a selector : {len(selected)}")
    print(f"    reached by NONE       : {len(unselected)}")
    print()

    # Of the SELECTED ones, which cannot actually run as invoked — the int8int8 and gpt_oss shapes.
    blocked = collections.defaultdict(list)
    for r in selected:
        if r["tag"]:
            blocked[f"build tag: {r['tag']}"].append(r["test"])
        elif r["envs"]:
            blocked["env-gated: " + ",".join(r["envs"])].append(r["test"])
        elif r["asset"]:
            blocked["asset-gated (skips without a checkpoint)"].append(r["test"])
    if blocked:
        print("  SELECTED but conditionally unreachable — these are the plumbing gaps:")
        for why, ts in sorted(blocked.items()):
            print(f"    {why}  ({len(ts)})")
            for t in sorted(ts)[:6]:
                print(f"      {t}")
            if len(ts) > 6:
                print(f"      … and {len(ts) - 6} more")
        print()

    print("  A green here means nothing became UNREACHABLE since a person last looked.")
    print("  It does NOT mean coverage is adequate — a test can be selected and still vacuous.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
