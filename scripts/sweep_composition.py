#!/usr/bin/env python3
"""sweep_composition.py — print parity_sweep.sh's coverage COMPOSITION along its axes.

The policy rule this implements: a gate whose value depends on an axis must print its composition
along that axis. `parity_sweep.sh` is the release gate and its axes are family x quant x loader, and
until now it reported pass/fail per gate with nothing saying what the set spanned. That is the shape
that let the forward goldens report "19 passed" through nine deps_hash refreshes while every one of
the 19 was f32 — an accurate count that could not distinguish "the axis is covered" from "the axis
collapsed to one value".

DERIVED, not declared. Quant comes from grepping each gate's own test source for `Quant: "..."`, and
loader from the test name. Hand-maintained axis metadata beside the gate list would be a second copy
to drift, which is the defect this repo keeps finding.

A gate whose test source cannot be located is reported as UNKNOWN rather than defaulted to f32 —
defaulting would inflate the f32 count with gates nobody checked, which is the opposite of the point.
"""

import collections
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SWEEP = ROOT / "scripts" / "parity_sweep.sh"

# Only the GATES=( ... ) array. EMIT_GATES is a different list with the columns the other way round,
# and reading both is how an ad-hoc version of this produced six rows with family and test swapped.
GATES_BLOCK = re.compile(r"^GATES=\((.*?)^\)", re.S | re.M)
ENTRY = re.compile(r'"\s*([^|"]+?)\s*\|\s*([A-Za-z0-9_]+)\s*"')


def test_source(test: str):
    r = subprocess.run(
        ["grep", "-rln", f"func {test}(", "decoder/", "cuda/", "metal/", "tokenizer/", "internal/"],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    out = [x for x in r.stdout.strip().split("\n") if x]
    return out[0] if out else None


def main() -> int:
    m = GATES_BLOCK.search(SWEEP.read_text())
    if not m:
        sys.stderr.write("sweep_composition: could not find the GATES=( ) block — refusing to "
                         "report a composition derived from nothing.\n")
        return 1
    entries = ENTRY.findall(m.group(1))
    if not entries:
        sys.stderr.write("sweep_composition: GATES block parsed to ZERO gates — extractor broken.\n")
        return 1

    rows, unknown = [], 0
    for fam, test in entries:
        src = test_source(test)
        if src is None:
            quant, unknown = "UNKNOWN", unknown + 1
        else:
            qs = sorted(set(re.findall(r'Quant:\s*"([a-z0-9]+)"', pathlib.Path(ROOT / src).read_text())))
            quant = "/".join(qs) if qs else "f32"
        loader = "gguf" if "GGUF" in test or "gguf" in test.lower() else "safetensors"
        rows.append((fam, test, quant, loader))

    if "-v" in sys.argv:
        print(f"  {'family':<22}{'quant':<14}{'loader':<14}test")
        for fam, test, q, l in rows:
            print(f"  {fam:<22}{q:<14}{l:<14}{test}")
        print()

    q = collections.Counter(r[2] for r in rows)
    l = collections.Counter(r[3] for r in rows)
    f = len({r[0] for r in rows})
    print(f"  COMPOSITION of the release gate — {len(rows)} gates over {f} family labels")
    print(f"    quant :  " + "  ".join(f"{k}={v}" for k, v in sorted(q.items())))
    print(f"    loader:  " + "  ".join(f"{k}={v}" for k, v in sorted(l.items())))
    if unknown:
        print(f"    NOTE: {unknown} gate(s) UNKNOWN — test source not located, NOT counted as f32")
    if len(q) == 1:
        print(f"    WARNING: the quant axis has collapsed to a single value ({next(iter(q))}). "
              f"This gate no longer varies over the axis it is supposed to protect.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
