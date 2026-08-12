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


def golden_axes():
    """Quantizations AND loaders the forward goldens actually drive.

    The loader axis was the one nobody had checked — the cross-gate check compared quant only, so
    "both gates span the same axes" was an answer about one axis presented as an answer about the
    gate. Derived the same way as the quant side: the selector comes from refresh_parity_hashes.sh's
    own GOLDEN_RE, and loader from the test name.
    """
    """Quantizations the forward goldens actually drive, derived the same way.

    The goldens are selected by refresh_parity_hashes.sh's GOLDEN_RE, so the set is derived from that
    regexp rather than from a second list that could drift from it.
    """
    sh = ROOT / "scripts" / "refresh_parity_hashes.sh"
    if not sh.exists():
        return None
    m = re.search(r"GOLDEN_RE='([^']+)'", sh.read_text())
    if not m:
        return None
    sel = re.compile(m.group(1))
    quants, loaders = set(), set()
    for f in sorted(ROOT.glob("decoder/*_test.go")):
        txt = f.read_text()
        names = [n for n in re.findall(r"func (Test[A-Za-z0-9_]+)\(", txt) if sel.search(n)]
        if not names:
            continue
        if re.search(r"^//go:build", txt, re.M) and "realckpt" in txt.split("\n")[0]:
            continue  # behind a build tag the refresh does not pass — invisible to it
        qs = set(re.findall(r'Quant:\s*"([a-z0-9]+)"', txt))
        quants |= qs if qs else {"f32"}
        for n in names:
            loaders.add("gguf" if "GGUF" in n or "gguf" in n.lower() else "safetensors")
    return quants, loaders


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
    # CROSS-GATE. parity_sweep.sh and the goldens refresh protect overlapping properties over the
    # same axis, and until both printed their composition the difference between them was invisible:
    # the sweep covered int4 all along while the refresh was f32-only, and nobody saw the gap because
    # neither said what it spanned. Printing both and their difference makes a future divergence
    # visible rather than inferred.
    gold_axes = golden_axes()
    if gold_axes is not None:
        gold, gold_l = gold_axes
        # ATOMISE both sides before comparing. A gate whose test file drives two quantizations gets a
        # composite label like "int4/int8", and comparing that against the atomic labels the other
        # side produces reports a difference that is purely notational — a permanent false positive
        # in the check built to make real differences visible.
        def atoms(xs):
            out = set()
            for x in xs:
                out |= set(x.split("/"))
            return out

        sweep_q, gold_q = atoms(q), atoms(gold)
        sweep_l, gl = atoms(l), atoms(gold_l)
        print()
        print("  CROSS-GATE quant coverage (release gate vs the freeze-exception goldens):")
        print(f"    parity_sweep.sh   : {' '.join(sorted(sweep_q)) or '(none)'}")
        print(f"    forward goldens   : {' '.join(sorted(gold_q)) or '(none)'}")
        only_sweep = sweep_q - gold_q
        only_gold = gold_q - sweep_q
        if only_sweep:
            print(f"    ONLY in the sweep : {' '.join(sorted(only_sweep))}")
            print("      -> a core edit can pass the goldens refresh and still be unproven on these,")
            print("         because the refresh is the ONLY numeric proof a frozen-core edit gets.")
        if only_gold:
            print(f"    ONLY in the goldens: {' '.join(sorted(only_gold))}")
        if not only_sweep and not only_gold:
            print("    -> the two gates span the same quantizations.")
        print()
        print("  CROSS-GATE loader coverage:")
        print(f"    parity_sweep.sh   : {' '.join(sorted(sweep_l))}")
        print(f"    forward goldens   : {' '.join(sorted(gl))}")
        ls_only, lg_only = sweep_l - gl, gl - sweep_l
        if ls_only:
            print(f"    ONLY in the sweep : {' '.join(sorted(ls_only))}")
        if lg_only:
            print(f"    ONLY in the goldens: {' '.join(sorted(lg_only))}")
        if not ls_only and not lg_only:
            print("    -> the two gates span the same loaders.")

    if len(q) == 1:
        print(f"    WARNING: the quant axis has collapsed to a single value ({next(iter(q))}). "
              f"This gate no longer varies over the axis it is supposed to protect.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
