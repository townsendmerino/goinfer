"""List the LOOSE numeric tolerances that gate a test result, and whether each records its basis.

A tight bar (>= 0.999) says "these should agree" and its basis is self-evident. A loose one encodes
a judgement about how much error is acceptable, and that judgement is only checkable if the
population it was calibrated on is written down beside it. When it is not, the bar outlives its
population silently -- which is exactly how the repo's first int4 T3 came to be measured against a
bar whose own comment said "int8 W8A8 vs bf16" (2026-08-27, G25).

    python3 scripts/audit_gate_bars.py [--threshold 0.999]

Reports, never fails: this is an inventory, not a gate. Making it a gate would mean deciding that
every bar must be documented, which is a policy question this script is not the place to settle.
"""
import argparse, os, re

PAT = re.compile(r'\b(?:cos|cosine|c|rel|err|ratio|acc|drift|sim)\w*\s*[<>]=?\s*(0\.\d{2,})')


def scan(threshold):
    rows = []
    for root, _, files in os.walk("."):
        if ".git" in root:
            continue
        for f in files:
            if not f.endswith("_test.go"):
                continue
            p = os.path.join(root, f)
            lines = open(p, errors="ignore").read().split("\n")
            for i, l in enumerate(lines):
                if l.strip().startswith("//"):
                    continue
                m = PAT.search(l)
                if not m or float(m.group(1)) >= threshold:
                    continue
                # Any comment within 6 lines above, plus a trailing one. The window matters: a
                # 2-line walk of contiguous comments misreports justified bars as bare, because the
                # comment usually sits above the enclosing `if cos := ...`, not above the compare.
                win = " ".join(x.strip() for x in lines[max(0, i - 6):i + 1]
                               if x.strip().startswith("//"))
                if "//" in l:
                    win += " " + l.split("//", 1)[1]
                rows.append((p.lstrip("./"), i + 1, float(m.group(1)), len(win.strip()) > 25))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.999,
                    help="bars at or above this are treated as tight and skipped")
    args = ap.parse_args()
    rows = scan(args.threshold)
    bare = [r for r in rows if not r[3]]
    print(f"loose tolerances (< {args.threshold}): {len(rows)} | "
          f"basis recorded: {len(rows) - len(bare)} | bare: {len(bare)}\n")
    for p, ln, v, _ in sorted(bare, key=lambda r: (r[2], r[0])):
        print(f"  {v:<7} {p}:{ln}")


if __name__ == "__main__":
    main()
