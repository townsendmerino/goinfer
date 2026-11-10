#!/usr/bin/env python3
"""#873: compare chat fit at A (589c6ebd9, the merge-base) and B (55128eca3) for every ~/models entry.
Per entry: exit codes, and the output with the live free-memory figures masked (they are read off the host at run time:
"fits N GB free", and the memory-derived context cap line). An entry whose fit was killed by the sweep's swap watchdog (MEMKILL)
on either head is NOT COMPARABLE here. Run: compare.py <run dir>."""
import os, re, sys, glob

R = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/goinfer-bench/873/run")
MASKS = [
    (re.compile(r"fits [0-9.]+ GB free"), "fits <N> GB free"),
    (re.compile(r"(context capped at )\d+( \(model allows \d+\) so that KV fits: weights )[0-9.]+( GB \+ KV )[0-9.]+( GB of budget )[0-9.]+( GB)"), r"\g<1><N>\2<N>\3<N>\4<N>\5"),
    (re.compile(r"(ctx=)\d+"), r"\1<N>"),  # a capped context shows in the header line
    (re.compile(r"(KV@)\d+ [0-9.]+ GB"), r"\1<N> <N> GB"),
]

# Host-memory advisories: whether they print, and the figures in them, follow the memory available when the fit ran. A banner that
# prints on one head only is advisory drift, not a plan change; the plan lines (dense, experts, KV@ctx, RESIDENT/stream) are compared.
ADVISORY = re.compile(r"^decoder: (fit is tight|context capped at) ")
AVAIL = re.compile(r"has [0-9.]+ GB of memory available \(budget [0-9.]+ GB = 70% of that\)( \(\d+% of budget\))?")

def norm(text):
    lines = [l for l in text.splitlines() if not ADVISORY.match(l)]
    text = "\n".join(lines)
    text = AVAIL.sub("has <N> GB of memory available (budget <N> GB = 70% of that)", text)
    for rx, rep in MASKS:
        text = rx.sub(rep, text)
    return text

rows = {"same": [], "same-raw": [], "differs": [], "rc-differs": [], "not-comparable": []}
for fa in sorted(glob.glob(f"{R}/out/A/*.rc")):
    n = os.path.basename(fa)[:-3]
    fb = f"{R}/out/B/{n}.rc"
    if not os.path.exists(fb):
        rows["not-comparable"].append((n, "no B result")); continue
    rca, rcb = open(fa).read().split()[0], open(fb).read().split()[0]
    ta, tb = open(f"{R}/out/A/{n}.txt", errors="replace").read(), open(f"{R}/out/B/{n}.txt", errors="replace").read()
    if "MEMKILL" in ta or "MEMKILL" in tb:
        side = "+".join(s for s, t in (("A", ta), ("B", tb)) if "MEMKILL" in t)
        rows["not-comparable"].append((n, f"MEMKILL on {side}")); continue
    if rca != rcb:
        rows["rc-differs"].append((n, f"rc {rca} vs {rcb}")); continue
    if ta == tb:
        rows["same-raw"].append((n, f"rc {rca}")); continue
    if norm(ta) == norm(tb):
        rows["same"].append((n, f"rc {rca}, identical once the host-memory advisories and figures are masked")); continue
    resid = [l for l in __import__("difflib").unified_diff(norm(ta).splitlines(), norm(tb).splitlines(), lineterm="", n=0) if l[:1] in "+-" and l[:3] not in ("+++", "---")]
    rows["differs"].append((n, f"rc {rca}, residual: " + " | ".join(r[:110] for r in resid[:4])))

total = sum(len(v) for v in rows.values())
print(f"{total} entries compared (A = 589c6ebd9, B = 55128eca3)")
for k in ("same-raw", "same", "rc-differs", "differs", "not-comparable"):
    print(f"{k}: {len(rows[k])}")
for k in ("rc-differs", "differs", "not-comparable"):
    for n, why in rows[k]:
        print(f"  [{k}] {n.replace('_home_francis_models_', '~/models/')}: {why}")
rcs = {}
for n, _ in rows["same-raw"] + rows["same"]:
    rcs[open(f"{R}/out/A/{n}.rc").read().split()[0]] = rcs.get(open(f"{R}/out/A/{n}.rc").read().split()[0], 0) + 1
print("exit codes among comparable entries (equal at both heads):", dict(sorted(rcs.items())))
