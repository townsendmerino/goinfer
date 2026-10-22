#!/usr/bin/env python3
"""Grade KV4's G2 decode-at-depth cells and G4's never-swap reading (kv4-cuda-2026-10-21.md §2) from bench_peer.py results run with
BENCH_ENGINES=goinfer,goinfer_old BENCH_ABBA=1 and procmem_sampler.py's log.

G2, per (model, depth): each block's ratio is goinfer's mean decode tok/s over goinfer_old's in that block; the cell's ratio is the median
over its blocks. PASS >= 0.98, AMBIGUOUS 0.97-0.98, FAIL < 0.97. A cell whose engine ran off the GPU, or whose start load was over the void
rule's 8.0, is VOID.

G4: each server's peak VmHWM and peak VmSwap, new against old in pairs of the same cell group. PASS when every new peak VmSwap is no more
than its old one's and every VmHWM is within 64 MB of its old one's.

  score_speed.py <procmem.jsonl> <results.json> [<results.json> ...]
"""
import json, statistics, sys

procs = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
cells = []
for path in sys.argv[2:]:
    tag = path.rsplit("/", 1)[-1].removesuffix(".json")
    cells += [dict(c, file=tag) for c in json.load(open(path)) if isinstance(c, dict) and c.get("engine") in ("goinfer", "goinfer_old")]

# One group per results file, model and depth: each invocation also runs its models' depth-128 cell, at that invocation's context.
groups = {}
for c in cells:
    groups.setdefault((c["model"], c["depth"], c["file"]), []).append(c)

verdicts = []
print(f"{'model':>10} {'depth':>6} {'file':>14}  {'blocks new/old':<28} {'median':>7}  verdict")
for (model, depth, tag), cs in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
    by_block = {}
    void = []
    for c in cs:
        if not c.get("runs") or c.get("error"):
            void.append(f"{c['engine']} block {c.get('block')}: {c.get('error') or 'no runs'}")
            continue
        path = c.get("decode_path") or (c.get("counts") or {}).get("decode_path")
        if not str(path or "").startswith("cuda"):
            void.append(f"{c['engine']} block {c.get('block')}: decode path {path}")
        load1 = (c.get("machine") or {}).get("loadavg", [0])[0]
        if load1 > 8.0:
            void.append(f"{c['engine']} block {c.get('block')}: load1 {load1:.2f}")
        by_block.setdefault(c.get("block"), {})[c["engine"]] = c
    ratios = []
    for b, pair in sorted(by_block.items(), key=lambda kv: (kv[0] is None, kv[0])):
        if "goinfer" in pair and "goinfer_old" in pair:
            ratios.append(statistics.mean(pair["goinfer"]["runs"]) / statistics.mean(pair["goinfer_old"]["runs"]))
    med = statistics.median(ratios) if ratios else float("nan")
    v = "VOID" if void or not ratios else ("PASS" if med >= 0.98 else "AMBIGUOUS" if med >= 0.97 else "FAIL")
    verdicts.append(v)
    print(f"{model:>10} {depth:>6} {tag:>14}  {' '.join(f'{x:.3f}' for x in ratios):<28} {med:7.3f}  {v}" + (f"  ({'; '.join(void)})" if void else ""))


print()
print("G2 verdict:", "FAIL" if "FAIL" in verdicts else "VOID" if "VOID" in verdicts else "AMBIGUOUS" if "AMBIGUOUS" in verdicts else "PASS")

# G4 from the sampler: every server process of each binary, peak VmHWM and VmSwap
print()
for name in ("serve-cuda-old", "serve-cuda-new"):
    ps = [p for p in procs if p["name"] == name]
    if not ps:
        print(f"{name}: no samples")
        continue
    print(f"{name}: {len(ps)} servers; VmHWM median {statistics.median(p['hwm_kb'] for p in ps) / 1024:.0f} MB, max {max(p['hwm_kb'] for p in ps) / 1024:.0f} MB; "
          f"peak VmSwap max {max(p['swap_peak_kb'] for p in ps) / 1024:.1f} MB; system swap {min(p['sys_swap_mb_first'] for p in ps)}-{max(p['sys_swap_mb_last'] for p in ps)} MB (context)")
# Servers run in the plan's order, A B B A per cell group, so the k-th new server and the k-th old one served the same cell group.
old = sorted((p for p in procs if p["name"] == "serve-cuda-old"), key=lambda p: p["first"])
new = sorted((p for p in procs if p["name"] == "serve-cuda-new"), key=lambda p: p["first"])
if old and new and len(old) == len(new):
    dh = [n["hwm_kb"] - o["hwm_kb"] for o, n in zip(old, new)]
    ds = [n["swap_peak_kb"] - o["swap_peak_kb"] for o, n in zip(old, new)]
    ok = max(ds) <= 0 and max(abs(x) for x in dh) <= 64 * 1024
    print(f"G4: {len(dh)} pairs; new - old VmHWM {min(dh) / 1024:+.1f} to {max(dh) / 1024:+.1f} MB, peak VmSwap {min(ds) / 1024:+.1f} to {max(ds) / 1024:+.1f} MB -> {'PASS' if ok else 'FAIL'}")
else:
    print(f"G4: VOID, {len(old)} old and {len(new)} new servers sampled")
