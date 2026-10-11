#!/usr/bin/env python3
"""T4b's day gate (docs/completed/task-giw-v16-transcode-2026-10.md, G-T2's last row, as amended 2026-10-10): the
single-thread CPU time of a transcode, new build over old build.

    t4b_cpu_ab.py --old <prequant> --new <prequant> --out-dir <scratch> --record <jsonl> name|source|quant|target ...

Each transcode runs with GOMAXPROCS=1 under time; the quantity is user + system seconds. Per cell: one discarded
warm-up per arm, then --pairs pairs in alternating order (old, new; new, old; ...). Exits 1 if a transcode fails or
the two arms write different bytes. It prints the paired ratios; the bands are in the task doc.
"""
import argparse
import hashlib
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import time

MAC = platform.system() == "Darwin"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(8 << 20):
            h.update(b)
    return h.hexdigest()


def run(binary, cell, out):
    for p in (out, out + ".verified"):
        if os.path.exists(p):
            os.remove(p)
    env = {k: v for k, v in os.environ.items() if not k.startswith("GOINFER_")}
    env["GOMAXPROCS"] = "1"
    cmd = ["/usr/bin/time", "-l" if MAC else "-v", binary, "-quant", cell["quant"], "-target", cell["target"], "-o", out, cell["source"]]
    t0 = time.monotonic()
    p = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=30 * 60)
    wall = time.monotonic() - t0
    text = p.stdout + p.stderr
    if MAC:
        m = re.search(r"([0-9.]+) real\s+([0-9.]+) user\s+([0-9.]+) sys", text)
        user, system = (float(m.group(2)), float(m.group(3))) if m else (None, None)
    else:
        mu, ms = re.search(r"User time \(seconds\): ([0-9.]+)", text), re.search(r"System time \(seconds\): ([0-9.]+)", text)
        user, system = (float(mu.group(1)), float(ms.group(1))) if mu and ms else (None, None)
    if p.returncode != 0 or user is None or not os.path.exists(out):
        sys.exit(f"FAILED {binary} on {cell['name']}: rc {p.returncode}: {text.strip()[-400:]}")
    ev = {"wall_s": round(wall, 3), "user_s": user, "sys_s": system, "cpu_s": round(user + system, 3), "sha256": sha256(out), "bytes": os.path.getsize(out)}
    os.remove(out)
    if os.path.exists(out + ".verified"):
        os.remove(out + ".verified")
    return ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--record", required=True)
    ap.add_argument("--pairs", type=int, default=5)
    ap.add_argument("cells", nargs="+")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    rec = open(a.record, "a")
    bins = {"old": a.old, "new": a.new}
    failed = False
    print(f"started {time.strftime('%Y-%m-%d %H:%M:%S %Z')} on {platform.node()} ({platform.system()}/{platform.machine()}), GOMAXPROCS=1, {a.pairs} pairs a cell", flush=True)
    for spec in a.cells:
        name, source, quant, target = spec.split("|")
        cell = {"name": name, "source": os.path.expanduser(source), "quant": quant, "target": target}
        out = os.path.join(a.out_dir, name + ".giw")
        shas, ratios, cpu = set(), [], {"old": [], "new": []}
        for arm in ("old", "new"):
            ev = run(bins[arm], cell, out)
            shas.add(ev["sha256"])
            rec.write(json.dumps({"cell": name, "phase": "warm-up", "arm": arm, **ev}) + "\n")
        for k in range(a.pairs):
            pair = {}
            for arm in (("old", "new") if k % 2 == 0 else ("new", "old")):
                ev = run(bins[arm], cell, out)
                shas.add(ev["sha256"])
                pair[arm] = ev["cpu_s"]
                cpu[arm].append(ev["cpu_s"])
                rec.write(json.dumps({"cell": name, "phase": f"pair {k + 1}", "arm": arm, **ev}) + "\n")
                rec.flush()
            ratios.append(pair["new"] / pair["old"])
            print(f"  {name} pair {k + 1}: old {pair['old']:.2f} s, new {pair['new']:.2f} s, new/old {ratios[-1]:.4f}", flush=True)
        identical = len(shas) == 1
        failed = failed or not identical
        line = {"cell": name, "phase": "summary", "identical": identical, "ratios": [round(r, 4) for r in ratios],
                "median": round(statistics.median(ratios), 4), "old_cpu_s_median": statistics.median(cpu["old"]), "new_cpu_s_median": statistics.median(cpu["new"])}
        rec.write(json.dumps(line) + "\n")
        rec.flush()
        print(f"{name}: new/old CPU median {line['median']:.4f} (range {min(ratios):.4f}-{max(ratios):.4f}), old {line['old_cpu_s_median']:.2f} s, new {line['new_cpu_s_median']:.2f} s, bytes {'IDENTICAL' if identical else 'DIFFERENT'}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
