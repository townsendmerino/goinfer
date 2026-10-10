#!/usr/bin/env python3
"""The night run for docs/tasks/task-giw-v16-transcode-2026-10.md: G-T1's real cells, G-T2, G-T2c and G-T3.

Two pre-built cmd/prequant binaries (old, new) transcode each cell's source. Per cell: one discarded warm-up per arm,
then pairs in ABAB order with the copy floor's two copies made straight after the old arm, whose output they read:

    pair 1: old, copies, new        pair 2: new, old, copies        pair 3: as pair 1 ...

Every event is one JSON line in --record. Nothing is graded here: the bands are in the task doc, and the summary this
prints is the paired ratios for the morning's grading. It exits 1 if a transcode failed or a cell's two arms wrote
different bytes.

A cell is  name|source|quant|target|pairs  (target as cmd/prequant's -target takes it).
"""
import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import time

MAC = platform.system() == "Darwin"
DD_BS = "bs=8m" if MAC else "bs=8M"
T0 = time.time()


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')} +{time.time() - T0:5.0f}s] {msg}", flush=True)


def swap_used_mb():
    if not MAC:
        return None
    out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
    m = re.search(r"used = ([0-9.]+)M", out)
    return float(m.group(1)) if m else None


def mem_available_bytes():
    """What the system calls available before a transcode: Linux's MemAvailable; on the Mac, vm_stat's free, inactive
    and speculative pages. None when it cannot be read."""
    try:
        if MAC:
            out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
            page = int(re.search(r"page size of (\d+) bytes", out).group(1))
            return page * sum(int(re.search(rf"Pages {k}:\s+(\d+)", out).group(1)) for k in ("free", "inactive", "speculative"))
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except (OSError, AttributeError, ValueError):
        pass
    return None


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(8 << 20)
            if not b:
                return h.hexdigest()
            h.update(b)


def source_files(src):
    """The files a perfect transcode reads: the GGUF, or a directory's weight shards."""
    if os.path.isdir(src):
        return sorted(os.path.join(src, f) for f in os.listdir(src) if f.endswith(".safetensors"))
    return [src]


def remove(path):
    for p in (path, path + ".verified"):
        try:
            os.remove(p)
        except FileNotFoundError:
            pass


def clean_env():
    return {k: v for k, v in os.environ.items() if not k.startswith("GOINFER_")}


def transcode(binary, cell, out):
    """One timed transcode. Returns the event's fields; the output is left at out."""
    remove(out)
    cmd = ["/usr/bin/time", "-l" if MAC else "-v", binary, "-quant", cell["quant"], "-target", cell["target"], "-o", out, cell["source"]]
    swap0 = swap_used_mb()
    avail = mem_available_bytes()
    t0 = time.monotonic()
    p = subprocess.run(cmd, capture_output=True, text=True, env=clean_env(), timeout=45 * 60)
    wall = time.monotonic() - t0
    text = p.stdout + p.stderr
    ev = {"wall_s": round(wall, 3), "rc": p.returncode, "mem_available_bytes_before": avail,
          "swap_used_mb_before": swap0, "swap_used_mb_after": swap_used_mb()}
    m = re.search(r"wrote .*?: (\d+) MB \((\w+)\)(?: in ([0-9.]+)s: stream ([0-9.]+)s, self-check ([0-9.]+)s)?", text)
    if m:
        ev["wrote_line"] = m.group(0)
        if m.group(3):
            ev.update(total_s=float(m.group(3)), stream_s=float(m.group(4)), selfcheck_s=float(m.group(5)))
    if MAC:
        for key, pat in (("peak_footprint_bytes", r"(\d+)\s+peak memory footprint"), ("max_rss_bytes", r"(\d+)\s+maximum resident set size")):
            m = re.search(pat, text)
            if m:
                ev[key] = int(m.group(1))
    else:
        m = re.search(r"Maximum resident set size \(kbytes\): (\d+)", text)
        if m:
            ev["max_rss_bytes"] = int(m.group(1)) * 1024
    if p.returncode != 0 or not os.path.exists(out):
        ev["error"] = text.strip().splitlines()[-12:]
        return ev
    ev["bytes"] = os.path.getsize(out)
    ev["verified_marker"] = os.path.exists(out + ".verified")
    return ev


def dd(src, dst):
    t0 = time.monotonic()
    p = subprocess.run(["dd", f"if={src}", f"of={dst}", DD_BS], capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"dd {src} -> {dst}: {p.stderr.strip()[-300:]}")
    return time.monotonic() - t0


def copies(cell, old_out, out_dir):
    """G-T2c's two copies, with dd and never cp (a clone on APFS and btrfs)."""
    tmp = os.path.join(out_dir, "copyfloor.tmp")
    files = source_files(cell["source"])
    src_bytes = sum(os.path.getsize(f) for f in files)
    ev = {"source_bytes": src_bytes, "source_files": len(files)}

    def free():
        return shutil.disk_usage(out_dir).free

    # C-src: a real copy of the source, each file in turn, summed.
    if free() < src_bytes * 1.1:
        ev["c_src_skipped"] = f"free {free()} bytes, under the source plus 10% ({int(src_bytes * 1.1)})"
    else:
        t = 0.0
        for f in files:
            t += dd(f, tmp)
            os.remove(tmp)
        ev["c_src_s"] = round(t, 3)
    # C-io: the transcode's own I/O with no compute: read the source, write the old arm's output.
    t = sum(dd(f, "/dev/null") for f in files)
    ev["c_io_read_s"] = round(t, 3)
    out_bytes = os.path.getsize(old_out)
    if free() < out_bytes + (1 << 30):
        ev["c_io_write_skipped"] = f"free {free()} bytes, under the output plus 1 GiB ({out_bytes + (1 << 30)})"
    else:
        ev["c_io_write_s"] = round(dd(old_out, tmp), 3)
        os.remove(tmp)
        ev["c_io_s"] = round(ev["c_io_read_s"] + ev["c_io_write_s"], 3)
    return ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin-dir", required=True, help="holds prequant-old and prequant-new")
    ap.add_argument("--out-dir", required=True, help="scratch directory on the filesystem a sidecar is written to")
    ap.add_argument("--record", required=True, help="JSON lines, one per event")
    ap.add_argument("--est-min", type=float, default=0, help="the estimate, for the expected-finish line")
    ap.add_argument("--no-copies", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="check the inputs and print the plan")
    ap.add_argument("cells", nargs="+")
    a = ap.parse_args()

    bins = {arm: os.path.join(a.bin_dir, f"prequant-{arm}") for arm in ("old", "new")}
    cells = []
    for spec in a.cells:
        name, source, quant, target, pairs = spec.split("|")
        cells.append({"name": name, "source": os.path.expanduser(source), "quant": quant, "target": target, "pairs": int(pairs)})
    missing = [p for p in bins.values() if not os.access(p, os.X_OK)] + [c["source"] for c in cells if not os.path.exists(c["source"])]
    for c in cells:
        if c["source"].startswith(("/Volumes/", "/srv/models")):
            sys.exit(f"{c['source']}: an archive path is not a bench surface")
    if missing:
        sys.exit("missing: " + ", ".join(missing))
    os.makedirs(a.out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.record)), exist_ok=True)
    provenance = ""
    try:
        provenance = open(os.path.join(a.bin_dir, "BUILD.txt")).read().strip()
    except OSError:
        pass
    log(f"started {time.strftime('%Y-%m-%d %H:%M:%S %Z')} on {platform.node()} ({platform.system()}/{platform.machine()})")
    if a.est_min:
        log(f"estimate {a.est_min:.0f} min, expect done ~{time.strftime('%H:%M', time.localtime(T0 + a.est_min * 60))}")
    log(f"binaries: {provenance or a.bin_dir}")
    for c in cells:
        log(f"cell {c['name']}: {c['quant']}:{c['target']} from {c['source']}, warm-up per arm then {c['pairs']} pair(s)")
    log(f"free in {a.out_dir}: {shutil.disk_usage(a.out_dir).free / 1e9:.1f} GB")
    if a.dry_run:
        return 0

    rec = open(a.record, "a")

    def emit(ev):
        ev = {"box": platform.node(), "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **ev}
        rec.write(json.dumps(ev) + "\n")
        rec.flush()

    emit({"event": "start", "provenance": provenance, "cells": cells, "argv": sys.argv[1:]})
    failed, results = False, {}
    for ci, c in enumerate(cells):
        res = results.setdefault(c["name"], {"old": [], "new": [], "sha": {"old": set(), "new": set()}, "copies": [], "peak": {"old": [], "new": []}})
        outs = {arm: os.path.join(a.out_dir, f"{c['name']}.{arm}.giw") for arm in ("old", "new")}

        def arm_run(arm, phase):
            nonlocal failed
            log(f"{c['name']} ({ci + 1}/{len(cells)}) {phase}: {arm} ...")
            ev = transcode(bins[arm], c, outs[arm])
            if "error" in ev:
                failed = True
                log(f"  FAILED rc={ev['rc']}: {ev['error'][-1] if ev['error'] else ''}")
            else:
                ev["sha256"] = sha256(outs[arm])
                res["sha"][arm].add(ev["sha256"])
                if phase != "warm-up":
                    res[arm].append(ev)
                    res["peak"][arm].append(ev.get("peak_footprint_bytes") or ev.get("max_rss_bytes") or 0)
                log(f"  {arm}: wall {ev['wall_s']:.1f}s ({ev.get('wrote_line', 'no line')[-58:]}), sha {ev['sha256'][:16]}")
            emit({"event": "transcode", "cell": c["name"], "phase": phase, "arm": arm, **ev})
            return "error" not in ev

        def copy_run(phase):
            if a.no_copies or not os.path.exists(outs["old"]):
                return
            try:
                ev = copies(c, outs["old"], a.out_dir)
            except Exception as e:  # a failed copy is recorded and does not stop the transcodes
                ev = {"error": str(e)}
            res["copies"].append(ev)
            log(f"  copies: C-src {ev.get('c_src_s', ev.get('c_src_skipped', '-'))}, C-io {ev.get('c_io_s', ev.get('c_io_write_skipped', '-'))}")
            emit({"event": "copies", "cell": c["name"], "phase": phase, **ev})

        for arm in ("old", "new"):
            arm_run(arm, "warm-up")
            remove(outs[arm])
        for k in range(c["pairs"]):
            phase = f"pair {k + 1}"
            if k % 2 == 0:
                if arm_run("old", phase):
                    copy_run(phase)
                remove(outs["old"])
                arm_run("new", phase)
                remove(outs["new"])
            else:
                arm_run("new", phase)
                remove(outs["new"])
                if arm_run("old", phase):
                    copy_run(phase)
                remove(outs["old"])

    log("== summary (paired, per cell; the bands are in the task doc) ==")
    for name, r in results.items():
        shas = r["sha"]["old"] | r["sha"]["new"]
        identical = len(shas) == 1 and r["sha"]["old"] and r["sha"]["new"]
        if not identical:
            failed = True
        n = min(len(r["old"]), len(r["new"]))
        line = {"event": "summary", "cell": name, "identical": bool(identical), "pairs": n}
        if n:
            wall = [r["new"][i]["wall_s"] / r["old"][i]["wall_s"] for i in range(n)]
            line["wall_new_over_old"] = [round(x, 4) for x in wall]
            line["wall_new_over_old_median"] = round(statistics.median(wall), 4)
            if all("stream_s" in e for e in r["old"][:n] + r["new"][:n]):
                st = [r["new"][i]["stream_s"] / r["old"][i]["stream_s"] for i in range(n) if r["old"][i]["stream_s"] > 0]
                line["stream_new_over_old"] = [round(x, 4) for x in st]
            line["peak_bytes_old_max"] = max(r["peak"]["old"])
            line["peak_bytes_new_max"] = max(r["peak"]["new"])
            cio = [e["c_io_s"] for e in r["copies"] if "c_io_s" in e]
            csrc = [e["c_src_s"] for e in r["copies"] if "c_src_s" in e]
            for key, floor in (("c_io", cio), ("c_src", csrc)):
                if floor:
                    f = statistics.median(floor)
                    line[f"{key}_s_median"] = round(f, 3)
                    line[f"old_over_{key}"] = round(statistics.median(e["wall_s"] for e in r["old"]) / f, 2)
                    line[f"new_over_{key}"] = round(statistics.median(e["wall_s"] for e in r["new"]) / f, 2)
        emit(line)
        log(json.dumps({k: v for k, v in line.items() if k != "event"}))
    emit({"event": "end", "failed": failed, "wall_s": round(time.time() - T0, 1)})
    log(f"finished in {(time.time() - T0) / 60:.1f} min: {'FAILED (a transcode failed or two arms differ)' if failed else 'every cell ran and its arms wrote the same bytes'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
