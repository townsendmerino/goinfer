#!/usr/bin/env python3
"""A sampled load of one model with the system figures beside the process's (c3-35b-sampled-load-2026-10-19/README.md).

build_peak.py's cell() samples the process's VmRSS/RssAnon/RssFile/VmHWM/VmSwap every 50 ms from spawn; this adds /proc/meminfo's swap in use, MemAvailable, Cached and
Mlocked, and the user slice's memory.current and memory.events high count, to every sample (by replacing its status reader), then reads the per-sample JSON build_peak wrote for the peaks and the instant of each.

  python3 sample35b.py --serve <binary> --model <gguf> --out <dir> [--extra -ctx,32768,-moe-cache-experts]
"""
import argparse, importlib.util, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("build_peak", os.path.join(HERE, "..", "cuda-build-heap-2026-10-18", "build_peak.py"))
bp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bp)
MiB = 1 << 20
_orig = bp.status


def meminfo():
    d = {}
    try:
        for line in open("/proc/meminfo"):
            k, _, v = line.partition(":")
            if k in ("SwapTotal", "SwapFree", "MemAvailable", "Cached", "Mlocked"):
                d[k] = int(v.split()[0]) * 1024
    except OSError:
        pass
    return d


SLICE = "/sys/fs/cgroup/user.slice/user-1000.slice"


def slice_figures():
    """The user slice's memory.current and the 'high' count of memory.events: how often the slice crossed its MemoryHigh and was throttled into reclaim."""
    d = {}
    try:
        d["SliceCur"] = int(open(SLICE + "/memory.current").read())
        for line in open(SLICE + "/memory.events"):
            k, _, v = line.partition(" ")
            if k == "high":
                d["SliceHigh"] = int(v)
    except (OSError, ValueError):
        pass
    return d


def status(pid):
    d = _orig(pid)
    d.update(slice_figures())
    m = meminfo()
    if "SwapTotal" in m and "SwapFree" in m:
        d["SysSwapUsed"] = m["SwapTotal"] - m["SwapFree"]
    for k in ("MemAvailable", "Cached", "Mlocked"):
        if k in m:
            d["Sys" + k] = m[k]
    return d


bp.status = status


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--serve", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--extra", default="-ctx,32768,-moe-cache-experts")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    bp.cell(a.serve, "sampled", a.model, a.extra.split(","), a.out)
    d = json.load(open(os.path.join(a.out, "cell-sampled.json")))
    s = [x for x in d["samples"] if "RssAnon" in x]
    if not s:
        sys.exit("no samples")
    G = 1e9
    peak = lambda k: max(s, key=lambda x: x.get(k, 0))
    pa, pf, pr, ps = peak("RssAnon"), peak("RssFile"), peak("VmHWM"), peak("VmSwap")
    sw0 = s[0].get("SysSwapUsed", 0)
    rep = {
        "ready_s": d["rec"].get("ready_s"),
        "peak_RssAnon_GB": round(pa["RssAnon"] / G, 2), "peak_RssAnon_at_s": pa["t"],
        "peak_RssFile_GB": round(pf["RssFile"] / G, 2), "peak_RssFile_at_s": pf["t"],
        "VmHWM_GB": round(pr["VmHWM"] / G, 2), "at_HWM_RssAnon_GB": round(pr.get("RssAnon", 0) / G, 2), "at_HWM_RssFile_GB": round(pr.get("RssFile", 0) / G, 2),
        "peak_VmSwap_MB": round(ps.get("VmSwap", 0) / MiB), "final_VmSwap_MB": round(s[-1].get("VmSwap", 0) / MiB),
        "sys_swap_growth_MB": round((max(x.get("SysSwapUsed", 0) for x in s) - sw0) / MiB),
        "peak_Mlocked_GB": round(max(x.get("SysMlocked", 0) for x in s) / G, 2),
        "min_MemAvailable_GB": round(min(x.get("SysMemAvailable", 1 << 60) for x in s) / G, 2),
        "final_RssAnon_GB": round(s[-1]["RssAnon"] / G, 2), "final_RssFile_GB": round(s[-1]["RssFile"] / G, 2),
        "prereg_bar_GB": 19.6, "verdict": "PASS" if pa["RssAnon"] / G <= 19.6 else "FAIL",
    }
    # The user slice, every 50 ms: does the slice limit still force reclaim? "high" counts the times the slice went over MemoryHigh; its growth over the load,
    # with memory.current against the limit, is the answer. The setting is read, never changed.
    sl = [x for x in s if "SliceCur" in x]
    if sl:
        try:
            lim = open(SLICE + "/memory.high").read().strip()
        except OSError:
            lim = "unknown"
        rises = [x["t"] for prev, x in zip(sl, sl[1:]) if x["SliceHigh"] > prev["SliceHigh"]]
        rep["slice_memory_high"] = lim
        rep["slice_current_GB_start_peak_end"] = [round(sl[0]["SliceCur"] / G, 2), round(max(x["SliceCur"] for x in sl) / G, 2), round(sl[-1]["SliceCur"] / G, 2)]
        rep["slice_high_events_start_end_growth"] = [sl[0]["SliceHigh"], sl[-1]["SliceHigh"], sl[-1]["SliceHigh"] - sl[0]["SliceHigh"]]
        rep["slice_high_samples_that_rose"] = len(rises)
        if rises:
            rep["slice_high_first_last_rise_s"] = [rises[0], rises[-1]]
    # The Go heap from the collector's own lines in the serve log (GODEBUG=gctrace=1): "gc N @T s ...: A->B->C MB" is the heap before the collection, after
    # it, and the live heap; the timeline here is the cycle times with those three figures, and the peaks.
    import re
    cyc = []
    try:
        for m in re.finditer(r"gc \d+ @([\d.]+)s \d+%: .*?\s(\d+)->(\d+)->(\d+) MB", open(os.path.join(a.out, "serve-sampled.log")).read()):
            cyc.append((float(m.group(1)), int(m.group(2)), int(m.group(4))))
    except OSError:
        pass
    if cyc:
        rep["heap_peak_alloc_MB"], rep["heap_peak_live_MB"] = max(c[1] for c in cyc), max(c[2] for c in cyc)
        rep["heap_timeline_s_alloc_live_MB"] = [c for c in cyc if c[0] > 12][::2]
    # RssAnon / RssFile / VmSwap / Mlocked about every 2 s, for the timeline the README asks for
    step = max(1, int(2.0 / 0.05))
    rep["timeline_t_anon_file_swap_mlocked_GB"] = [[x["t"], round(x["RssAnon"] / G, 2), round(x.get("RssFile", 0) / G, 2), round(x.get("VmSwap", 0) / G, 3), round(x.get("SysMlocked", 0) / G, 2)] for x in s[::step]]
    print(json.dumps(rep, indent=1))
    json.dump(rep, open(os.path.join(a.out, "report.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
