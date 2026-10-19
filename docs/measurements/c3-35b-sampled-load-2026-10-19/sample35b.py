#!/usr/bin/env python3
"""A sampled load of one model with the system figures beside the process's (c3-35b-sampled-load-2026-10-19/README.md).

build_peak.py's cell() samples the process's VmRSS/RssAnon/RssFile/VmHWM/VmSwap every 50 ms from spawn; this adds /proc/meminfo's swap in use, MemAvailable, Cached and
Mlocked to every sample (by replacing its status reader), then reads the per-sample JSON build_peak wrote for the peaks and the instant of each.

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


def status(pid):
    d = _orig(pid)
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
    print(json.dumps(rep, indent=1))
    json.dump(rep, open(os.path.join(a.out, "report.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
