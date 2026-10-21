#!/usr/bin/env python3
"""int4 resident loads on Metal, before and after the int4 lane's no-staging change (1cc27f17), registered in
docs/measurements/int4-staging-7b-2026-10-18/README.md. A Mac night job (night.py holds the timing lock).

Builds: serve-metal at a2732f00 (before: int4Concat stages each projection) and at 53267aee (main, with 1cc27f17).
- Graded: the 1.5B q4_k_m loaded directly from its .gguf (-direct-load), the path int4Concat runs on, in order A B B A.
- Control: the 7B q4_k_m on its default path (the sidecar .giw, aliased: no int4Concat), A B.
The 7B direct load does not fit this Mac (the fit guard prices weights, KV and the mapped source, about 9.3 GB).
Each file is read once, untimed, first, so the page cache does not favour the first load. Each load is a fresh server
with GODEBUG=gctrace=1 (the Go heap at every collection from the process's start). It is read once it serves: the lifetime
peak physical footprint (footprint.py) and swap before and after. Then it is stopped, with no request.
Usage: python3 -u run.py OUTDIR"""
import json, os, re, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
BIN = os.path.expanduser("~/goinfer-bench/int4-heap")
BUILDS = {"before": f"{BIN}/serve-metal-a2732f00", "after": f"{BIN}/serve-metal-53267aee"}
M15 = os.path.expanduser("~/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf")
M7 = os.path.expanduser("~/models/qwen2.5-7b-instruct-q4_k_m.gguf")
PLAN = [("1.5B-direct", M15, ["-direct-load"], b) for b in ["before", "after", "after", "before"]] +        [("7B-sidecar", M7, [], b) for b in ["before", "after"]]
OUT = sys.argv[1]


def swap_mb():
    m = re.search(r"used = ([0-9.]+)M", subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout)
    return float(m.group(1)) if m else None


def ts():
    return time.strftime("%H:%M:%S %Z") + " / " + time.strftime("%H:%M:%S UTC", time.gmtime())


def load(i, name, model, extra, build):
    log = os.path.join(OUT, f"load{i}-{name}-{build}.log")
    s0 = swap_mb()
    with open(log, "w") as lf:
        p = subprocess.Popen([BUILDS[build], "-model", f"bench={model}", "-backend", "metal", "-quant", "int4", "-ctx", "4096",
                              "-addr", f"127.0.0.1:{18960 + i}"] + extra, stdout=lf, stderr=subprocess.STDOUT,
                             env={**os.environ, "GODEBUG": "gctrace=1"})
    t0, state, fp = time.time(), "TIMEOUT", None
    while time.time() - t0 < 300:
        txt = open(log).read()
        if re.search(r"does not fit resident RAM|NOT loaded|resident build needs", txt):
            state = "REFUSED"
            break
        if "goinfer serving on" in txt:
            state = "SERVING"
            fp = subprocess.run([sys.executable, os.path.join(HERE, "footprint.py"), str(p.pid)], capture_output=True, text=True).stdout.strip()
            break
        if p.poll() is not None:
            state = "EXITED"
            break
        time.sleep(0.25)
    wall = time.time() - t0
    s1 = swap_mb()
    p.terminate()
    try:
        p.wait(30)
    except subprocess.TimeoutExpired:
        p.kill()
    txt = open(log).read()
    heaps = [max(int(a), int(b)) for a, b, _ in re.findall(r"(\d+)->(\d+)->(\d+) MB", txt)]
    dp = (re.search(r"decode path: (.*)", txt) or [None, None])[1]
    fit = (re.search(r"^\s*fit: (.*)$", txt, re.M) or [None, None])[1]
    peak = re.search(r"lifetime_max_phys_footprint (\d+) MB", fp or "")
    row = {"load": i, "case": name, "build": build, "state": state, "load_wall_s": round(wall, 2), "decode_path": dp, "fit": fit,
           "gc_count": len(heaps), "go_heap_max_mb": max(heaps) if heaps else None,
           "lifetime_peak_footprint_mb": int(peak.group(1)) if peak else None, "footprint": fp,
           "swap_before_mb": s0, "swap_after_mb": s1, "swap_growth_mb": None if None in (s0, s1) else round(s1 - s0, 2)}
    print(f"  {ts()} load {i} ({name}, {build}): {json.dumps(row)}", flush=True)
    return row


def main():
    os.makedirs(OUT, exist_ok=True)
    for b in BUILDS.values():
        if not os.access(b, os.X_OK):
            sys.exit(f"missing {b}")
    print(f"=== {ts()} START; warm the file pages (untimed)", flush=True)
    for m in (M15, M7, M7.replace(".gguf", ".int4.metal.v17.giw")):
        if os.path.exists(m):
            subprocess.run(["cat", m], stdout=subprocess.DEVNULL)
    rows = []
    for i, (name, model, extra, b) in enumerate(PLAN):
        rows.append(load(i, name, model, extra, b))
        time.sleep(10)  # the previous server's memory returned before the next load
    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump({"builds": BUILDS, "plan": [(n, m, e, b) for n, m, e, b in PLAN], "rows": rows}, f, indent=2)
    print(f"=== {ts()} END; records {OUT}", flush=True)
    sys.exit(2 if any(r["state"] != "SERVING" for r in rows) else 0)


if __name__ == "__main__":
    main()
