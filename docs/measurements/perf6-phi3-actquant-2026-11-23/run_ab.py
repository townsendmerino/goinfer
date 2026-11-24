#!/usr/bin/env python3
"""Perf finding 6's speed check (README beside this file): BenchmarkResidentDecode at depth 128, A = a test binary of the base commit
(the per-32 activation kernels on one block), B = one of this change (a warp a group, spread over the grid), both built with the
serve-auto scratch init (f16 KV, as serve loads), A B B A x 5 blocks, Phi-3 mini q4k (--models phi3). Under the exclusive timing lock;
each cell waits for a 1-minute load under 1.0, records its start state, and is VOID (re-run in place, listed) if it starts above load 8.0,
with under 5000 MiB available, or beside a foreign build, test or serve process.
  python3 scripts/timing_lock.py run --label perf6-ab -- python3 run_ab.py --a BIN_A --b BIN_B --out DIR --models phi3"""
import argparse, datetime, os, re, subprocess, sys, time

MODELS = [
    ("q05", os.path.expanduser("~/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf"), ""),
    ("phi3", os.path.expanduser("~/models/phi3-mini-4k-gguf/Phi-3-mini-4k-instruct-q4.gguf"), "q4k"),
]
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))
FOREIGN = re.compile(r"^(compile|link|cgo|asm|vet|nvcc|cicc|ptxas|go|ollama|llama-server|.*\.test|serve.*)$")


def load1():
    return float(open("/proc/loadavg").read().split()[0])


def state(mine):
    avail = next(int(l.split()[1]) // 1024 for l in open("/proc/meminfo") if l.startswith("MemAvailable:"))
    ps = subprocess.run(["ps", "-eo", "pid,comm", "--no-headers"], capture_output=True, text=True).stdout.splitlines()
    foreign = [f"{c.strip()}:{p}" for p, c in (r.split(None, 1) for r in ps if r.strip()) if FOREIGN.match(c.strip()) and int(p) not in mine]
    temp = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    return load1(), avail, ",".join(foreign) or "-", temp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", default="q05,phi3")
    ap.add_argument("--blocks", type=int, default=5)
    a = ap.parse_args()
    bins = {"A": a.a, "B": a.b}
    os.makedirs(os.path.join(a.out, "logs"), exist_ok=True)
    tsv = os.path.join(a.out, "cells.tsv")
    new = not os.path.exists(tsv)
    with open(tsv, "a") as out:
        if new:
            out.write("start\tmodel\tblock\tcell\tarm\tsettle_s\tload1\tmemavail_mib\tforeign\tgpu_temp\ttok_s\twall_s\n")
        for name, path, quant in [m for m in MODELS if m[0] in a.models.split(",")]:
            for block in range(1, a.blocks + 1):
                for cell, arm in enumerate("ABBA", 1):
                    for attempt in range(1, 6):
                        if datetime.datetime.now().strftime("%H%M") >= "2125":
                            print("STOP: 21:25 PT reached", flush=True)
                            sys.exit(3)
                        t0 = time.time()
                        while load1() >= 1.0 and time.time() - t0 < 600:
                            time.sleep(10)
                        waited = round(time.time() - t0)
                        l1, avail, foreign, temp = state({os.getpid(), os.getppid()})
                        start = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
                        log = os.path.join(a.out, "logs", f"{name}-b{block}-c{cell}-{arm}-a{attempt}.log")
                        t1 = time.time()
                        if l1 > 8.0 or avail < 5000 or foreign != "-":
                            tok = f"VOID start load {l1} avail {avail} foreign {foreign}"
                        else:
                            env = dict(os.environ, GOINFER_BENCH_MODEL=path, GOINFER_BENCH_QUANT=quant, GOINFER_BENCH_DEPTH="128", GAP_SERVE_AUTO="1")
                            with open(log, "w") as lf:
                                rc = subprocess.run([bins[arm], "-test.run", "^$", "-test.bench", "^BenchmarkResidentDecode$", "-test.benchtime", "128x",
                                                     "-test.count", "1", "-test.v", "-test.timeout", "15m"], env=env, stdout=lf, stderr=subprocess.STDOUT,
                                                    cwd=os.path.join(REPO, "backend/cuda")).returncode
                            text = open(log).read()
                            m = re.search(r"([0-9.]+) tok/s", text)
                            tok = m.group(1) if (m and rc == 0) else f"FAILED rc={rc}"
                            if "gap-scratch: kvF16Auto restored to true" not in text:
                                tok = "VOID no-serve-auto-marker"
                        wall = time.time() - t1
                        out.write(f"{start}\t{name}\t{block}\t{cell}\t{arm}\t{waited}\t{l1}\t{avail}\t{foreign}\t{temp}\t{tok}\t{wall:.1f}\n")
                        out.flush()
                        print(f"{start} {name} b{block} c{cell} {arm} a{attempt} settle {waited}s load {l1} -> {tok} ({wall:.0f}s)", flush=True)
                        if tok.startswith("FAILED"):
                            sys.exit(f"FAILED {name} b{block} c{cell}")
                        if not tok.startswith("VOID"):
                            break
                    else:
                        sys.exit(f"STOPPED: {name} b{block} c{cell} void five times")
    print("PERF6-AB-DONE", flush=True)


if __name__ == "__main__":
    main()
