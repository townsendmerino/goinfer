#!/usr/bin/env python3
"""Gate 3's in-process half (README beside this file): BenchmarkResidentDecode on Phi-3 mini (q4k, serve's f16 KV through the scratch
init), A = the hd-96 lane off (GOINFER_CUDA_FLASH_DECODE=0), B = on (GOINFER_CUDA_FLASH_DECODE=16 and GOINFER_BENCH_FA_HD96=1), one binary,
A B B A x 5 blocks at depth 3,900, then at 128. Under the exclusive timing lock; each cell waits for a 1-minute load under 1.0, records its
start state, and is VOID (re-run in place, listed) if it starts above load 8.0 or beside a foreign build, test or serve process.
  python3 scripts/timing_lock.py run --label hd96-gate3 -- python3 run_gate3_inproc.py --bin BIN --out DIR"""
import argparse, datetime, os, re, subprocess, sys, time

MODEL = os.path.expanduser("~/models/phi3-mini-4k-gguf/Phi-3-mini-4k-instruct-q4.gguf")
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
    ap.add_argument("--bin", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--depths", default="3900,128")
    ap.add_argument("--blocks", type=int, default=5)
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "logs"), exist_ok=True)
    tsv = os.path.join(a.out, "cells.tsv")
    new = not os.path.exists(tsv)
    with open(tsv, "a") as out:
        if new:
            out.write("start\tdepth\tblock\tcell\tarm\tsettle_s\tload1\tmemavail_mib\tforeign\tgpu_temp\ttok_s\twall_s\n")
        for depth in [int(d) for d in a.depths.split(",")]:
            for block in range(1, a.blocks + 1):
                for cell, arm in enumerate("ABBA", 1):
                    for attempt in range(1, 4):
                        if datetime.datetime.now().strftime("%H%M") >= "2125":
                            print("STOP: 21:25 PT reached", flush=True)
                            sys.exit(3)
                        t0 = time.time()
                        while load1() >= 1.0 and time.time() - t0 < 600:
                            time.sleep(10)
                        waited = round(time.time() - t0)
                        l1, avail, foreign, temp = state({os.getpid(), os.getppid()})
                        start = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
                        log = os.path.join(a.out, "logs", f"d{depth}-b{block}-c{cell}-{arm}-a{attempt}.log")
                        t1 = time.time()
                        if l1 > 8.0 or foreign != "-":
                            tok = f"VOID start load {l1} foreign {foreign}"
                        else:
                            env = dict(os.environ, GOINFER_BENCH_MODEL=MODEL, GOINFER_BENCH_QUANT="q4k", GOINFER_BENCH_DEPTH=str(depth), GAP_SERVE_AUTO="1")
                            env.pop("GOINFER_BENCH_FA_HD96", None)
                            if arm == "B":
                                env["GOINFER_CUDA_FLASH_DECODE"], env["GOINFER_BENCH_FA_HD96"] = "16", "1"
                            else:
                                env["GOINFER_CUDA_FLASH_DECODE"] = "0"
                            with open(log, "w") as lf:
                                rc = subprocess.run([a.bin, "-test.run", "^$", "-test.bench", "^BenchmarkResidentDecode$", "-test.benchtime", "128x",
                                                     "-test.count", "1", "-test.v", "-test.timeout", "15m"], env=env, stdout=lf, stderr=subprocess.STDOUT,
                                                    cwd=os.path.join(REPO, "backend/cuda")).returncode
                            text = open(log).read()
                            m = re.search(r"([0-9.]+) tok/s", text)
                            tok = m.group(1) if (m and rc == 0) else f"FAILED rc={rc}"
                            if "gap-scratch: kvF16Auto restored to true" not in text:
                                tok = "VOID no-serve-auto-marker"
                        wall = time.time() - t1
                        out.write(f"{start}\t{depth}\t{block}\t{cell}\t{arm}\t{waited}\t{l1}\t{avail}\t{foreign}\t{temp}\t{tok}\t{wall:.1f}\n")
                        out.flush()
                        print(f"{start} d{depth} b{block} c{cell} {arm} a{attempt} settle {waited}s load {l1} -> {tok} ({wall:.0f}s)", flush=True)
                        if tok.startswith("FAILED"):
                            sys.exit(f"FAILED d{depth} b{block} c{cell}")
                        if not tok.startswith("VOID"):
                            break
                    else:
                        sys.exit(f"STOPPED: d{depth} b{block} c{cell} void three times")
    print("GATE3-INPROC-DONE", flush=True)

if __name__ == "__main__":
    main()
