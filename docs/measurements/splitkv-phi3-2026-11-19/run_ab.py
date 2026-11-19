#!/usr/bin/env python3
"""One A B B A block of BenchmarkResidentDecode on Phi-3 (README.md beside this file).

Run under the exclusive timing lock, one block per hold:
  python3 scripts/timing_lock.py run --wait 7200 --label splitkv-phi3 -- python3 run_ab.py --depth D --block K --bin BIN --out OUT.tsv
Each cell appends one row to OUT.tsv: what the box looked like at the cell's start (the void rule's inputs) and the cell's tok/s.
"""
import argparse, datetime, os, re, subprocess, time

MODEL = os.path.expanduser("~/models/phi3-mini-4k-gguf/Phi-3-mini-4k-instruct-q4.gguf")
FOREIGN = re.compile(r"^(compile|link|cgo|asm|vet|nvcc|cicc|ptxas|gopls|go|serve|ollama|llama-server|.*\.test)$")

def box_state(mine):
    load1 = float(open("/proc/loadavg").read().split()[0])
    avail = 0
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            avail = int(line.split()[1]) // 1024
    ps = subprocess.run(["ps", "-eo", "pid,ppid,comm", "--no-headers"], capture_output=True, text=True).stdout.split("\n")
    foreign = []
    for row in ps:
        f = row.split(None, 2)
        if len(f) == 3 and FOREIGN.match(f[2]) and int(f[0]) not in mine:
            foreign.append(f"{f[2]}:{f[0]}")
    gpu = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip().replace("\n", "; ")
    temp = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,memory.used", "--format=csv,noheader"],
                          capture_output=True, text=True).stdout.strip()
    return load1, avail, foreign, gpu, temp

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, required=True)
    ap.add_argument("--block", type=int, required=True)
    ap.add_argument("--bin", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--logdir", required=True)
    a = ap.parse_args()
    os.makedirs(a.logdir, exist_ok=True)
    new = not os.path.exists(a.out)
    with open(a.out, "a") as out:
        if new:
            out.write("start\tdepth\tblock\tcell\tarm\tload1\tmemavail_mib\tforeign\tgpu_apps\tgpu_temp_mem\ttok_s\twall_s\n")
        for cell, arm in enumerate("ABBA", 1):
            env = dict(os.environ, GOINFER_BENCH_MODEL=MODEL, GOINFER_BENCH_QUANT="q4k", GOINFER_BENCH_DEPTH=str(a.depth))
            env.pop("GOINFER_SPLITKV_MIN_KEYS", None)
            if arm == "B":
                env["GOINFER_SPLITKV_MIN_KEYS"] = "0"
            mine = {os.getpid(), os.getppid()}
            load1, avail, foreign, gpu, temp = box_state(mine)
            start = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
            log = os.path.join(a.logdir, f"d{a.depth}-b{a.block}-c{cell}-{arm}.log")
            t0 = time.time()
            with open(log, "w") as lf:
                rc = subprocess.run([a.bin, "-test.run", "^$", "-test.bench", "^BenchmarkResidentDecode$", "-test.benchtime", "128x",
                                     "-test.count", "1", "-test.v", "-test.timeout", "15m"], env=env, stdout=lf, stderr=subprocess.STDOUT).returncode
            wall = time.time() - t0
            m = re.search(r"([0-9.]+) tok/s", open(log).read())
            tok = m.group(1) if (m and rc == 0) else f"FAILED rc={rc}"
            out.write(f"{start}\t{a.depth}\t{a.block}\t{cell}\t{arm}\t{load1}\t{avail}\t{','.join(foreign) or '-'}\t{gpu}\t{temp}\t{tok}\t{wall:.1f}\n")
            out.flush()
            print(f"{start} d{a.depth} b{a.block} c{cell} {arm} load {load1} avail {avail} foreign {foreign or '-'} -> {tok} ({wall:.0f}s)", flush=True)

if __name__ == "__main__":
    main()
