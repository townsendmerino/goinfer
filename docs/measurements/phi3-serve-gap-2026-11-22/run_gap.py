#!/usr/bin/env python3
"""Phi-3 mini on CUDA, in-process against served, A B B A (README.md beside this file). Run under the exclusive timing lock:
  python3 scripts/timing_lock.py run --label phi3-gap -- python3 run_gap.py --test-bin T --serve-bin S --out DIR
A = BenchmarkResidentDecode (the tree's, plus the scratch init restoring serve's f16 KV), 128 timed tokens after a prefill of D.
B = bench_peer.py's goinfer arm alone (serve over HTTP, 3 runs x 8 completions x 64 tokens), the 11-19 peer cell's instrument.
Before every cell the 1-minute load must settle under 1.0 (both arms, the harness's own idle gate), and the cell's start state is
recorded; a cell started above load 8.0 or with a foreign build or serve process is VOID and re-run in place. Writes DIR/cells.tsv."""
import argparse, datetime, json, os, re, statistics, subprocess, sys, time

MODEL = os.path.expanduser("~/models/phi3-mini-4k-gguf/Phi-3-mini-4k-instruct-q4.gguf")
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))
FOREIGN = re.compile(r"^(compile|link|cgo|asm|vet|nvcc|cicc|ptxas|go|ollama|llama-server|.*\.test|serve.*)$")


def load1():
    return float(open("/proc/loadavg").read().split()[0])


def state(mine):
    avail = next(int(l.split()[1]) // 1024 for l in open("/proc/meminfo") if l.startswith("MemAvailable:"))
    swap = {l.split(":")[0]: int(l.split()[1]) for l in open("/proc/meminfo") if l.startswith(("SwapTotal", "SwapFree"))}
    ps = subprocess.run(["ps", "-eo", "pid,comm", "--no-headers"], capture_output=True, text=True).stdout.splitlines()
    foreign = [f"{c}:{p}" for p, c in (r.split(None, 1) for r in ps if r.strip()) if FOREIGN.match(c.strip()) and int(p) not in mine]
    gpu = subprocess.run(["nvidia-smi", "--query-compute-apps=process_name,used_memory", "--format=csv,noheader"], capture_output=True,
                         text=True).stdout.strip().replace("\n", "; ")
    temp = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    return dict(load1=load1(), avail=avail, swap_mib=(swap["SwapTotal"] - swap["SwapFree"]) // 1024, foreign=",".join(foreign) or "-",
                gpu=gpu, temp=temp)


def settle(limit=1.0, max_s=600):
    t0 = time.time()
    while load1() >= limit and time.time() - t0 < max_s:
        time.sleep(10)
    return round(time.time() - t0)


def arm_a(a, depth, log):
    env = dict(os.environ, GOINFER_BENCH_MODEL=MODEL, GOINFER_BENCH_QUANT="q4k", GOINFER_BENCH_DEPTH=str(depth), GAP_SERVE_AUTO="1")
    with open(log, "w") as lf:
        rc = subprocess.run([a.test_bin, "-test.run", "^$", "-test.bench", "^BenchmarkResidentDecode$", "-test.benchtime", "128x",
                             "-test.count", "1", "-test.v", "-test.timeout", "15m"], env=env, stdout=lf, stderr=subprocess.STDOUT,
                            cwd=os.path.join(REPO, "backend/cuda")).returncode
    text = open(log).read()
    m = re.search(r"([0-9.]+) tok/s", text)
    if rc != 0 or not m:
        return f"FAILED rc={rc}", ""
    if "gap-scratch: kvF16Auto restored to true" not in text:
        return "VOID no-serve-auto-marker", ""
    return m.group(1), ""


def arm_b(a, depth, log, res):
    env = dict(os.environ, GOINFER_SERVE_CUDA=a.serve_bin, GOINFER_SERVE=a.serve_bin, BENCH_ENGINES="goinfer", BENCH_MODELS="phi3-mini",
               BENCH_BACKENDS="cuda", BENCH_RUNS="3", BENCH_QUANT_OVERRIDE="phi3-mini=default",
               BENCH_PHASES="A" if depth == 128 else "B", BENCH_DEPTHS="none" if depth == 128 else str(depth))
    with open(log, "w") as lf:
        rc = subprocess.run([sys.executable, "-u", "scripts/bench_peer.py", res], env=env, stdout=lf, stderr=subprocess.STDOUT, cwd=REPO).returncode
    try:
        cells = [c for c in json.load(open(res)) if isinstance(c, dict) and c.get("engine") == "goinfer" and c.get("depth") == depth]
    except (OSError, ValueError):
        cells = []
    if rc != 0 or len(cells) != 1 or not cells[0].get("runs"):
        return f"FAILED rc={rc} cells={len(cells)}", ""
    c = cells[0]
    k = c["counts"] if isinstance(c["counts"], dict) else {}
    if (k.get("swap") or {}).get("swap_void") or (k.get("token_gate") or {}).get("verdict") != "ok":
        return f"VOID {(k.get('swap') or {}).get('swap_void')} {(k.get('token_gate') or {}).get('verdict')}", ""
    if not str(k.get("decode_path", "")).startswith("cuda-resident (q4k)"):
        return f"VOID decode path {k.get('decode_path')}", ""
    return f"{statistics.mean(c['runs']):.2f}", "/".join(f"{r:.2f}" for r in c["runs"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-bin", required=True)
    ap.add_argument("--serve-bin", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--depths", default="128,3900")
    ap.add_argument("--blocks", type=int, default=2)
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "logs"), exist_ok=True)
    tsv = os.path.join(a.out, "cells.tsv")
    new = not os.path.exists(tsv)
    with open(tsv, "a") as out:
        if new:
            out.write("start\tdepth\tblock\tcell\tarm\tsettle_s\tload1\tmemavail_mib\tswap_mib\tforeign\tgpu_apps\tgpu_temp\ttok_s\truns\twall_s\n")
        for depth in [int(d) for d in a.depths.split(",")]:
            for block in range(1, a.blocks + 1):
                for cell, arm in enumerate("ABBA", 1):
                    for attempt in range(1, 4):
                        if datetime.datetime.now().strftime("%H%M") >= "2125":
                            print("STOP: 21:25 PT reached", flush=True)
                            sys.exit(3)
                        waited = settle()
                        st = state({os.getpid(), os.getppid()})
                        start = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
                        tag = f"d{depth}-b{block}-c{cell}-{arm}-a{attempt}"
                        t0 = time.time()
                        void = st["load1"] > 8.0 or st["foreign"] != "-"
                        if void:
                            tok, runs = f"VOID start load {st['load1']} foreign {st['foreign']}", ""
                        elif arm == "A":
                            tok, runs = arm_a(a, depth, os.path.join(a.out, "logs", tag + ".log"))
                        else:
                            tok, runs = arm_b(a, depth, os.path.join(a.out, "logs", tag + ".log"), os.path.join(a.out, "logs", tag + ".json"))
                        wall = time.time() - t0
                        out.write(f"{start}\t{depth}\t{block}\t{cell}\t{arm}\t{waited}\t{st['load1']}\t{st['avail']}\t{st['swap_mib']}\t"
                                  f"{st['foreign']}\t{st['gpu']}\t{st['temp']}\t{tok}\t{runs}\t{wall:.1f}\n")
                        out.flush()
                        print(f"{start} {tag} settle {waited}s load {st['load1']} -> {tok} ({wall:.0f}s)", flush=True)
                        if not tok.startswith(("VOID", "FAILED")):
                            break
                        if tok.startswith("FAILED"):
                            sys.exit(f"FAILED: {tag}: {tok}")
                    else:
                        sys.exit(f"STOPPED: {tag} void three times")
    print("GAP-RUN-DONE", flush=True)


if __name__ == "__main__":
    main()
