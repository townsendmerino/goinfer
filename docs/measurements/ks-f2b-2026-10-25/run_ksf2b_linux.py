#!/usr/bin/env python3
"""KS-F2b on nobara (docs/measurements/ks-f2b-2026-10-25/README.md, "nobara"): run_ksf2b.py's design on Linux. The old attendQueryP
(A, main before #247) against the fanned schedule (B, main after it), one process a run, through TestKSF2_cpuDecodeSpeed. Per cell:
2 warm runs of each arm, then 10 valid pairs, A B then B A alternately. A run is void when the 1-minute load at its start is over 8.0
(half nobara's 16 threads), when MemAvailable at its start is under the cell's need plus 2 GiB, or when the test process swaps
(its VmSwap peak, sampled every 0.5 s, above 0). Before each run an idle gate waits up to 5 minutes for a 1-minute load of 4.0
or less. A void pair is re-run, up to 6 times a cell. Each run also records the process's
VmHWM and the system's pswpin/pswpout delta (context only). Usage: run_ksf2b_linux.py <bin-dir> <model> <depth> <need-GiB> <out.jsonl>
[--fit]. --fit runs one B run at 2 tokens and prints its peak (the fit check). Run under timing_lock.py (exclusive)."""
import json, os, subprocess, sys, tempfile, time

bins, model, depth, need, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
fit = "--fit" in sys.argv[6:]
WT = os.path.expanduser("~/goinfer-bench/ks-f2b-B/decoder")  # cwd only: the binaries carry their code


def load1():
    return float(open("/proc/loadavg").read().split()[0])


def avail():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024


def swapio():
    v = dict(line.split() for line in open("/proc/vmstat"))
    return int(v["pswpin"]) + int(v["pswpout"])


def status_kib(pid, key):
    try:
        for line in open(f"/proc/{pid}/status"):
            if line.startswith(key + ":"):
                return int(line.split()[1])
    except OSError:
        pass
    return None


def settle():
    # The idle gate: a run that keeps 16 threads busy leaves the 1-minute load near 10 behind it, so without a wait the next run
    # starts over the void rule's 8.0 on the harness's own residue. Wait up to 5 minutes for 4.0 or less; the void rule still decides.
    t0 = time.time()
    while load1() > 4.0 and time.time() - t0 < 300:
        time.sleep(5)
    return round(time.time() - t0)


def run(arm, tokens=32):
    waited = settle()
    l0, a0, s0 = load1(), avail(), swapio()
    with tempfile.NamedTemporaryFile("r", suffix=".jsonl") as f:
        env = dict(os.environ, KSF2_MODEL=model, KSF2_DEPTH=str(depth), KSF2_TOKENS=str(tokens), KSF2_OUT=f.name, KSF2_ARM=arm)
        p = subprocess.Popen([f"{bins}/decoder-{arm}.test", "-test.run", "^TestKSF2_cpuDecodeSpeed$", "-test.count=1", "-test.v",
                              "-test.timeout=30m"], cwd=WT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        hwm = swap = 0
        while p.poll() is None:
            hwm = max(hwm, status_kib(p.pid, "VmHWM") or 0)
            swap = max(swap, status_kib(p.pid, "VmSwap") or 0)
            time.sleep(0.5)
        text = p.stdout.read()
        lines = f.read().strip().splitlines()
        rec = json.loads(lines[-1]) if p.returncode == 0 and lines else None
    if rec is None:
        sys.exit(f"{arm} run failed (rc {p.returncode}):\n{text[-3000:]}")
    valid = l0 <= 8.0 and a0 >= (need + 2) * 2**30 and swap == 0
    rec.update(load1=l0, avail_gib=round(a0 / 2**30, 2), vmhwm_mib=hwm // 1024, vmswap_peak_kib=swap,
               sys_swapio_pages=swapio() - s0, valid=valid, settled_s=waited, t=time.strftime("%T"))
    return rec


with open(out, "a") as o:
    def log(rec, tag):
        rec["tag"] = tag
        o.write(json.dumps(rec) + "\n")
        o.flush()
        print(f"[{rec['t']}] {tag} {rec['arm']} {rec['decode_ms_per_token']:.2f} ms/token prefill {rec['prefill_s']:.1f} s load {rec['load1']} "
              f"avail {rec['avail_gib']} GiB VmHWM {rec['vmhwm_mib']} MiB swap {rec['vmswap_peak_kib']} KiB {'ok' if rec['valid'] else 'void, re-run [expected]'}", flush=True)
    if fit:
        log(run("B", tokens=2), "fit")
        sys.exit(0)
    for arm in "ABAB":
        log(run(arm), "warm")
    pairs, voids = [], 0
    while len(pairs) < 10:
        order = "AB" if len(pairs) % 2 == 0 else "BA"
        rs = {}
        for arm in order:
            rs[arm] = run(arm)
            log(rs[arm], f"pair{len(pairs)}")
        if not (rs["A"]["valid"] and rs["B"]["valid"]):
            voids += 1
            if voids > 6:
                sys.exit("VOID cell: over 6 void pairs")
            time.sleep(30)
            continue
        pairs.append(rs["A"]["decode_ms_per_token"] / rs["B"]["decode_ms_per_token"])
    s = sorted(pairs)
    med = (s[4] + s[5]) / 2
    faster = sum(1 for x in pairs if x > 1)
    grade = "MEETS the merge rule" if med >= 1.05 and faster >= 9 else "below the merge rule"
    slower = med < 1.0 or faster < 9
    summ = dict(tag="grade", model=model, depth=depth, median_speedup=round(med, 4), min=round(s[0], 4), max=round(s[-1], 4),
                faster=faster, voids=voids, grade=grade, slower_than_old=slower)
    o.write(json.dumps(summ) + "\n")
    print(json.dumps(summ), flush=True)
    if slower:
        print(f"DECISION NEEDED: {model} at depth {depth} reads slower than the old schedule (median {med:.4f}, {faster} of 10 faster)", flush=True)
