#!/usr/bin/env python3
"""KS-F2b's timed A/B (docs/measurements/ks-f2b-2026-10-25/README.md): the old attendQueryP (A, main's build) against the fanned
schedule (B, the measurement branch's build), one process a run, through TestKSF2_cpuDecodeSpeed. Per cell: 2 warm runs of each
arm, then 10 valid pairs, A B then B A alternately. A run is void when the 1-minute load at its start is over 4.0, when reclaimable
memory at its start is under the cell's need plus 2 GiB, or when the machine pages or swaps during it; a void pair is re-run, up to
6 times a cell. Usage: run_ksf2b.py <model-dir> <depth> <need-GiB> <out.jsonl>. Run under timing_lock.py (exclusive)."""
import json, os, re, subprocess, sys, tempfile, time

D = os.path.expanduser("~/goinfer-bench/ksf2b-night")
WT = os.path.expanduser("~/goinfer-bench/ksf2b-wt/decoder")
model, depth, need, out = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), sys.argv[4]

def load1():
    return float(subprocess.run(["sysctl", "-n", "vm.loadavg"], capture_output=True, text=True).stdout.strip("{} \n").split()[0])

def recl():
    return int(subprocess.run([f"{D}/hostmem-cur", "-reclaimable"], capture_output=True, text=True).stdout.strip())

def paging():
    o = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    return sum(int(m) for m in re.findall(r"(?:Pageouts|Swapouts):\s+(\d+)\.", o))

def run(arm):
    l0, r0, p0 = load1(), recl(), paging()
    with tempfile.NamedTemporaryFile("r", suffix=".jsonl") as f:
        env = dict(os.environ, KSF2_MODEL=model, KSF2_DEPTH=str(depth), KSF2_TOKENS="32", KSF2_OUT=f.name, KSF2_ARM=arm)
        p = subprocess.run([f"{D}/decoder-{arm}.test", "-test.run", "^TestKSF2_cpuDecodeSpeed$", "-test.count=1"], cwd=WT, env=env,
                           capture_output=True, text=True)
        rec = json.loads(f.read().strip().splitlines()[-1]) if p.returncode == 0 else None
    p1 = paging()
    if rec is None:
        sys.exit(f"{arm} run failed:\n{p.stdout[-2000:]}\n{p.stderr[-2000:]}")
    valid = l0 <= 4.0 and r0 >= (need + 2) * 2**30 and p1 == p0
    rec.update(load1=l0, reclaimable_gib=round(r0 / 2**30, 2), paged=p1 - p0, valid=valid, t=time.strftime("%T"))
    return rec

with open(out, "a") as o:
    def log(rec, tag):
        rec["tag"] = tag
        o.write(json.dumps(rec) + "\n"); o.flush()
        print(f"[{rec['t']}] {tag} {rec['arm']} {rec['decode_ms_per_token']:.2f} ms/token load {rec['load1']} recl {rec['reclaimable_gib']} paged {rec['paged']} {'ok' if rec['valid'] else 'VOID'}", flush=True)
    for arm in "ABAB":
        log(run(arm), "warm")
    pairs, voids = [], 0
    while len(pairs) < 10:
        order = "AB" if len(pairs) % 2 == 0 else "BA"
        rs = {}
        for arm in order:
            rs[arm] = run(arm); log(rs[arm], f"pair{len(pairs)}")
        if not (rs["A"]["valid"] and rs["B"]["valid"]):
            voids += 1
            if voids > 6:
                sys.exit("over 6 void pairs")
            time.sleep(30); continue
        pairs.append(rs["A"]["decode_ms_per_token"] / rs["B"]["decode_ms_per_token"])
    s = sorted(pairs)
    med = (s[4] + s[5]) / 2
    faster = sum(1 for x in pairs if x > 1)
    grade = "MEETS the merge rule" if med >= 1.05 and faster >= 9 else "below the merge rule"
    summ = dict(tag="grade", model=model, depth=depth, median_speedup=round(med, 4), min=round(s[0], 4), max=round(s[-1], 4), faster=faster, voids=voids, grade=grade)
    o.write(json.dumps(summ) + "\n")
    print(json.dumps(summ), flush=True)
