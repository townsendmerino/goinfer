#!/usr/bin/env python3
"""B41 gates 1b and 1c (README.md here), on this Mac.

  python3 gate1.py --hostmem <cmd/hostmem binary> 1b            1 GiB of anonymous memory must lower the reading by 0.8-1.2 GiB
  python3 gate1.py --hostmem <cmd/hostmem binary> 1c --giw F    F's pages, cached by touching every one, must lower it by < 0.3 GB

Each prints the new reading (cmd/hostmem) and, beside it as the control, the old one (vm_stat: free + inactive + speculative +
purgeable). Exit 0 PASS, 1 FAIL, 2 VOID.
"""
import argparse, mmap, os, re, statistics, subprocess, sys, time

GIB = 1 << 30


def new_bytes(binary):
    return int(subprocess.run([binary], capture_output=True, text=True, check=True).stdout.strip())


def old_bytes():
    o = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True).stdout
    ps = int(re.search(r"page size of (\d+) bytes", o).group(1))
    return sum(int(re.search(re.escape(k) + r"\s+(\d+)\.", o).group(1))
               for k in ("Pages free:", "Pages inactive:", "Pages speculative:", "Pages purgeable:")) * ps


def swap_used():
    o = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
    m = re.search(r"used = ([\d.]+)M", o)
    return float(m.group(1)) * (1 << 20) if m else 0.0


def sample(binary, label, n=3):
    rows = []
    for _ in range(n):
        rows.append((new_bytes(binary), old_bytes()))
        time.sleep(1)
    new, old = statistics.median(r[0] for r in rows), statistics.median(r[1] for r in rows)
    print(f"{time.strftime('%H:%M:%S')} {label}: new {new / GIB:.3f} GiB, old {old / GIB:.3f} GiB, swap used {swap_used() / GIB:.3f} GiB")
    return new, old


PLANT_ANON = r'''
import sys, time
n = int(sys.argv[1]); b = bytearray(n)
for i in range(0, n, 4096): b[i] = 1
print("ready", flush=True); sys.stdin.read()
'''

PLANT_FILE = r'''
import mmap, sys
f = open(sys.argv[1], "rb"); m = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)
s = 0
for i in range(0, len(m), 16384): s += m[i]
m.close(); f.close(); print("done", s & 1, flush=True)
'''


def gate_1b(binary):
    s0 = swap_used()
    before = sample(binary, "before")
    p = subprocess.Popen([sys.executable, "-c", PLANT_ANON, str(GIB)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    p.stdout.readline()
    held = sample(binary, "1 GiB held")
    p.stdin.close(); p.wait()
    time.sleep(2)
    after = sample(binary, "after exit")
    drop, back = before[0] - held[0], after[0] - held[0]
    print(f"1b: the reading fell {drop / GIB:.3f} GiB while 1 GiB was held (bar 0.8-1.2) and rose {back / GIB:.3f} GiB after (same band); "
          f"old reading fell {(before[1] - held[1]) / GIB:.3f} GiB; swap moved {(swap_used() - s0) / GIB:.3f} GiB")
    if abs(swap_used() - s0) > 256 << 20:
        print("VERDICT VOID (swap in use moved by more than 256 MB during the plant)")
        return 2
    ok = 0.8 * GIB <= drop <= 1.2 * GIB and 0.8 * GIB <= back <= 1.2 * GIB
    print(f"VERDICT {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def gate_1c(binary, giw):
    size = os.path.getsize(giw)
    before = sample(binary, "before")
    t0 = time.time()
    subprocess.run([sys.executable, "-c", PLANT_FILE, giw], check=True, capture_output=True)
    print(f"touched every page of {giw} ({size / GIB:.2f} GiB) in {time.time() - t0:.1f} s, then unmapped")
    after = sample(binary, "after the file is cached")
    new_drop, old_drop = before[0] - after[0], before[1] - after[1]
    print(f"1c: new reading fell {new_drop / GIB:.3f} GiB (bar < {0.3e9 / GIB:.3f}), old reading fell {old_drop / GIB:.3f} GiB (control)")
    if old_drop <= 0.3e9:
        print("VERDICT VOID (the old reading did not fall: the file was already cached; purge first)")
        return 2
    ok = new_drop < 0.3e9
    print(f"VERDICT {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hostmem", required=True)
    ap.add_argument("gate", choices=["1b", "1c"])
    ap.add_argument("--giw")
    a = ap.parse_args()
    return gate_1b(a.hostmem) if a.gate == "1b" else gate_1c(a.hostmem, a.giw)


if __name__ == "__main__":
    sys.exit(main())
