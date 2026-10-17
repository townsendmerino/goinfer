#!/usr/bin/env python3
"""P28 step 2 (goinfer docs/queue-performance.md, P28; owner 2026-10-17): where Gemma 4 E2B's prefill time goes on Metal.
E2B cold at 4,096, 6,144 and 8,192 prompt tokens, each on a FRESH server (so each length has its own lifetime peak footprint
and nothing is reused). Per length: vm_stat's swap and compression counters and vm.swapusage before and after the request,
and the server's lifetime peak physical footprint (proc_pid_rusage, ri_lifetime_max_phys_footprint). The pinned
serve-metal at 784e7c82, -ctx 16384, --quant int4, as step 1. Run it under the timing lock:

    python3 ~/tmcode/goinfer/scripts/timing_lock.py run --label p28b -- python3 -u run-p28b.py
"""
import ctypes, json, os, re, subprocess, sys, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REV = "784e7c82"
SERVE = os.path.join(HERE, f"serve-metal-{REV}")
MODEL = os.path.expanduser("~/models/gemma-4-e2b-gguf/gemma-4-E2B_q4_0-it.gguf")
LENGTHS = [4096, 6144, 8192]
PORT = 18942
OUT = os.path.join(HERE, "results-step2-" + time.strftime("%Y-%m-%d-%H%M"))
VM_KEYS = ["Pageouts", "Swapins", "Swapouts", "Compressions", "Decompressions", "Pages occupied by compressor",
           "Pages stored in compressor", "Pages free"]


class RUsageV4(ctypes.Structure):  # <libproc.h> rusage_info_v4, through ri_lifetime_max_phys_footprint
    _fields_ = [("ri_uuid", ctypes.c_uint8 * 16)] + [(n, ctypes.c_uint64) for n in (
        "ri_user_time", "ri_system_time", "ri_pkg_idle_wkups", "ri_interrupt_wkups", "ri_pageins", "ri_wired_size",
        "ri_resident_size", "ri_phys_footprint", "ri_proc_start_abstime", "ri_proc_exit_abstime", "ri_child_user_time",
        "ri_child_system_time", "ri_child_pkg_idle_wkups", "ri_child_interrupt_wkups", "ri_child_pageins",
        "ri_child_elapsed_abstime", "ri_diskio_bytesread", "ri_diskio_byteswritten", "ri_cpu_time_qos_default",
        "ri_cpu_time_qos_maintenance", "ri_cpu_time_qos_background", "ri_cpu_time_qos_utility", "ri_cpu_time_qos_legacy",
        "ri_cpu_time_qos_user_initiated", "ri_cpu_time_qos_user_interactive", "ri_billed_system_time",
        "ri_serviced_system_time", "ri_logical_writes", "ri_lifetime_max_phys_footprint")] + [("pad", ctypes.c_uint64 * 32)]


def footprint(pid):
    lib = ctypes.CDLL("/usr/lib/libproc.dylib")
    ru = RUsageV4()
    if lib.proc_pid_rusage(pid, 4, ctypes.byref(ru)) != 0:
        return None, None
    return ru.ri_phys_footprint, ru.ri_lifetime_max_phys_footprint


def vmstat():
    txt = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    page = int(re.search(r"page size of (\d+)", txt).group(1))
    out = {k: int(m.group(1)) for k in VM_KEYS if (m := re.search(rf'^"?{re.escape(k)}"?:\s+(\d+)', txt, re.M))}
    out["page"] = page
    out["swapusage"] = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout.strip()
    return out


def ts():
    return time.strftime("%H:%M:%S %Z") + " / " + time.strftime("%H:%M:%S UTC", time.gmtime())


def post(path, body, timeout=1800):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def messages(lead, n):
    vocab = "the of and to in is was for on that with as by at from it an be this which or are were has had".split()
    words = [vocab[(i * 7 + len(lead)) % len(vocab)] + str(i % 97) for i in range(n)]
    return [{"role": "user", "content": f"{lead} " + " ".join(words) + "\nSummarize the text above in one word."}]


def fit(lead, target):
    count = lambda n: post("/v1/messages/count_tokens", {"model": "bench", "messages": messages(lead, n)})["input_tokens"]
    lo, hi = 1, target
    while lo < hi:
        mid = (lo + hi) // 2
        lo, hi = (mid + 1, hi) if count(mid) < target else (lo, mid)
    return messages(lead, lo), count(lo)


def main():
    if not os.access(SERVE, os.X_OK) or not os.path.exists(MODEL):
        sys.exit("missing the pinned server or the model")
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for n in LENGTHS:
        log = os.path.join(OUT, f"serve-{n}.log")
        print(f"=== {ts()} START {n}", flush=True)
        with open(log, "w") as lf:
            proc = subprocess.Popen([SERVE, "-model", f"bench={MODEL}", "-backend", "metal", "-quant", "int4", "-ctx", "16384",
                                     "-addr", f"127.0.0.1:{PORT}", "-log-requests"], stdout=lf, stderr=subprocess.STDOUT)
        try:
            t0 = time.time()
            while "goinfer serving on" not in open(log).read():
                if proc.poll() is not None or time.time() - t0 > 600:
                    sys.exit(f"{n}: the server did not come up; see {log}")
                time.sleep(0.5)
            banner = open(log).read()
            pp = (re.search(r"^\s*prefill path: (.*)$", banner, re.M) or [None, None])[1]
            post("/v1/chat/completions", {"model": "bench", "messages": messages("warmup", 8), "max_tokens": 1, "temperature": 0})
            msgs, got = fit(f"c{n}x", n)
            fp0, _ = footprint(proc.pid)
            before = vmstat()
            t1 = time.time()
            r = post("/v1/chat/completions", {"model": "bench", "messages": msgs, "max_tokens": 1, "temperature": 0})
            wall = time.time() - t1
            after = vmstat()
            fp1, peak = footprint(proc.pid)
            u = r.get("usage") or {}
            delta = {k: after[k] - before[k] for k in VM_KEYS if k in before and k in after}
            row = {"target": n, "prompt_tokens": u.get("prompt_tokens"), "reused": u.get("prefill_reused_tokens"),
                   "wall_s": round(wall, 3), "tok_s": round((u.get("prompt_tokens") or got) / wall, 1), "prefill_path_banner": pp,
                   "footprint_before_mb": fp0 and round(fp0 / 2**20), "footprint_after_mb": fp1 and round(fp1 / 2**20),
                   "peak_footprint_mb": peak and round(peak / 2**20), "vm_delta_pages": delta,
                   "swap_before": before["swapusage"], "swap_after": after["swapusage"], "page_bytes": before["page"]}
            rows.append(row)
            print(f"  {ts()} {row['prompt_tokens']} tokens: {wall:.1f} s, {row['tok_s']} tok/s, reused {row['reused']}; peak footprint "
                  f"{row['peak_footprint_mb']} MB; vm deltas {delta}; swap {before['swapusage']} -> {after['swapusage']}", flush=True)
        finally:
            proc.terminate()
            try:
                proc.wait(30)
            except subprocess.TimeoutExpired:
                proc.kill()
        print(f"=== {ts()} END {n}", flush=True)
    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump({"rev": REV, "host": os.uname().nodename, "model": MODEL, "rows": rows}, f, indent=2)
    print(f"records: {OUT}")


if __name__ == "__main__":
    main()
