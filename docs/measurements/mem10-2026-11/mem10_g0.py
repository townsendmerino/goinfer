#!/usr/bin/env python3
"""MEM10-G0 (README.md here): the host footprint of the Gemma 4 26B-A4B on CUDA as `serve` loads it today. One server, started at serve's defaults except --backend cuda, --ctx 4096
and --moe-cache-experts; /proc/<pid>/status is sampled every 250 ms until the server answers /v1/models, read again when it listens, 5 s after, and after one 16-token greedy
completion; the largest mappings, and Rss summed by pathname, are read from /proc/<pid>/smaps once the server listens; the process's device memory comes from nvidia-smi.

  python3 mem10_g0.py --serve <bin> --model <gguf> --out <dir>

Reads, grades nothing. Every size in reading.json and in the printout is in MiB (1 MiB = 1,048,576 bytes), keys end in _mib; GiB is MiB / 1024 and is printed as such.

VOID (exit 2), enforced: the one-minute load average above LOAD_VOID (8.0) when the script starts the server, a server that does not start, or a server whose log does not say
`decode path: cuda-resident` (it ran on the CPU or declined). A void run is not a reading. Known limits: one run; /proc is read from outside the process, so the sampler can miss a
peak shorter than 250 ms; `nvidia-smi` lists the process's device memory only when it reports compute apps, and then device_mib is null (printed n/a).
"""
import argparse, json, os, signal, subprocess, sys, threading, time, urllib.request

PORT = 18223
BASE = f"http://127.0.0.1:{PORT}"
KiB, MiB, GiB = 1 << 10, 1 << 20, 1 << 30
LOAD_VOID = 8.0
FIELDS = ("RssAnon", "RssFile", "RssShmem", "VmHWM", "VmSwap", "VmLck")


class Void(Exception):
    pass


def status_mib(pid):
    d = {}
    try:
        for line in open(f"/proc/{pid}/status"):
            k, _, v = line.partition(":")
            if k in FIELDS:
                d[k + "_mib"] = round(int(v.split()[0]) * KiB / MiB, 1)
    except FileNotFoundError:
        pass
    return d


def device_mib(pid):
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    for ln in out.splitlines():
        p, _, m = ln.partition(",")
        if p.strip() == str(pid):
            return int(m.strip())
    return None


def big_mappings(pid, floor_mib=256):
    """The mappings whose Rss is at least floor_mib, with what /proc/<pid>/smaps says each is: the pathname (a file, [heap], a /memfd: or SYSV name for shared memory, or anonymous),
    its permissions, and its Rss, Anonymous and shared parts, in MiB."""
    out, cur = [], None
    for line in open(f"/proc/{pid}/smaps"):
        head = line.split()
        if len(head) >= 5 and "-" in head[0] and not line.startswith(("Rss", "Pss", "Size")) and head[0].replace("-", "").isalnum() and ":" not in head[0]:
            cur = {"range": head[0], "perms": head[1], "path": " ".join(head[5:]) or "(anonymous)"}
            out.append(cur)
        elif cur is not None and ":" in line:
            k, _, v = line.partition(":")
            if k in ("Rss", "Anonymous", "Shared_Clean", "Shared_Dirty", "Private_Clean", "Private_Dirty"):
                cur[k.lower() + "_mib"] = round(int(v.split()[0]) * KiB / MiB, 1)
    return sorted((m for m in out if m.get("rss_mib", 0) >= floor_mib), key=lambda m: -m["rss_mib"])


def rss_by_path(pid, top=8):
    """The same smaps, summed by pathname: one row per path (or "(anonymous)"), with the number of mappings, the total Rss, the Anonymous part and the permissions seen, in MiB.
    A region made of many mappings each under big_mappings' floor shows here."""
    rows = {}
    for m in big_mappings(pid, floor_mib=0):
        r = rows.setdefault(m["path"], {"path": m["path"], "mappings": 0, "rss_mib": 0.0, "anonymous_mib": 0.0, "perms": set()})
        r["mappings"] += 1
        r["rss_mib"] += m.get("rss_mib", 0)
        r["anonymous_mib"] += m.get("anonymous_mib", 0)
        r["perms"].add(m["perms"])
    out = sorted(rows.values(), key=lambda r: -r["rss_mib"])[:top]
    for r in out:
        r["rss_mib"], r["anonymous_mib"], r["perms"] = round(r["rss_mib"], 1), round(r["anonymous_mib"], 1), sorted(r["perms"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--serve", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    load0 = float(open("/proc/loadavg").read().split()[0])
    if load0 > LOAD_VOID:
        print(f"VERDICT: VOID (one-minute load average {load0:.2f} over {LOAD_VOID} at the start)")
        sys.exit(2)
    avail0_mib = [int(l.split()[1]) for l in open("/proc/meminfo") if l.startswith("MemAvailable")][0] * KiB // MiB
    logpath = os.path.join(a.out, "serve.log")
    log = open(logpath, "w")
    p = subprocess.Popen([a.serve, "--model", a.model, "--backend", "cuda", "--ctx", "4096", "--moe-cache-experts", "--addr", f"127.0.0.1:{PORT}"],
                         stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    peak, lock, stop = {}, threading.Lock(), threading.Event()

    def sample():
        while not stop.is_set():
            s = status_mib(p.pid)
            with lock:
                for k, v in s.items():
                    peak[k] = max(peak.get(k, 0), v)
            time.sleep(0.25)

    threading.Thread(target=sample, daemon=True).start()
    t0 = time.time()
    rec = {"load_avg_1min": load0, "mem_available_mib": avail0_mib}
    try:
        while True:
            if p.poll() is not None:
                raise Void(f"the server exited {p.returncode} (serve.log)")
            if time.time() - t0 > 900:
                raise Void("the server did not listen in 900 s")
            try:
                urllib.request.urlopen(BASE + "/v1/models", timeout=1).read()
                break
            except Exception:
                time.sleep(0.5)
        rec["load_s"] = round(time.time() - t0, 1)
        rec["decode_path"] = [l.strip() for l in open(logpath) if "decode path:" in l]
        if not any("cuda-resident" in l for l in rec["decode_path"]):
            raise Void(f"the server's log does not say `decode path: cuda-resident` (it says {rec['decode_path'] or 'nothing'})")
        rec["at_listen"] = status_mib(p.pid)
        rec["big_mappings_at_listen"] = big_mappings(p.pid)
        rec["rss_by_path_at_listen"] = rss_by_path(p.pid)
        time.sleep(5)
        rec["after_5s"] = status_mib(p.pid)
        rec["device_mib"] = device_mib(p.pid)
        mid = json.load(urllib.request.urlopen(BASE + "/v1/models", timeout=5))["data"][0]["id"]
        body = {"model": mid, "prompt": "The capital of France is", "max_tokens": 16, "temperature": 0}
        r = json.load(urllib.request.urlopen(urllib.request.Request(BASE + "/v1/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=300))
        rec["completion"] = r["choices"][0]["text"]
        rec["after_request"] = status_mib(p.pid)
    except Void as e:
        print(f"VERDICT: VOID ({e})")
        rec["void"] = str(e)
        rec["exit"] = 2
    finally:
        stop.set()
        time.sleep(0.3)
        with lock:
            rec["peak_sampled"] = dict(peak)
        try:
            os.killpg(p.pid, signal.SIGTERM)
            p.wait(timeout=60)
        except Exception:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except Exception:
                pass
    json.dump(rec, open(os.path.join(a.out, "reading.json"), "w"), indent=1)
    if rec.get("exit") == 2:
        sys.exit(2)
    for k in ("at_listen", "after_5s", "after_request", "peak_sampled"):
        print(f"{k}: " + ", ".join(f"{x[:-4]} {v:,.1f} MiB ({v / 1024:.2f} GiB)" for x, v in rec[k].items()))
    print("largest mappings at listen:")
    for m in rec["big_mappings_at_listen"][:6]:
        print(f"  {m['rss_mib']:>9,.1f} MiB rss ({m['rss_mib'] / 1024:.2f} GiB) anon {m.get('anonymous_mib', 0):,.1f}  {m['perms']}  {m['path']}")
    print("Rss summed by pathname at listen:")
    for r in rec["rss_by_path_at_listen"]:
        print(f"  {r['rss_mib']:>9,.1f} MiB ({r['rss_mib'] / 1024:.2f} GiB) in {r['mappings']} mapping(s), anon {r['anonymous_mib']:,.1f} MiB  {','.join(r['perms'])}  {r['path']}")
    dev = rec["device_mib"]
    print(f"load {rec['load_s']} s; device {'n/a' if dev is None else f'{dev:,} MiB ({dev / 1024:.2f} GiB)'}; load avg {rec['load_avg_1min']:.2f}; "
          f"{rec['mem_available_mib']:,} MiB available at start; {rec['decode_path']}; completion {rec['completion']!r}")


if __name__ == "__main__":
    main()
