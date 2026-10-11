#!/usr/bin/env python3
"""gcl_night.py: the timed gates of docs/tasks/task-giw-cold-load-2026-10.md (G-CL2, G-CL3, G-CL4) on nobara.

Two pinned CPU `serve` binaries, old and new. A start is: the bundle evicted (or read into the page cache, for a warm
start), the server started, one greedy 24-token reply taken, the server stopped by PID. Recorded per start: seconds from
process start to serving and to the finished reply, the bytes the process had read from disk when it began serving, how
much of the file was in the page cache then, whether the load announced a read-first, and the reply. Pairs alternate
their order. The bars are in the task doc; this script prints each cell's figures and verdict and writes JSON lines.
"""
import argparse, ctypes, json, mmap, os, statistics, subprocess, sys, time, urllib.request

T0 = time.time()
PAGE = os.sysconf("SC_PAGESIZE")
libc = ctypes.CDLL("libc.so.6", use_errno=True)
libc.mmap.restype = ctypes.c_void_p
libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_ubyte)]
libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
PROMPT = "List the first five prime numbers and say in one sentence why 1 is not among them."


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')} +{time.time() - T0:5.0f}s] {msg}", flush=True)


def resident_fraction(path):
    size = os.path.getsize(path)
    fd = os.open(path, os.O_RDONLY)
    p = libc.mmap(None, size, mmap.PROT_READ, mmap.MAP_SHARED, fd, 0)
    os.close(fd)
    n = (size + PAGE - 1) // PAGE
    vec = (ctypes.c_ubyte * n)()
    libc.mincore(p, size, vec)
    libc.munmap(p, size)
    return sum(b & 1 for b in vec) / n


def evict(path):
    fd = os.open(path, os.O_RDONLY)
    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    os.close(fd)


def warm(path):
    with open(path, "rb") as f:
        while f.read(1 << 24):
            pass


def read_bytes(pid):
    try:
        for line in open(f"/proc/{pid}/io"):
            if line.startswith("read_bytes:"):
                return int(line.split()[1])
    except OSError:
        pass
    return -1


def start(binary, model, port, logpath, state, extra, limit_gb):
    if state == "cold":
        evict(model)
    else:
        warm(model)
    before = resident_fraction(model)
    cmd = [binary, "-model", model, "-backend", "cpu", "-addr", f"127.0.0.1:{port}"] + extra
    if limit_gb:
        cmd = ["systemd-run", "--user", "--scope", "-q", "-p", f"MemoryMax={limit_gb}G", "-p", "MemorySwapMax=0"] + cmd
    out = open(logpath, "w")
    t0 = time.time()
    proc = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT)
    ev = {"resident_before": round(before, 4)}
    try:
        while True:
            if "goinfer serving on" in open(logpath).read():
                break
            if proc.poll() is not None:
                ev["error"] = "the server exited before it was serving: " + open(logpath).read()[-300:]
                return ev
            if time.time() - t0 > 900:
                ev["error"] = "not serving after 900 s"
                return ev
            time.sleep(0.05)
        ev["serving_s"] = round(time.time() - t0, 2)
        ev["read_bytes_at_serving"] = read_bytes(proc.pid)
        ev["resident_at_serving"] = round(resident_fraction(model), 4)
        name = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=30))["data"][0]["id"]
        body = json.dumps({"model": name, "max_tokens": 24, "temperature": 0, "messages": [{"role": "user", "content": PROMPT}]}).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", body, {"content-type": "application/json"})
        reply = json.load(urllib.request.urlopen(req, timeout=1800))
        ev["first_reply_s"] = round(time.time() - t0, 2)
        ev["reply"] = reply["choices"][0]["message"]["content"]
        ev["read_first_announced"] = "not in memory" in open(logpath).read()
    except Exception as e:  # recorded; the run goes on
        ev["error"] = repr(e)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
        out.close()
    return ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin-dir", required=True, help="holds serve-old and serve-new")
    ap.add_argument("--log-dir", required=True)
    ap.add_argument("--record", required=True)
    ap.add_argument("--est-min", type=float, default=0)
    ap.add_argument("cells", nargs="+", help="name|model|state(cold,warm)|pairs|arms(old:new or old:old)|extra args|memory limit GB or 0")
    a = ap.parse_args()
    os.makedirs(a.log_dir, exist_ok=True)
    rec = open(a.record, "a")

    def emit(ev):
        rec.write(json.dumps({"at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **ev}) + "\n")
        rec.flush()

    log(f"started {time.strftime('%Y-%m-%d %H:%M:%S %Z')} on {os.uname().nodename}")
    if a.est_min:
        log(f"estimate {a.est_min:.0f} min, expect done ~{time.strftime('%H:%M', time.localtime(T0 + a.est_min * 60))}")
    try:
        log("binaries: " + open(os.path.join(a.bin_dir, "BUILD.txt")).read().strip().replace("\n", " | "))
    except OSError:
        pass
    port, failed = 18900, False
    for ci, spec in enumerate(a.cells):
        name, model, state, pairs, arms, extra, limit = spec.split("|")
        model, pairs, limit = os.path.expanduser(model), int(pairs), int(limit)
        first, second = arms.split(":")
        extra = extra.split() if extra.strip() else []
        size = os.path.getsize(model)
        log(f"cell {name} ({ci + 1}/{len(a.cells)}): {os.path.basename(model)}, {size / 1e9:.1f} GB, {state}, {pairs} pair(s) of {first} and {second}" + (f", in a {limit} GB memory scope" if limit else "") + (f", args {' '.join(extra)}" if extra else ""))
        runs = {"a": [], "b": []}
        for k in range(pairs):
            order = [("a", first), ("b", second)] if k % 2 == 0 else [("b", second), ("a", first)]
            for slot, arm in order:
                port += 1
                ev = start(os.path.join(a.bin_dir, f"serve-{arm}"), model, port, os.path.join(a.log_dir, f"{name}-{slot}-{arm}-{k + 1}.log"), state, extra, limit)
                ev.update({"event": "start", "cell": name, "pair": k + 1, "slot": slot, "arm": arm, "file_bytes": size})
                emit(ev)
                runs[slot].append(ev)
                if "error" in ev:
                    failed = True
                    log(f"  pair {k + 1} {arm}: ERROR {ev['error'][:200]}")
                else:
                    log(f"  pair {k + 1} {arm}: serving at {ev['serving_s']:.1f}s, first reply done at {ev['first_reply_s']:.1f}s, read {ev['read_bytes_at_serving'] / 1e9:.2f} GB by serving, {100 * ev['resident_at_serving']:.0f}% resident, read-first {ev['read_first_announced']}")
        ok = [(x, y) for x, y in zip(runs["a"], runs["b"]) if "error" not in x and "error" not in y]
        line = {"event": "summary", "cell": name, "state": state, "arms": arms, "pairs": len(ok)}
        if ok:
            line["first_reply_b_over_a"] = [round(y["first_reply_s"] / x["first_reply_s"], 4) for x, y in ok]
            line["first_reply_b_over_a_median"] = round(statistics.median(line["first_reply_b_over_a"]), 4)
            line["serving_b_minus_a_s"] = [round(y["serving_s"] - x["serving_s"], 3) for x, y in ok]
            line["serving_b_minus_a_s_median"] = round(statistics.median(line["serving_b_minus_a_s"]), 3)
            line["a_first_reply_s_median"] = round(statistics.median(x["first_reply_s"] for x, _ in ok), 2)
            line["b_first_reply_s_median"] = round(statistics.median(y["first_reply_s"] for _, y in ok), 2)
            line["b_read_fraction_at_serving_max"] = round(max(y["read_bytes_at_serving"] for _, y in ok) / size, 4)
            line["a_read_fraction_at_serving_max"] = round(max(x["read_bytes_at_serving"] for x, _ in ok) / size, 4)
            line["replies_identical"] = len({e["reply"] for pair in ok for e in pair}) == 1
            if not line["replies_identical"]:
                failed = True
        emit(line)
        log("  " + json.dumps({k: v for k, v in line.items() if k != "event"}))
    emit({"event": "end", "failed": failed, "wall_s": round(time.time() - T0, 1)})
    log(f"finished in {(time.time() - T0) / 60:.1f} min: " + ("FAILED (a start failed, or the replies of one cell differ)" if failed else "every start ran and each cell's replies are the same"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
