#!/usr/bin/env python3
"""MEM4(b) on CUDA (README.md here): a vision tower attached on the device, with and without serve dropping its host blocks (`releaseTowerBlocks`). One server per arm, the release arm first.
Each server is started at serve's defaults except --backend cuda and --ctx 4096, with the Qwen3-VL-2B checkpoint as both the model and the vision tower. /proc/<pid>/status is read after
load, then after each of two image turns (two different 1920x1080 PNGs, so the feature cache cannot answer the second) at the turn's end and 10 s later; the peak is sampled every 250 ms.

  python3 mem4b_cuda.py --serve-release <bin> --serve-norelease <bin> --model <dir> --out <dir>

Reads, grades by the README's rule. Every size in the JSON and the printout is MiB (1 MiB = 1,048,576 bytes), keys end in _mib; GiB is MiB / 1024.

VOID (exit 2), enforced: the one-minute load average above LOAD_VOID (8.0) when an arm starts; a server that does not start; a log that does not show the vision tower on `cuda`
(TOWER_RE); an image turn that is not answered with 200. A void arm is a void run.
Known limits: one run per arm; /proc is read from outside the process, so a peak under 250 ms can be missed; RssAnon is the process's anonymous pages (the Go heap and the
driver's pinned buffers alike), not the Go heap in use alone.
"""
import argparse, base64, json, os, re, signal, struct, subprocess, sys, threading, time, urllib.request, zlib

PORT = 18241
BASE = f"http://127.0.0.1:{PORT}"
KiB, MiB = 1 << 10, 1 << 20
LOAD_VOID = 8.0
TOWER_RE = re.compile(r"vision.*(cuda|CUDA)|(cuda|CUDA).*vision|tower.*(cuda|CUDA)|(cuda|CUDA).*tower")
FIELDS = ("RssAnon", "RssFile", "RssShmem", "VmHWM", "VmSwap")


class Void(Exception):
    pass


def png(w, h, seed):
    """A w x h RGB PNG of smooth gradients and a grid whose position depends on seed; deterministic, written with zlib alone."""
    rows = bytearray()
    for y in range(h):
        rows.append(0)
        for x in range(w):
            g = 255 if ((x + seed * 37) % 120 < 3 or (y + seed * 53) % 120 < 3) else 0
            rows += bytes(((x * 255) // w, (y * 255) // h, g))
    chunk = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(bytes(rows), 6)) + chunk(b"IEND", b"")


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


def arm(name, binary, model, out):
    load = float(open("/proc/loadavg").read().split()[0])
    if load > LOAD_VOID:
        raise Void(f"{name}: one-minute load average {load:.2f} over {LOAD_VOID} at the arm's start")
    logpath = os.path.join(out, f"serve-{name}.log")
    log = open(logpath, "w")
    p = subprocess.Popen([binary, "--model", model, "--vision", model, "--backend", "cuda", "--ctx", "4096", "--addr", f"127.0.0.1:{PORT}"],
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
    rec = {"arm": name, "load_avg_1min": load}
    t0 = time.time()
    try:
        while True:
            if p.poll() is not None:
                raise Void(f"{name}: the server exited {p.returncode} (serve-{name}.log)")
            if time.time() - t0 > 600:
                raise Void(f"{name}: the server did not listen in 600 s")
            try:
                urllib.request.urlopen(BASE + "/v1/models", timeout=1).read()
                break
            except Exception:
                time.sleep(0.5)
        rec["load_s"] = round(time.time() - t0, 1)
        time.sleep(2)
        rec["after_load"] = status_mib(p.pid)
        mid = json.load(urllib.request.urlopen(BASE + "/v1/models", timeout=5))["data"][0]["id"]
        rec["turns"] = []
        for i in (1, 2):
            img = base64.b64encode(png(1920, 1080, i)).decode()
            body = {"model": mid, "max_tokens": 8, "temperature": 0, "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + img}}, {"type": "text", "text": "In one word, what is in this picture?"}]}]}
            t1 = time.time()
            r = urllib.request.urlopen(urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=1800)
            if r.status != 200:
                raise Void(f"{name}: image turn {i} answered {r.status}")
            j = json.load(r)
            end = status_mib(p.pid)
            time.sleep(10)
            rec["turns"].append({"turn": i, "secs": round(time.time() - t1 - 10, 1), "at_end": end, "plus_10s": status_mib(p.pid), "text": j["choices"][0]["message"]["content"]})
        rec["tower_lines"] = [l.strip() for l in open(logpath) if TOWER_RE.search(l)][:6]
        if not rec["tower_lines"]:
            raise Void(f"{name}: the log shows no vision tower on cuda (TOWER_RE); see serve-{name}.log")
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
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--serve-release", required=True)
    ap.add_argument("--serve-norelease", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    try:
        recs = [arm("release", a.serve_release, a.model, a.out), arm("norelease", a.serve_norelease, a.model, a.out)]
    except Void as e:
        print(f"VERDICT: VOID ({e})")
        sys.exit(2)
    json.dump(recs, open(os.path.join(a.out, "readings.json"), "w"), indent=1)
    for r in recs:
        print(f"{r['arm']}: load {r['load_s']} s (load avg {r['load_avg_1min']:.2f}); after load RssAnon {r['after_load']['RssAnon_mib']:,.1f} MiB; peak {r['peak_sampled']['RssAnon_mib']:,.1f} MiB")
        for t in r["turns"]:
            print(f"  turn {t['turn']}: {t['secs']} s; RssAnon at end {t['at_end']['RssAnon_mib']:,.1f} MiB, +10 s {t['plus_10s']['RssAnon_mib']:,.1f} MiB ({t['plus_10s']['RssAnon_mib'] / 1024:.2f} GiB); text {t['text']!r}")
        print(f"  tower lines: {r['tower_lines'][:2]}")
    fall = recs[1]["turns"][-1]["plus_10s"]["RssAnon_mib"] - recs[0]["turns"][-1]["plus_10s"]["RssAnon_mib"]
    same = [t["text"] for t in recs[0]["turns"]] == [t["text"] for t in recs[1]["turns"]]
    print(f"RssAnon 10 s after the second image turn, no-release minus release: {fall:,.1f} MiB ({fall / 1024:.2f} GiB); the replies {'equal' if same else 'DIFFER'}")


if __name__ == "__main__":
    main()
