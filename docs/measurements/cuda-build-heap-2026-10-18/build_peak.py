#!/usr/bin/env python3
"""The host memory of a resident CUDA build, sampled from process start (cuda-build-heap-2026-10-18/README.md).

One fresh `serve -backend cuda` per cell, GODEBUG=gctrace=1 so the Go collector reports the heap at each cycle. The process is read every 50 ms from the moment
it is spawned, through the load and 3 s past readiness: VmRSS, RssAnon (heap and anonymous buffers), RssFile (mapped checkpoint / sidecar), VmHWM, VmSwap.
Reported per cell: peak RssAnon and when, peak VmRSS and VmHWM, RssAnon / RssFile once ready, and the heap sizes the collector printed (largest "before GC"
figure and largest live figure after marking).

  python3 build_peak.py --serve <binary> --out <dir> --cell name:model_path:extra,args ...
"""
import argparse, json, os, re, signal, subprocess, threading, time, urllib.request

MiB = 1 << 20
PORT = 18198


def status(pid):
    d = {}
    try:
        for line in open(f"/proc/{pid}/status"):
            k, _, v = line.partition(":")
            if k in ("VmRSS", "RssAnon", "RssFile", "VmHWM", "VmSwap"):
                d[k] = int(v.split()[0]) * 1024
    except OSError:
        pass
    return d


def ready():
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=1).read()
        return True
    except Exception:
        return False


def cell(serve, name, model, extra, out):
    env = dict(os.environ, GODEBUG="gctrace=1")
    log = open(os.path.join(out, f"serve-{name}.log"), "w")
    t0 = time.time()
    proc = subprocess.Popen([serve, "-model", f"bench={model}", "-backend", "cuda", "-addr", f"127.0.0.1:{PORT}", *extra], stdout=log, stderr=subprocess.STDOUT,
                            env=env, start_new_session=True)
    samples, t_ready = [], None
    try:
        while time.time() - t0 < 900:
            s = status(proc.pid)
            s["t"] = round(time.time() - t0, 2)
            samples.append(s)
            if proc.poll() is not None:
                break
            if t_ready is None and s["t"] > 1 and ready():
                t_ready = s["t"]
            if t_ready is not None and s["t"] > t_ready + 3:
                break
            time.sleep(0.05)
    finally:
        # The server may already have exited (a swap-guard abort, a crash): killing a process group that is gone must not lose the samples.
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=30)
        except ProcessLookupError:
            pass
        except Exception:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    log.close()
    rec = {"cell": name, "model": model, "extra": extra, "ready_s": t_ready, "samples": len(samples)}
    ok = [s for s in samples if "RssAnon" in s]
    if ok:
        pa = max(ok, key=lambda s: s["RssAnon"])
        pr = max(ok, key=lambda s: s["VmRSS"])
        rec.update(peak_anon_mib=round(pa["RssAnon"] / MiB), peak_anon_at_s=pa["t"], peak_rss_mib=round(pr["VmRSS"] / MiB), peak_rss_at_s=pr["t"],
                   hwm_mib=round(max(s["VmHWM"] for s in ok) / MiB), peak_swap_mib=round(max(s.get("VmSwap", 0) for s in ok) / MiB))
        last = ok[-1]
        rec.update(final_anon_mib=round(last["RssAnon"] / MiB), final_file_mib=round(last["RssFile"] / MiB))
    gc = [tuple(int(x) for x in m.groups()) for m in re.finditer(r"gc \d+ @[\d.]+s \d+%: .*?\s(\d+)->(\d+)->(\d+) MB", open(os.path.join(out, f"serve-{name}.log")).read())]
    if gc:
        rec.update(gc_cycles=len(gc), gc_max_before_mb=max(g[0] for g in gc), gc_max_live_mb=max(g[2] for g in gc))
    json.dump({"rec": rec, "samples": samples}, open(os.path.join(out, f"cell-{name}.json"), "w"))
    return rec


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--serve", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cell", action="append", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for c in a.cell:
        name, model, extra = (c.split(":", 2) + [""])[:3]
        print(f"# {time.strftime('%T')} {name}", flush=True)
        print(json.dumps(cell(a.serve, name, os.path.expanduser(model), [x for x in extra.split(",") if x], a.out)), flush=True)
