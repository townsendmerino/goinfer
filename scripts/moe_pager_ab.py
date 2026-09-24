#!/usr/bin/env python3
"""S5 pager A/B harness (docs/tasks/task-never-swap-2026-09.md, S5 "Measure").

One invocation = ONE server start + ONE streamed greedy request, so a mode is never measured
against a warm server the other mode did not also get. The driver (see --help / the S5 record)
alternates modes: mmap, pool, mmap, pool, ...

What one run records (JSON on stdout, samples CSV beside it):
  * banner facts: pager mode, resolved budget, S4's predicted tok/s
  * token timestamps -> TTFT and decode tok/s between token 1 and token N (footprint sampling is
    done on a thread, never on the request path)
  * `footprint <pid>` at tokens 1, 16 and 32: phys_footprint plus the dirty/clean/reclaimable
    split of the main categories
  * swap-used, RSS, cumulative page faults and page-ins (from top) every --sample-s seconds
  * GC CPU from GODEBUG=gctrace=1 stderr (last cumulative percentage, and the GC count)

SAFETY: this loads a checkpoint the machine cannot hold resident. It enforces its own out-of-process
kill switch (swap-used growth over its own baseline, and a wall-clock box) and expects the S3
in-process tripwire to be armed too (goinfer-serve arms it by default). It is a MEASUREMENT tool:
every number it emits is only valid when the model was read from LOCAL DISK (docs/benchmarks.md
"Model storage") — it refuses a --model under /Volumes or /srv/models.
"""
import argparse, atexit, json, os, re, signal, subprocess, sys, threading, time, urllib.request

_PROCS = []


def _reap():
    """Never leave a server (and its multi-GB mapping) behind, whatever killed this script."""
    for p in _PROCS:
        if p.poll() is None:
            p.kill()


atexit.register(_reap)
for _sig in (signal.SIGTERM, signal.SIGINT):
    signal.signal(_sig, lambda *_: sys.exit(1))

TEXT = ("The history of computing is a story of abstraction: each generation of engineers "
        "hid the complexity of the layer beneath it so that the next could build higher. "
        "Machine code gave way to assemblers, assemblers to compilers, compilers to managed "
        "runtimes, and runtimes to services reached over a network. At every step a cost was "
        "paid in performance and recovered in productivity, and at every step someone insisted "
        "the old way was better. The trade-off, in short, is that")


def swap_used_mb():
    out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
    m = re.search(r"used = ([0-9.]+)M", out)
    return float(m.group(1)) if m else -1.0


def ps_fields(pid):
    """RSS from ps; cumulative page FAULTS (minor+major) and PAGEINS (pages read from disk) from top —
    macOS ps prints '-' for majflt/minflt, top is the only stock source of both."""
    rss = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True).stdout.split()
    top = subprocess.run(["top", "-l", "1", "-pid", str(pid), "-stats", "pid,faults,pageins"],
                         capture_output=True, text=True).stdout.strip().splitlines()
    if not rss or not top:
        return None
    f = top[-1].split()
    if len(f) < 3 or not (f[1].isdigit() and f[2].isdigit()):
        return None
    return {"rss_kb": int(rss[0]), "faults": int(f[1]), "pageins": int(f[2])}


def footprint(pid):
    out = subprocess.run(["footprint", str(pid)], capture_output=True, text=True).stdout
    res = {}
    m = re.search(r"phys_footprint:\s*([0-9.]+)\s*(\w+)", out)
    if m:
        res["phys_footprint_mb"] = to_mb(m.group(1), m.group(2))
    cats = {}
    for line in out.splitlines():
        m = re.match(r"\s*([0-9.]+ \w+|0 B)\s+([0-9.]+ \w+|0 B)\s+([0-9.]+ \w+|0 B)\s+\d+\s+(.+)$", line)
        if m:
            d, c, r, name = m.groups()
            cats[name.strip()] = {"dirty_mb": pstr(d), "clean_mb": pstr(c), "reclaimable_mb": pstr(r)}
    keep = {k: v for k, v in cats.items() if k in ("TOTAL", "mapped file", "untagged (VM_ALLOCATE)",
                                                   "MALLOC_LARGE", "MALLOC_SMALL", "IOAccelerator (graphics)")}
    res["categories"] = keep
    return res


def pstr(s):
    if s.strip() == "0 B":
        return 0.0
    n, u = s.split()
    return to_mb(n, u)


def to_mb(n, u):
    n = float(n)
    return n * {"B": 1 / 1048576, "KB": 1 / 1024, "MB": 1.0, "GB": 1024.0}.get(u.upper(), 1.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--binary", required=True, help="goinfer-serve binary (CPU build)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", choices=["mmap", "pool"], default="", help="CPU expert pager mode; omit for a non-CPU backend")
    ap.add_argument("--backend", default="cpu")
    ap.add_argument("--extra", default="", help="extra server flags, e.g. '-moe-cache-experts -moe-cache-slots 8'")
    ap.add_argument("--no-stream-weights", action="store_true", help="omit -stream-weights (e.g. a Metal resident load)")
    ap.add_argument("--budget-gb", type=float, default=0.0, help="-weight-cache; 0 = auto (S4's live figure)")
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--prompt-repeat", type=int, default=1, help="repeat the ~90-word prompt to set depth")
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--port", type=int, default=18097)
    ap.add_argument("--label", default="run")
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--gomemlimit", default="", help="'off' sets GOMEMLIMIT=off; empty leaves it to the server")
    ap.add_argument("--kill-swap-mb", type=float, default=1024, help="kill if swap-used exceeds baseline by this")
    ap.add_argument("--load-box-s", type=int, default=900)
    ap.add_argument("--decode-box-s", type=int, default=900)
    ap.add_argument("--sample-s", type=float, default=2.0)
    ap.add_argument("--load-only", action="store_true", help="start, print the banner facts, stop (no request)")
    a = ap.parse_args()

    if a.model.startswith(("/Volumes", "/srv/models")):
        sys.exit("refusing: model path is on the archive/network mount — numbers from it are void (docs/benchmarks.md)")
    os.makedirs(a.outdir, exist_ok=True)
    base = os.path.join(a.outdir, a.label)
    swap0 = swap_used_mb()
    env = dict(os.environ, GODEBUG="gctrace=1")
    env.pop("GOINFER_MOE_PREAD_CPU", None)  # -moe-pager wins anyway (applyMoEPagerEnv); keep the env clean
    if a.gomemlimit:
        env["GOMEMLIMIT"] = a.gomemlimit
    cmd = [a.binary, "-addr", f"127.0.0.1:{a.port}", "-model", a.model, "-backend", a.backend, "-ctx", str(a.ctx)]
    if not a.no_stream_weights:
        cmd += ["-stream-weights"]
    if a.mode:
        cmd += ["-moe-pager", a.mode]
    cmd += a.extra.split()
    if a.budget_gb > 0:
        cmd += ["-weight-cache", str(a.budget_gb)]
    errf = open(base + ".server.log", "w")
    t_start = time.time()
    proc = subprocess.Popen(cmd, stdout=errf, stderr=subprocess.STDOUT, env=env)
    _PROCS.append(proc)
    result = {"label": a.label, "mode": a.mode, "model": a.model, "cmd": cmd, "swap_baseline_mb": swap0,
              "started": time.strftime("%H:%M:%S"), "gomemlimit": a.gomemlimit or "(server default)"}
    samples, killed, stop = [], [None], threading.Event()

    def sampler():
        with open(base + ".samples.csv", "w") as f:
            f.write("t_s,swap_mb,swap_delta_mb,rss_kb,faults,pageins\n")
            while not stop.is_set() and proc.poll() is None:
                sw, pf = swap_used_mb(), ps_fields(proc.pid)
                t = time.time() - t_start
                if pf:
                    row = (round(t, 1), sw, round(sw - swap0, 1), pf["rss_kb"], pf["faults"], pf["pageins"])
                    samples.append(row)
                    f.write(",".join(map(str, row)) + "\n"); f.flush()
                if sw - swap0 > a.kill_swap_mb:
                    killed[0] = f"swap delta {sw - swap0:.0f}MB > {a.kill_swap_mb:.0f}MB at t+{t:.0f}s"
                    proc.send_signal(signal.SIGKILL)
                    return
                stop.wait(a.sample_s)

    th = threading.Thread(target=sampler, daemon=True); th.start()

    def finish(reason=None):
        stop.set()
        # A server that is already gone when the harness finishes died on its own (or was killed by
        # something else): record HOW — returncode -9 is SIGKILL (jetsam or a watcher), -10/-11 a bus/segv
        # fault, 134 an abort — which the request error alone ("RemoteDisconnected") never says.
        result["server_exit_before_finish"] = proc.poll()
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(15)
            except subprocess.TimeoutExpired:
                proc.kill()
        th.join(5); errf.close()
        log = open(base + ".server.log").read()
        gcs = re.findall(r"^gc (\d+) @([0-9.]+)s (\d+)%", log, re.M)
        result["gc"] = {"count": int(gcs[-1][0]) if gcs else 0, "last_cum_gc_cpu_pct": int(gcs[-1][2]) if gcs else None}
        result["killed"] = killed[0]; result["ended_reason"] = reason
        if samples:
            result["max_rss_gb"] = round(max(s[3] for s in samples) / 1048576, 2)
            result["max_swap_delta_mb"] = max(s[2] for s in samples)
        json.dump(result, sys.stdout, indent=1); print()

    # -- wait for the banner
    banner = None
    next_fp, load_fps = time.time() + 6, []
    result["footprints_during_load"] = load_fps
    while time.time() - t_start < a.load_box_s:
        if proc.poll() is not None:
            finish("server exited during load"); return
        if time.time() >= next_fp:  # the split at peak, in case the run is killed before it finishes loading
            fpx = footprint(proc.pid)
            load_fps.append({"t_s": round(time.time() - t_start, 1), "swap_mb": swap_used_mb(),
                             "ps": ps_fields(proc.pid), "footprint": fpx})
            next_fp = time.time() + 6
        log = open(base + ".server.log").read()
        if "goinfer serving" in log:
            banner = log; break
        time.sleep(2)
    if banner is None:
        finish("load box exceeded"); return
    result["load_s"] = round(time.time() - t_start, 1)
    m = re.search(r"decode path: ([^\n]+)", banner)
    if m:
        result["decode_path"] = m.group(1).strip()
    fp_loaded = {"swap_mb": swap_used_mb(), "ps": ps_fields(proc.pid), "footprint": footprint(proc.pid)}
    result["footprint_after_load"] = fp_loaded
    m = re.search(r"expert paging \((\w+)\): (\d+) experts, ([0-9.]+) GB total, ([0-9.]+) GB budget", banner)
    if m:
        result["banner"] = {"pager": m.group(1), "experts": int(m.group(2)),
                            "total_gb": float(m.group(3)), "budget_gb": float(m.group(4))}
    m = re.search(r"predicted paged-MoE decode rate ~([0-9.]+) tok/s", banner)
    if m:
        result["predicted_tok_s"] = float(m.group(1))
    if a.load_only:
        finish("load-only"); return

    # -- one streamed greedy request
    prompt = TEXT * a.prompt_repeat
    mm = re.search(r"goinfer serving on \S+ \[\w+:\[([^\]]+)\]", banner)
    served = mm.group(1) if mm else os.path.basename(a.model)
    result["served_name"] = served
    body = json.dumps({"model": served, "prompt": prompt, "max_tokens": a.max_tokens,
                       "temperature": 0, "stream": True}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{a.port}/v1/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    snaps, ts, text, raw_head = {}, [], [], []
    fp_threads = []
    pf0 = ps_fields(proc.pid); t_req = time.time()

    def snap(n):
        snaps[n] = {"t_s": round(time.time() - t_req, 1), "swap_mb": swap_used_mb(),
                    "ps": ps_fields(proc.pid), "footprint": footprint(proc.pid)}

    try:
        with urllib.request.urlopen(req, timeout=a.decode_box_s) as r:
            result["http_status"] = r.status
            for raw in r:
                line = raw.decode().strip()
                if len(raw_head) < 5:
                    raw_head.append(line[:300])
                if not line.startswith("data:") or line == "data: [DONE]":
                    continue
                try:
                    ch = json.loads(line[5:])
                except ValueError:
                    continue
                txt = ch.get("choices", [{}])[0].get("text", "")
                if txt == "" and ch.get("choices", [{}])[0].get("finish_reason"):
                    continue
                ts.append(time.time() - t_req); text.append(txt)
                if len(ts) in (1, 16, 32):
                    t = threading.Thread(target=snap, args=(len(ts),), daemon=True); t.start(); fp_threads.append(t)
                if killed[0]:
                    break
                if time.time() - t_req > a.decode_box_s:
                    result["decode_box"] = "exceeded"; break
    except Exception as e:  # noqa: BLE001 — record, do not hide
        result["request_error"] = repr(e)
    for t in fp_threads:
        t.join(30)
    pf1 = ps_fields(proc.pid)
    result["tokens"] = len(ts)
    result["raw_head"] = raw_head
    result["text"] = "".join(text)
    if ts:
        result["ttft_s"] = round(ts[0], 2)
        if len(ts) > 1:
            result["decode_tok_s_1_to_N"] = round((len(ts) - 1) / (ts[-1] - ts[0]), 4)
        result["token_times_s"] = [round(x, 2) for x in ts]
    result["prompt_words"] = len(prompt.split())
    result["snapshots"] = {str(k): v for k, v in sorted(snaps.items())}
    if pf0 and pf1:
        result["faults_during_request"] = pf1["faults"] - pf0["faults"]
        result["pageins_during_request"] = pf1["pageins"] - pf0["pageins"]
    finish("done")


if __name__ == "__main__":
    main()
