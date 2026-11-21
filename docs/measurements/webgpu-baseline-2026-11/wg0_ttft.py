#!/usr/bin/env python3
"""WG0's TTFT cells: two goinfer serve binaries (a native backend and WebGPU), A B B A, one server per cell.

bench_peer_prefill.py has no WebGPU arm and no A B B A order, so this drives its request helpers (ttft, uniq, the cache check's shape)
with its own arm list. Each cell: evict the checkpoint page cache, wait for the load to settle, take void_rule's start reading, start
the server, run the cache check (repeat/fresh) and n unique-prefix requests of max_tokens 1, stop the server, take the end reading.
A cell void on start or on swap re-runs in place, listed; three voids in a row stop the run.

  wg0_ttft.py out.json --arm cuda=<serve>:cuda --arm webgpu=<serve>:webgpu --models 0.5B,1.5B,7B --quads 3 --n 6 --ctx 4096
"""
import argparse, json, os, re, signal, statistics, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("WG0_REPO", os.path.expanduser("~/goinfer-bench/wg0-run/wt"))
sys.path.insert(0, os.path.join(REPO, "scripts"))
import bench_peer_prefill as bpp  # noqa: E402
import void_rule  # noqa: E402

PORT = 18298
PROMPT_KEY = {"0.5B": "0.5B:1024", "1.5B": "1.5B:1024", "7B": "1.5B:1024"}  # one Qwen2.5 tokenizer; 7B has no 1024 entry


def evict(roots):
    if not hasattr(os, "posix_fadvise"):
        return "no posix_fadvise on this OS: page cache not evicted"
    return subprocess.run([sys.executable, os.path.join(HERE, "evict_cache.py")] + roots,
                          capture_output=True, text=True).stdout.strip()


def settle(cap=1.0, max_s=600):
    """Waits up to max_s for the 1-minute load to fall under cap (WG0_SETTLE_LOAD, WG0_SETTLE_MAX_S override); returns (load, seconds
    waited), seconds None when it never fell. The void rule decides the cell either way."""
    cap = float(os.environ.get("WG0_SETTLE_LOAD", cap))
    max_s = int(os.environ.get("WG0_SETTLE_MAX_S", max_s))
    t0 = time.time()
    while True:
        la = os.getloadavg()[0]
        if la < cap:
            return la, round(time.time() - t0)
        if time.time() - t0 > max_s:
            return la, None
        time.sleep(15)


def run_cell(arm, binp, be, mk, n, ctx, logdir, label):
    path = bpp.MODELS[mk][0]
    base = bpp.PROMPTS[PROMPT_KEY[mk]]["text"]
    log = os.path.join(logdir, f"ttft-{label}-{arm}-{mk}.log")
    off = os.path.getsize(log) if os.path.exists(log) else 0
    argv = [binp, "-model", f"bench={path}", "-backend", be, "-addr", f"127.0.0.1:{PORT}", "-quant", "int4",
            "-ctx", str(ctx), "-require-backend"]
    url = f"http://127.0.0.1:{PORT}/v1/chat/completions"
    payload = lambda t: {"model": "bench", "stream": True, "max_tokens": 1, "temperature": 0, "seed": 1,
                         "stream_options": {"include_usage": True}, "messages": [{"role": "user", "content": t}]}
    with open(log, "ab") as lf:
        proc = subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=lf, preexec_fn=os.setsid)
    rec = {"argv": argv}
    try:
        if not bpp.wait_port(PORT, timeout=300):
            rec["error"] = "server did not come up"
            return rec
        f = lambda i: bpp.ttft(url, payload(bpp.uniq(i) + base), bpp.parse_openai)
        f(8000)
        fresh = [f(8100 + i)[0] for i in range(2)]
        rep = payload(bpp.uniq(8200) + base)
        bpp.ttft(url, rep, bpp.parse_openai)
        reps = [bpp.ttft(url, rep, bpp.parse_openai)[0] for _ in range(2)]
        rec["repeat_over_fresh"] = round(statistics.median(reps) / statistics.median(fresh), 3)
        f(9000)  # warm-up on its own prompt
        ts, pt = [], []
        for i in range(n):
            t, p = f(i)
            if t:
                ts.append(t)
            if p:
                pt.append(p)
        rec.update({"ttft_ms_all": [round(t * 1000, 2) for t in ts], "prompt_tokens": pt})
        if len(ts) == n:
            rec["ttft_ms_median"] = round(statistics.median(ts) * 1000, 2)
        else:
            rec["error"] = f"{len(ts)} of {n} requests timed"
    except Exception as e:  # recorded, the cell then counts as invalid
        rec["error"] = repr(e)
    finally:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=45)
        except Exception:
            proc.kill()
        time.sleep(2)
    with open(log, "rb") as lf:
        lf.seek(off)
        txt = lf.read().decode("utf-8", "replace")
    m = re.search(r"decode path:\s*(.*)", txt)
    rec["decode_path"] = m.group(1).strip() if m else None
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--arm", action="append", required=True, help="name=<serve binary>:<backend>; order is block 0's")
    ap.add_argument("--models", default="0.5B,1.5B,7B")
    ap.add_argument("--quads", type=int, default=3)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--evict", default=None, help="comma-separated files or directories (default: the cell's checkpoint)")
    a = ap.parse_args()
    arms = []
    for s in a.arm:
        name, rest = s.split("=", 1)
        b, be = rest.rsplit(":", 1)
        arms.append((name, b, be))
    logdir = os.path.join(os.path.dirname(os.path.abspath(a.out)), "serve-logs")
    os.makedirs(logdir, exist_ok=True)
    out = json.load(open(a.out)) if os.path.exists(a.out) else []
    done = {(r["model"], r["block"], r["arm"]) for r in out if r.get("valid")}
    for mk in a.models.split(","):
        need = os.path.getsize(bpp.MODELS[mk][0]) / 1e6  # lower bound: the checkpoint's size
        for q in range(a.quads):
            for b, order in ((2 * q, arms), (2 * q + 1, arms[::-1])):
                for name, binp, be in order:
                    if (mk, b, name) in done:
                        continue
                    voids = []
                    while True:
                        ev = evict(a.evict.split(",") if a.evict else [bpp.MODELS[mk][0]])
                        la, waited = settle()
                        start = void_rule.snapshot()
                        start["need_mb"] = need
                        why = void_rule.start_void(start, need)
                        sampler = void_rule.SwapSampler(base=start["swap_mb"]).start()
                        print(f"# {time.strftime('%T')} {mk} block {b} {name}: load {start['load1']} avail {start['available_mb']} MB "
                              f"(settled {waited}s) {ev}", flush=True)
                        rec = run_cell(name, binp, be, mk, a.n, a.ctx, logdir, "wg0")
                        res = sampler.stop()
                        end = void_rule.snapshot()
                        why = why or void_rule.end_void(start, end, res.get("peak_mb"), peak_required=True)
                        if not why:
                            break
                        voids.append(why)
                        print(f"#   VOID ({why}); re-running in place", flush=True)
                        if len(voids) >= 3:
                            sys.exit(f"STOPPED: three voids in a row at {mk} block {b} {name}")
                    res.pop("samples", None)
                    rec.update({"model": mk, "block": b, "arm": name, "backend": be, "binary": binp, "ctx": a.ctx,
                                "prompt_key": PROMPT_KEY[mk], "machine": {"start": start, "end": end, "swap": res},
                                "void_attempts": voids, "valid": "error" not in rec})
                    out = [r for r in out if not (r["model"] == mk and r["block"] == b and r["arm"] == name)] + [rec]
                    with open(a.out, "w") as f:
                        json.dump(out, f, indent=1)
                    print(json.dumps({k: rec.get(k) for k in ("model", "block", "arm", "ttft_ms_median", "repeat_over_fresh",
                                                              "decode_path", "error")}), flush=True)


if __name__ == "__main__":
    main()
