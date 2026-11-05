#!/usr/bin/env python3
"""KV5 G2's prefill gate on Metal (docs/tasks/task-kv-architecture-2026-10.md, KV5's plan): KV4's ttft_ab.py with the backend Metal and G4 as Francis amended it.
Time to first token at K = 512 and K = 3,900, new serve against old, three alternating pairs, a fresh server per run at -ctx 4096, six distinct prompts per depth
(a unique prefix each, so no reuse can serve them), a warm-up request first. Client-timed to the first streamed content event (bench_peer_prefill.ttft).
The ratio is old/new per pair and its median over pairs: PASS at 0.97 or more, AMBIGUOUS from 0.96 to 0.97, FAIL under 0.96.
A run with any swap-out re-runs in place, listed, never averaged; three in a row stop. The harness's heavy-load check runs before each server.

  ttft_ab.py <old serve> <new serve> <out json>      env: TTFT_MODEL (1.5B or 7B), TTFT_SCRIPTS (the harness's directory)
"""
import json, os, signal, statistics, subprocess, sys, time

HERE = os.environ.get("TTFT_SCRIPTS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../scripts"))
sys.path.insert(0, HERE)
import bench_peer as bench  # noqa: E402
import bench_peer_prefill as bp  # noqa: E402

old, new, out = sys.argv[1:4]
MODELS = {"1.5B": os.path.expanduser("~/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"), "7B": os.path.expanduser("~/models/qwen2.5-7b-instruct-q4_k_m.gguf")}
MK = os.environ.get("TTFT_MODEL", "1.5B")
MODEL = MODELS[MK]
DEPTHS = (512, 3900)
PORT = bp.GPORT
NEED = bench.fit_need(MK)[0]
log = []


def serve_rss_kb(pid):
    r = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True)
    return int(r.stdout.strip() or 0)


def attempt(binary):
    i0, o0 = bench.swap_io_mb()
    p = subprocess.Popen([binary, "-model", f"bench={MODEL}", "-backend", "metal", "-addr", f"127.0.0.1:{PORT}", "-quant", "int4", "-ctx", "4096", "-require-backend"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    try:
        assert bp.wait_port(PORT), "serve did not come up"
        url = f"http://127.0.0.1:{PORT}/v1/chat/completions"
        res, rss = {}, 0
        for K in DEPTHS:
            base = bp.PROMPTS[f"{MK}:{K}"]["text"]
            payload = lambda text: {"model": "bench", "stream": True, "max_tokens": 1, "temperature": 0, "stream_options": {"include_usage": True},
                                    "messages": [{"role": "user", "content": text}]}
            bp.ttft(url, payload(bp.uniq(9000) + base), bp.parse_openai)  # warm-up on its own prompt
            ts = [bp.ttft(url, payload(bp.uniq(i) + base), bp.parse_openai)[0] for i in range(6)]
            res[K] = [t for t in ts if t]
            rss = max(rss, serve_rss_kb(p.pid))
    finally:
        os.killpg(p.pid, signal.SIGTERM)
        time.sleep(6)
    i1, o1 = bench.swap_io_mb()
    return res, {"swapin_mb": round(i1 - i0, 2), "swapout_mb": round(o1 - o0, 2), "rss_kb": rss}


def run(binary, arm, pair):
    voids = 0
    while True:
        (res, swap), machine, void_starts = bench.start_clean(bench.machine_state, NEED, lambda: attempt(binary))
        entry = {"pair": pair, "arm": arm, "swap": swap, "load": (machine.get("loadavg") or [None])[0], "void_starts": len(void_starts)}
        if swap["swapout_mb"] <= 0:
            log.append(entry)
            return res
        voids += 1
        log.append(dict(entry, void=True))
        print(f"pair {pair} {arm}: VOID, swap-outs {swap['swapout_mb']} MB; {voids} in a row", flush=True)
        if voids >= 3:
            sys.exit("STOPPED: three swap voids in a row")


pairs = []
for i, order in enumerate([("old", "new"), ("new", "old"), ("old", "new")], 1):
    pair = {}
    for arm in order:
        pair[arm] = run(old if arm == "old" else new, arm, i)
        print(f"pair {i} {arm}: " + ", ".join(f"K={K} median {statistics.median(v) * 1000:.1f} ms" for K, v in pair[arm].items()) + f"; {log[-1]['swap']}", flush=True)
    pairs.append(pair)
    json.dump(dict(pairs=pairs, log=log), open(out, "w"), indent=1)
summary = {}
for K in DEPTHS:
    ratios = [statistics.median(p["old"][K]) / statistics.median(p["new"][K]) for p in pairs]
    m = round(statistics.median(ratios), 3)
    summary[K] = dict(ratios=[round(r, 3) for r in ratios], median=m, verdict="PASS" if m >= 0.97 else "AMBIGUOUS" if m >= 0.96 else "FAIL")
print("SUMMARY", MK, json.dumps(summary), flush=True)
json.dump(dict(model=MK, pairs=pairs, log=log, summary=summary), open(out, "w"), indent=1)
