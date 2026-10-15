#!/usr/bin/env python3
"""KV3 admission gate (docs/measurements/kv3-cuda-2026-10-13.md §3.1): time to first token at K = 512 and K = 3,900 on the 1.5B, the new
serve binary against the old, three alternating pairs, a fresh server per run, six distinct prompts per depth (a unique prefix
each, so no reuse can serve them), a warm-up request first. TTFT is client-timed to the first streamed content event
(scripts/bench_peer_prefill.py's own ttft()). The ratio is old/new per pair and its median over pairs; PASS is 0.97 or better at both depths.

  ttft_ab.py <old serve> <new serve> <out json>
"""
import json, os, signal, statistics, subprocess, sys, time

HERE = "/home/francis/mycode/goinfer/scripts"
sys.path.insert(0, HERE)
import bench_peer_prefill as bp  # noqa: E402

old, new, out = sys.argv[1:4]
MODEL = os.path.expanduser("~/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf")
DEPTHS = (512, 3900)
PORT = bp.GPORT


def run(binary):
    p = subprocess.Popen([binary, "-model", f"bench={MODEL}", "-backend", "cuda", "-addr", f"127.0.0.1:{PORT}", "-quant", "int4", "-require-backend"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    try:
        assert bp.wait_port(PORT), "serve did not come up"
        url = f"http://127.0.0.1:{PORT}/v1/chat/completions"
        res = {}
        for K in DEPTHS:
            base = bp.PROMPTS[f"1.5B:{K}"]["text"]
            payload = lambda text: {"model": "bench", "stream": True, "max_tokens": 1, "temperature": 0, "stream_options": {"include_usage": True},
                                    "messages": [{"role": "user", "content": text}]}
            bp.ttft(url, payload(bp.uniq(9000) + base), bp.parse_openai)  # warm-up on its own prompt
            ts = [bp.ttft(url, payload(bp.uniq(i) + base), bp.parse_openai)[0] for i in range(6)]
            res[K] = [t for t in ts if t]
        return res
    finally:
        os.killpg(p.pid, signal.SIGTERM)
        time.sleep(6)


pairs = []
for i, order in enumerate([("old", "new"), ("new", "old"), ("old", "new")], 1):
    pair = {}
    for arm in order:
        pair[arm] = run(old if arm == "old" else new)
        print(f"pair {i} {arm}: " + ", ".join(f"K={K} median {statistics.median(v) * 1000:.1f} ms" for K, v in pair[arm].items()), flush=True)
    pairs.append(pair)
    json.dump(pairs, open(out, "w"), indent=1)
summary = {}
for K in DEPTHS:
    ratios = [statistics.median(p["old"][K]) / statistics.median(p["new"][K]) for p in pairs]
    summary[K] = dict(ratios=[round(r, 3) for r in ratios], median=round(statistics.median(ratios), 3), verdict="PASS" if statistics.median(ratios) >= 0.97 else "FAIL")
print("SUMMARY", json.dumps(summary), flush=True)
json.dump(dict(pairs=pairs, summary=summary), open(out, "w"), indent=1)
