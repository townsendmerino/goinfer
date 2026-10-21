#!/usr/bin/env python3
"""KV3 tiled exact attention, gate G6 (docs/measurements/kv3-tiled-exact-2026-10-13.md §3 and §9): no regression where the row kernels do
not route. Time to first token at the 1.5B's K = 512 and K = 3,900 prompts, on the 1.5B and on Gemma-3-4B (the same texts; Gemma's
tokenizer makes them about as long), the new serve binary against the old, alternating pairs, a fresh server per run, six distinct
prompts per depth, a warm-up first. It is docs/measurements/kv3-cuda-2026-10-13/ttft_ab.py with the model named. The ratio is old/new
per pair and its median over pairs; PASS is 0.97 or better at both depths on both models.

  ttft_ab_models.py <old serve> <new serve> <out json>      env: KV3_TILED_PAIRS (default 3), KV3_TILED_MODELS (default both)
"""
import json, os, signal, statistics, subprocess, sys, time

HERE = "/home/francis/mycode/goinfer/scripts"
sys.path.insert(0, HERE)
import bench_peer_prefill as bp  # noqa: E402

old, new, out = sys.argv[1:4]
MODELS = {"1.5B": ("~/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf", ["-quant", "int4"]), "gemma-3-4b": ("~/models/gemma-3-4b-it-Q4_K_M.gguf", [])}
ONLY = [m for m in os.environ.get("KV3_TILED_MODELS", ",".join(MODELS)).split(",") if m]
PAIRS = int(os.environ.get("KV3_TILED_PAIRS", "3"))
DEPTHS = (512, 3900)
PORT = bp.GPORT


def run(binary, model, flags):
    p = subprocess.Popen([binary, "-model", f"bench={os.path.expanduser(model)}", "-backend", "cuda", "-addr", f"127.0.0.1:{PORT}", "-require-backend", *flags],
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
            assert len(res[K]) == 6, f"K={K}: {len(res[K])} of 6 requests returned a time"
        return res
    finally:
        os.killpg(p.pid, signal.SIGTERM)
        time.sleep(6)


rec, summary = {}, {}
for name in ONLY:
    model, flags = MODELS[name]
    pairs = []
    for i in range(1, PAIRS + 1):
        pair = {}
        for arm in (("old", "new") if i % 2 else ("new", "old")):
            pair[arm] = run(old if arm == "old" else new, model, flags)
            print(f"{name} pair {i} {arm}: " + ", ".join(f"K={K} median {statistics.median(v) * 1000:.1f} ms" for K, v in pair[arm].items()), flush=True)
        pairs.append(pair)
        rec[name] = pairs
        json.dump(rec, open(out, "w"), indent=1)
    summary[name] = {}
    for K in DEPTHS:
        ratios = [statistics.median(p["old"][K]) / statistics.median(p["new"][K]) for p in pairs]
        summary[name][K] = dict(ratios=[round(r, 3) for r in ratios], median=round(statistics.median(ratios), 3), verdict="PASS" if statistics.median(ratios) >= 0.97 else "FAIL")
verdict = "PASS" if all(c["verdict"] == "PASS" for m in summary.values() for c in m.values()) else "FAIL"
print("SUMMARY", json.dumps(summary), "VERDICT", verdict, flush=True)
json.dump(dict(pairs=rec, summary=summary, verdict=verdict), open(out, "w"), indent=1)
