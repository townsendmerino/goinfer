#!/usr/bin/env python3
"""The prefill stall: what a newcomer's long prompt costs the conversations already decoding (MC3 / chunked prefill,
docs/tasks/task-concurrency-2026-09.md).

W7 cannot show it: its turns reuse their history and prefill only ~150-token suffixes. Here D "decoder" clients each
stream one long greedy generation from a short prompt, and while they run, N "newcomer" requests arrive one after
another, each with a long prompt (~P tokens of deterministic prose) and a short answer. Recorded per arm:
  - every decoder token's arrival time (SSE), so the inter-token gaps they suffer — above all during a newcomer's
    prefill — are visible directly;
  - each newcomer's time to first token;
  - a hash of every reply (decoders and newcomers), so two arms' outputs compare exactly (--seed not needed: greedy).

One fresh server per cell. Usage:
  GOINFER_SERVE_CPU=<serve binary> python3 scripts/bench_prefill_stall.py OUT.json --key old [--decoders 3]
      [--newcomers 3] [--prompt-words 2400] [--decode-tokens 400] [--server-log LOG]
"""
import argparse, hashlib, json, os, shlex, signal, subprocess, sys, threading, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench_w7_plain as w7  # GoinferServer, the model path, the free-memory guard

PARA = ("The quick survey of rate limiters begins with the token bucket, which refills at a steady rate and lets a "
        "burst spend what has accumulated; the sliding window log keeps every timestamp and counts those inside the "
        "window; the fixed window counter is cheaper but double-admits at the boundary; and the leaky bucket smooths "
        "output to a constant drain. Section {i} weighs these against a distributed deployment where clocks drift. ")


def long_prompt(words, tag):
    out, i = [], 0
    while len(" ".join(out).split()) < words:
        out.append(PARA.format(i=i))
        i += 1
    return f"[{tag}] Read the following notes, then answer in one sentence: which approach do they favour?\n\n" + "".join(out)


def stream(url, messages, max_tokens, on_token):
    body = {"model": "bench", "messages": messages, "max_tokens": max_tokens, "temperature": 0, "stream": True}
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    text = []
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                ev = json.loads(data)
            except ValueError:
                continue
            ch = (ev.get("choices") or [{}])[0]
            piece = (ch.get("delta") or {}).get("content") or ""
            if piece:
                text.append(piece)
                on_token(time.perf_counter())
    return "".join(text)


def cell(url, a):
    t0 = time.perf_counter()
    dec = [{"times": [], "text": None} for _ in range(a.decoders)]
    new = [{"sent": None, "first": None, "text": None} for _ in range(a.newcomers)]

    def decoder(i):
        msgs = [{"role": "user", "content": f"[stall-d{i}] Write a long, detailed essay on testing rate limiters under burst load."}]
        dec[i]["text"] = stream(url, msgs, a.decode_tokens, lambda t: dec[i]["times"].append(t - t0))

    def newcomer(j):
        msgs = [{"role": "user", "content": long_prompt(a.prompt_words, f"stall-n{j}")}]
        new[j]["sent"] = time.perf_counter() - t0

        def first(t):
            if new[j]["first"] is None:
                new[j]["first"] = t - t0
        new[j]["text"] = stream(url, msgs, a.newcomer_tokens, first)

    ths = [threading.Thread(target=decoder, args=(i,)) for i in range(a.decoders)]
    for th in ths:
        th.start()
    time.sleep(a.warmup)  # let the decoders reach steady decode before the first newcomer
    for j in range(a.newcomers):
        newcomer(j)  # one after another, while the decoders keep decoding
        time.sleep(a.gap)
    for th in ths:
        th.join()
    wall = time.perf_counter() - t0
    gaps = sorted(g for d in dec for g in (b - a_ for a_, b in zip(d["times"], d["times"][1:])))
    pct = lambda p: gaps[max(0, int(round(p / 100 * len(gaps))) - 1)] * 1000 if gaps else None
    return {
        "wall_s": round(wall, 3),
        "decoder_tokens": sum(len(d["times"]) for d in dec),
        "decoder_gap_ms": {"p50": pct(50), "p99": pct(99), "max": gaps[-1] * 1000 if gaps else None},
        "newcomer_ttft_s": [round(n["first"] - n["sent"], 3) if n["first"] else None for n in new],
        "decoder_times": [d["times"] for d in dec],
        "newcomer_windows": [[n["sent"], n["first"]] for n in new],
        "decoder_sha": [hashlib.sha256((d["text"] or "").encode()).hexdigest()[:16] for d in dec],
        "newcomer_sha": [hashlib.sha256((n["text"] or "").encode()).hexdigest()[:16] for n in new],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--key", required=True)
    ap.add_argument("--decoders", type=int, default=3)
    ap.add_argument("--newcomers", type=int, default=3)
    ap.add_argument("--prompt-words", type=int, default=2400)
    ap.add_argument("--decode-tokens", type=int, default=400)
    ap.add_argument("--newcomer-tokens", type=int, default=16)
    ap.add_argument("--warmup", type=float, default=2.0)
    ap.add_argument("--gap", type=float, default=0.5)
    ap.add_argument("--serve-args", default="")
    ap.add_argument("--server-log", default="")
    a = ap.parse_args()
    res = {}
    if os.path.exists(a.out):
        res = json.load(open(a.out)).get("results", {})
    with w7.GoinferServer("metal", a.serve_args, a.server_log) as srv:
        # one warm request so the first cell does not pay pipeline compile / page-in inside a decoder's timing
        w7.post(srv.url, {"model": "bench", "messages": [{"role": "user", "content": "warm"}], "max_tokens": 4, "temperature": 0})
        r = cell(srv.url, a)
    res[a.key] = r
    json.dump({"header": w7.machine_header(), "args": vars(a), "results": res}, open(a.out, "w"), indent=1)
    gap = r["decoder_gap_ms"]
    gaps = "no decoders" if gap["max"] is None else f"decoder gap p50 {gap['p50']:.1f} p99 {gap['p99']:.1f} max {gap['max']:.1f} ms"
    print(f"[stall] {a.key}: wall {r['wall_s']}s; {gaps}; newcomer TTFT {r['newcomer_ttft_s']}", file=sys.stderr)


if __name__ == "__main__":
    main()
