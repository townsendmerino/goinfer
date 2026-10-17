#!/usr/bin/env python3
"""P28's tier 1 day measurement (goinfer docs/queue-performance.md, P28): cold prefill tok/s on Metal for Gemma 4 E2B against
Qwen2.5-Coder-1.5B at 4,096 and 8,192 prompt tokens, and the prefill path each server reports. One pinned serve-metal, each
model at -ctx 16384 and --quant int4 (tack's E0 arms read metal-resident (int4)). Per model: start the server, read its
banner, one untimed 16-token warm-up, then one cold request per length (a fresh lead token, so no prefix is reused; the
reuse count is recorded to prove it). Run it under the timing lock:

    python3 ~/tmcode/goinfer/scripts/timing_lock.py run --label p28 -- python3 run-p28.py
"""
import json, os, re, subprocess, sys, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REV = "784e7c82"
SERVE = os.path.join(HERE, f"serve-metal-{REV}")
HOME = os.path.expanduser("~")
MODELS = [
    ("gemma4-e2b-q40", f"{HOME}/models/gemma-4-e2b-gguf/gemma-4-E2B_q4_0-it.gguf"),
    ("qwen2.5-coder-1.5b-q4km", f"{HOME}/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"),
]
LENGTHS = [4096, 8192]
PORT = 18941
OUT = os.path.join(HERE, "results-" + time.strftime("%Y-%m-%d-%H%M"))


def ts():
    return time.strftime("%H:%M:%S %Z") + " / " + time.strftime("%H:%M:%S UTC", time.gmtime())


def post(path, body, timeout=1800):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def messages(lead, words):
    return [{"role": "user", "content": f"{lead} " + " ".join(words) + "\nSummarize the text above in one word."}]


def count(msgs):
    return post("/v1/messages/count_tokens", {"model": "bench", "messages": msgs})["input_tokens"]


def fit(lead, target):
    """A word list whose chat prompt is within 16 tokens of target, by bisection on the word count."""
    vocab = "the of and to in is was for on that with as by at from it an be this which or are were has had".split()
    words = lambda n: [vocab[(i * 7 + len(lead)) % len(vocab)] + str(i % 97) for i in range(n)]
    lo, hi = 1, target
    while lo < hi:
        mid = (lo + hi) // 2
        if count(messages(lead, words(mid))) < target:
            lo = mid + 1
        else:
            hi = mid
    msgs = messages(lead, words(lo))
    return msgs, count(msgs)


def main():
    if not os.access(SERVE, os.X_OK):
        sys.exit(f"missing {SERVE}")
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for name, path in MODELS:
        if path.startswith(("/Volumes/", "/srv/models")) or not os.path.exists(path):
            sys.exit(f"{name}: {path} is missing or on the archive")
        log = os.path.join(OUT, f"serve-{name}.log")
        print(f"=== {ts()} START {name}", flush=True)
        with open(log, "w") as lf:
            proc = subprocess.Popen([SERVE, "-model", f"bench={path}", "-backend", "metal", "-quant", "int4", "-ctx", "16384",
                                     "-addr", f"127.0.0.1:{PORT}", "-log-requests"], stdout=lf, stderr=subprocess.STDOUT)
        try:
            t0 = time.time()
            while "goinfer serving on" not in open(log).read():
                if proc.poll() is not None or time.time() - t0 > 600:
                    sys.exit(f"{name}: the server did not come up; see {log}")
                time.sleep(0.5)
            banner = open(log).read()
            line = lambda key: (re.search(rf"^\s*{key}: (.*)$", banner, re.M) or [None, None])[1]
            info = {"model": name, "path": path, "decode_path": line("decode path"), "prefill_path": line("prefill path"),
                    "declines": [l.strip() for l in banner.splitlines() if re.search(r"declin|batched step off", l)]}
            print(f"  decode path: {info['decode_path']}\n  prefill path: {info['prefill_path']}", flush=True)
            post("/v1/chat/completions", {"model": "bench", "messages": messages("warmup", ["x"] * 8), "max_tokens": 1, "temperature": 0})
            for n in LENGTHS:
                msgs, got = fit(f"cold{n}{name[:3]}", n)
                print(f"  {ts()} {name} {got} tokens: sending", flush=True)
                t1 = time.time()
                r = post("/v1/chat/completions", {"model": "bench", "messages": msgs, "max_tokens": 1, "temperature": 0})
                wall = time.time() - t1
                u = r.get("usage") or {}
                row = dict(info, target=n, prompt_tokens=u.get("prompt_tokens"), reused=u.get("prefill_reused_tokens"),
                           wall_s=round(wall, 3), tok_s=round((u.get("prompt_tokens") or got) / wall, 1))
                rows.append(row)
                print(f"  {ts()} {name} {row['prompt_tokens']} tokens: {wall:.1f} s, {row['tok_s']} tok/s, reused {row['reused']}", flush=True)
        finally:
            proc.terminate()
            try:
                proc.wait(30)
            except subprocess.TimeoutExpired:
                proc.kill()
        print(f"=== {ts()} END {name}", flush=True)
    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump({"rev": REV, "host": os.uname().nodename, "rows": rows}, f, indent=2)
    print(f"records: {OUT}")


if __name__ == "__main__":
    main()
