"""Speed guard for the windowed-KV step compaction: decode rate, new / main, two goinfer arms, ABBA, served.

Registered in docs/measurements/prefix-reuse-windowed-kv-2026-10-14.md section 8 before any run. Usage:

    speed_guard.py <serve-main> <serve-new> <out-dir>           # the registered run
    SMOKE=1 speed_guard.py <serve-main> <serve-new> <out-dir>   # one Mellum session per arm, one short completion

Each session starts one arm's server, runs K completions of NGEN tokens on /v1/completions (raw text, greedy) from a
prompt deep enough that decode crosses the window's compactions, times decode client-side from the first streamed token
to the last, and stops the server by its PID. Sessions run in the order main, new, new, main, new, main, main, new per model. Results go to
<out-dir>/results.json after every completion; <out-dir>/summary.json holds the per-model ratios and the verdict.
"""
import json, os, signal, statistics, subprocess, sys, time, urllib.request

MAIN, NEW, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
SMOKE = os.environ.get("SMOKE") == "1"
M = os.path.expanduser("~/models")
PORT = 18231
MODELS = [
    ("mellum", ["--model", f"{M}/mellum2.1-thinking", "--ctx", "16384", "--moe-cache-experts"]),
    ("gemma4-26b", ["--moe-cache-experts", "--model", f"{M}/gemma4-26b-int4.giw"]),
]
ORDER = ["main", "new", "new", "main", "new", "main", "main", "new"]  # two quads, ABBA then BAAB: 4 restart pairs an arm (amendment 1)
K, NGEN, NUMBERS = 3, 640, 400  # the prompt counts 1..NUMBERS: about 1,950 tokens on Mellum2.1
if SMOKE:
    MODELS, ORDER, K, NGEN = MODELS[:1], ["main", "new"], 1, 64
PROMPT = ", ".join(str(i) for i in range(1, NUMBERS + 1)) + ","

os.makedirs(OUT, exist_ok=True)
results = []


def save():
    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(results, f, indent=1)


def wait_idle(limit=300):
    t0 = time.time()
    while os.getloadavg()[0] >= 1.0 and time.time() - t0 < limit:
        time.sleep(5)
    return round(os.getloadavg()[0], 2), round(time.time() - t0, 1)


def start(binary, args, log_path):
    log = open(log_path, "ab")
    p = subprocess.Popen([binary, "--backend", "cuda", "--addr", f"127.0.0.1:{PORT}"] + args,
                         stdout=subprocess.DEVNULL, stderr=log, preexec_fn=os.setsid)
    t0 = time.time()
    while time.time() - t0 < 600:
        text = open(log_path, errors="replace").read()
        if "goinfer serving" in text:
            return p, text
        if p.poll() is not None:
            break
        time.sleep(1)
    stop(p)
    raise RuntimeError(f"server did not come up: {log_path}")


def stop(p):
    if p.poll() is None:
        os.killpg(p.pid, signal.SIGTERM)
        try:
            p.wait(60)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            p.wait()


def model_id():
    with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=60) as r:
        return json.load(r)["data"][0]["id"]


def complete(model):
    body = json.dumps({"model": model, "prompt": PROMPT, "max_tokens": NGEN, "temperature": 0, "stream": True,
                       "stream_options": {"include_usage": True}}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/completions", body, {"Content-Type": "application/json"})
    t_first = t_last = None
    events, text, usage = 0, [], {}
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode(errors="replace").strip()
            if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                continue
            d = json.loads(line[5:])
            if d.get("usage"):
                usage = d["usage"]
            ch = d.get("choices") or []
            if ch and ch[0].get("finish_reason") is None and ch[0].get("text") is not None:
                now = time.time()
                t_first = t_first or now
                t_last = now
                events += 1
                text.append(ch[0]["text"])
    n = usage.get("completion_tokens") or events
    rate = (n - 1) / (t_last - t_first) if t_first and t_last > t_first else None
    return {"rate": rate, "tokens": n, "events": events, "prompt_tokens": usage.get("prompt_tokens"),
            "ttft": round(t_first - t0, 2) if t_first else None, "text": "".join(text)}


for name, args in MODELS:
    for block, arm in enumerate(ORDER):
        load, waited = wait_idle()
        log_path = os.path.join(OUT, f"serve-{name}-{block}-{arm}.log")
        p, text = start(MAIN if arm == "main" else NEW, args, log_path)
        try:
            session = {"model": name, "block": block, "arm": arm, "load1": load, "idle_wait_s": waited,
                       "resident": "decode path: cuda-resident" in text, "windowed": "cuda: windowed KV:" in text,
                       "plan": [l for l in text.splitlines() if l.startswith(("cuda:", "[cuda] C′", "  KV plan"))]}
            model = model_id()
            for k in range(K):
                c = complete(model)
                results.append(dict(session, k=k, **c))
                save()
                print(f"[{time.strftime('%H:%M:%S')}] {name} block {block} {arm} k={k}: {c['rate'] and round(c['rate'], 2)} tok/s, "
                      f"{c['tokens']} tokens, prompt {c['prompt_tokens']}", flush=True)
        finally:
            stop(p)

summary = {"registered": "docs/measurements/prefix-reuse-windowed-kv-2026-10-14.md section 8", "smoke": SMOKE, "models": {}}
verdicts = []
for name, _ in MODELS:
    rows = [r for r in results if r["model"] == name]
    void = [f"block {r['block']} {r['arm']} k={r['k']}: " + ("not resident" if not r["resident"] else "no windowed KV" if not r["windowed"]
            else f"{r['tokens']} tokens") for r in rows
            if not r["resident"] or not r["windowed"] or r["rate"] is None or r["tokens"] < 0.9 * NGEN]
    med = {arm: statistics.median([r["rate"] for r in rows if r["arm"] == arm and r["rate"]] or [0]) for arm in ("main", "new")}
    ratio = med["new"] / med["main"] if med["main"] else None
    texts = {arm: {r["text"] for r in rows if r["arm"] == arm} for arm in ("main", "new")}
    v = "VOID" if void or ratio is None else "PASS" if ratio >= 0.99 else "FAIL" if ratio < 0.97 else "AMBIGUOUS"
    verdicts.append(v)
    summary["models"][name] = {"median_main": med["main"], "median_new": med["new"], "ratio": ratio, "verdict": v, "void": void,
                               "texts_within_main": len(texts["main"]), "texts_within_new": len(texts["new"]),
                               "texts_identical_across_arms": texts["main"] == texts["new"]}
summary["verdict"] = "VOID" if "VOID" in verdicts else "FAIL" if "FAIL" in verdicts else "AMBIGUOUS" if "AMBIGUOUS" in verdicts else "PASS"
with open(os.path.join(OUT, "summary.json"), "w") as f:
    json.dump(summary, f, indent=1)
print(json.dumps(summary, indent=1))
