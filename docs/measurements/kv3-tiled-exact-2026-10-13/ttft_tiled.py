#!/usr/bin/env python3
"""KV3 tiled exact attention, gate G5 (docs/measurements/kv3-tiled-exact-2026-10-13.md §3): time to the first token of a prompt past the
12,160-key bound on Gemma-3-4B and Gemma-4-E4B, the batched prefill against the per-token path (the same binary with
GOINFER_BATCHED_PREFILL=0), same session, interleaved by process: batched, per-token, batched. One request per length per process, a
unique prompt each (a counter prefix), max_tokens 1, so the request's wall time is its time to first token. The ratio at a length is
per-token / median(batched). PASS: 3x or more at the first length on both models; AMBIGUOUS 1.5x to 3x; below 1.5x FAIL.

  ttft_tiled.py <serve binary> <out json>      env: KV3_TILED_LENGTHS (default 12400,20000), KV3_TILED_MODELS (default both)
"""
import json, os, signal, statistics, subprocess, sys, time, urllib.request

serve, out = sys.argv[1:3]
LENGTHS = [int(x) for x in os.environ.get("KV3_TILED_LENGTHS", "12400,20000").split(",")]
MODELS = {"gemma-3-4b": "~/models/gemma-3-4b-it-Q4_K_M.gguf", "gemma-4-e4b": "~/models/gemma-4-e4b-gguf/gemma-4-E4B_q4_0-it.gguf"}
ONLY = [m for m in os.environ.get("KV3_TILED_MODELS", ",".join(MODELS)).split(",") if m]
PORT = 18195
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.0f}s] {msg}", flush=True)


def words(n, salt):
    return f"Request {salt}. " + " ".join(f"item{(i * 7919 + salt) % 9973} of {(i * 31) % 997}" for i in range(n))


def start(model, env_extra, logpath):
    p = subprocess.Popen([serve, "--model", os.path.expanduser(model), "--backend", "cuda", "--addr", f"127.0.0.1:{PORT}", "--log-requests", "--ctx", "24576"],
                         stdout=open(logpath, "w"), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True,
                         env={**os.environ, "GOINFER_SWAP_GUARD": "4096", **env_extra})
    for _ in range(600):
        if p.poll() is not None:
            raise SystemExit(f"serve exited during load; see {logpath}")
        try:
            if json.load(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=2)).get("data"):
                return p
        except Exception:
            time.sleep(1)
    raise SystemExit("serve did not come up in 600 s")


def post(nwords, salt, timeout):
    mid = json.load(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models"))["data"][0]["id"]
    body = json.dumps({"model": mid, "max_tokens": 1, "temperature": 0, "stream": False, "messages": [{"role": "user", "content": words(nwords, salt) + "\nReply ok."}]}).encode()
    t0 = time.time()
    r = json.load(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", body, {"Content-Type": "application/json"}), timeout=timeout))
    return dict(seconds=round(time.time() - t0, 2), prompt_tokens=r["usage"]["prompt_tokens"])


def stop(p):
    try:
        os.killpg(p.pid, signal.SIGTERM); p.wait(timeout=90)
    except Exception:
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    time.sleep(5)


rec = dict(serve=serve, lengths=LENGTHS, models={})
salt = 100
for name in ONLY:
    runs = []
    for k, (arm, env) in enumerate((("batched", {}), ("per-token", {"GOINFER_BATCHED_PREFILL": "0"}), ("batched", {}))):
        logpath = f"{out}.{name}.{k}.{arm}.serve.log"
        log(f"{name} process {k + 1} of 3 ({arm}): loading")
        p = start(MODELS[name], env, logpath)
        try:
            post(20, salt, 600); salt += 1                      # warm-up
            cal = post(150, salt, 900); salt += 1               # tokens a word unit, for sizing
            ratio = cal["prompt_tokens"] / 150
            run = dict(arm=arm, cells={})
            for L in LENGTHS:
                log(f"{name} {arm}: a prompt of about {L} tokens ...")
                r = post(int(L / ratio), salt, 2400); salt += 1
                txt = open(logpath).read()
                r["declined"] = "batched prefill declined" in txt
                run["cells"][str(L)] = r
                log(f"{name} {arm}: {r['prompt_tokens']} tokens in {r['seconds']} s" + (" (the server logged a batched-prefill decline)" if r["declined"] else ""))
            runs.append(run)
        finally:
            stop(p)
        rec["models"][name] = runs
        json.dump(rec, open(out, "w"), indent=1)

summary = {}
for name, runs in rec["models"].items():
    summary[name] = {}
    for L in LENGTHS:
        b = [r["cells"][str(L)]["seconds"] for r in runs if r["arm"] == "batched"]
        pt = [r["cells"][str(L)]["seconds"] for r in runs if r["arm"] == "per-token"]
        declined = any(r["cells"][str(L)]["declined"] for r in runs if r["arm"] == "batched")
        ratio = pt[0] / statistics.median(b)
        summary[name][str(L)] = dict(batched=b, per_token=pt[0], ratio=round(ratio, 2), batched_declined=declined)
first = str(LENGTHS[0])
rs = [summary[m][first]["ratio"] for m in summary]
bad = any(summary[m][first]["batched_declined"] for m in summary)
verdict = "FAIL" if bad or min(rs) < 1.5 else ("PASS" if min(rs) >= 3 else "AMBIGUOUS")
rec["summary"], rec["verdict"] = summary, verdict
json.dump(rec, open(out, "w"), indent=1)
print("SUMMARY", json.dumps(summary), "VERDICT", verdict, flush=True)
