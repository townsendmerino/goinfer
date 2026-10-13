#!/usr/bin/env python3
"""Census follow-up (docs/measurements/kv3-long-prompt-census-2026-10-13.md §3): CENSUS_MODE=banner reads the context the server resolves with the flags given
(no --ctx); CENSUS_MODE=long posts CENSUS_WORDS words once and records the time and any decline.

  CENSUS_MODE=banner|long [CENSUS_WORDS=n] census2.py <serve> <model> <label> <out json> [-- serve flags]
"""

import json, os, re, signal, subprocess, sys, time, urllib.request

serve, model, label, out = sys.argv[1:5]
mode = os.environ["CENSUS_MODE"]; nwords = int(os.environ.get("CENSUS_WORDS", "0"))
flags = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
PORT = 18194
rec = dict(label=label, model=model, flags=flags)


def words(n, salt):
    return f"Request {salt}. " + " ".join(f"item{(i * 7919 + salt) % 9973} of {(i * 31) % 997}" for i in range(n))


def start(env_extra):
    log = f"{out}.{len(rec.get('runs', []))}.serve.log"
    p = subprocess.Popen([serve, "--model", model, "--backend", "cuda", "--addr", f"127.0.0.1:{PORT}", "--log-requests", *flags], stdout=open(log, "w"),
                         stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True, env={**os.environ, "GOINFER_SWAP_GUARD": "4096", **env_extra})
    for _ in range(900):
        if p.poll() is not None:
            return p, log, False
        try:
            if json.load(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=2)).get("data"):
                return p, log, True
        except Exception:
            time.sleep(1)
    return p, log, False


def post(nwords, salt, timeout):
    mid = json.load(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models"))["data"][0]["id"]
    body = json.dumps({"model": mid, "max_tokens": 1, "temperature": 0, "stream": False, "messages": [{"role": "user", "content": words(nwords, salt) + "\nReply ok."}]}).encode()
    t0 = time.time()
    try:
        r = json.load(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", body, {"Content-Type": "application/json"}), timeout=timeout))
        return dict(status=200, seconds=round(time.time() - t0, 2), prompt_tokens=r["usage"]["prompt_tokens"])
    except urllib.error.HTTPError as e:
        return dict(status=e.code, seconds=round(time.time() - t0, 2), error=e.read()[:200].decode("utf-8", "replace"))
    except Exception as e:
        return dict(status=None, seconds=round(time.time() - t0, 2), error=type(e).__name__)


def stop(p):
    try:
        os.killpg(p.pid, signal.SIGTERM); p.wait(timeout=90)
    except Exception:
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    time.sleep(5)



rec["runs"] = []
p, log, ok = start({})
txt = open(log).read()
run = dict(arm=mode, up=ok, decode_path=re.findall(r"decode path: (.*)", txt)[:1], context=re.findall(r"context: (.*)", txt)[:1], fit=re.findall(r"fit: (.*)", txt)[:1])
if ok and mode == "long":
    post(20, 1, 600)
    run["long"] = post(nwords, 33, 400)
    run["declined_lines"] = re.findall(r"batched prefill declined for a (\d+)-token prompt.*?(layer \d+ attention at \d+ attended keys[^:]*)", open(log).read())[:1]
rec["runs"].append(run)
json.dump(rec, open(out, "w"), indent=1)
stop(p)
print(json.dumps(rec, indent=1))
