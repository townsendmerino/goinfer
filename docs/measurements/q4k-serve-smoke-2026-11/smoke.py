#!/usr/bin/env python3
"""The q4k serve smoke (README.md here). python3 smoke.py <serve binary> <out dir>. Exit 0 PASS, 1 FAIL."""
import glob, json, os, subprocess, sys, threading, time, urllib.request

PORT, HOME = 18227, os.path.expanduser("~")
BASE = f"http://127.0.0.1:{PORT}"
PHI3 = f"{HOME}/models/phi3-mini-4k-gguf/Phi-3-mini-4k-instruct-q4.gguf"
Q05B = f"{HOME}/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf"
CASES = [("phi3-default", PHI3, []), ("phi3-q4k", PHI3, ["--quant", "q4k"]), ("q05b-q4k", Q05B, ["--quant", "q4k"])]
PROMPT = "What is the capital of France? Answer in one short sentence."


def sidecars(model):
    stem = model[: -len(".gguf")]
    return {p: (os.path.getsize(p), os.path.getmtime(p)) for p in glob.glob(stem + ".*giw*")}


def run(binary, model, flags, log):
    out = open(log, "w")
    t0 = time.time()
    p = subprocess.Popen([binary, "--model", model, "--backend", "cuda", "--ctx", "4096", "--addr", f"127.0.0.1:{PORT}", "--log-requests"] + flags, stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    swap, stop = [0], threading.Event()

    def sample():
        while not stop.wait(1.0):
            try:
                for l in open(f"/proc/{p.pid}/status"):
                    if l.startswith("VmSwap"):
                        swap[0] = max(swap[0], int(l.split()[1]) // 1024)
            except OSError:
                return

    threading.Thread(target=sample, daemon=True).start()
    try:
        last = time.time()
        while True:
            if p.poll() is not None:
                return {"error": f"the server exited {p.returncode}", "load_s": time.time() - t0}
            if time.time() - t0 > 480:
                return {"error": "did not listen in 480 s", "load_s": time.time() - t0}
            if time.time() - last > 30:
                print(f"    [{time.strftime('%H:%M:%S')}] still loading, {time.time() - t0:.0f} s", flush=True)
                last = time.time()
            try:
                models = json.load(urllib.request.urlopen(BASE + "/v1/models", timeout=1))
                break
            except OSError:
                time.sleep(0.5)
        load_s = time.time() - t0
        body = {"model": models["data"][0]["id"], "messages": [{"role": "user", "content": PROMPT}], "max_tokens": 48, "temperature": 0, "seed": 1}
        req = urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
        t1 = time.time()
        r = json.load(urllib.request.urlopen(req, timeout=300))
        return {"load_s": load_s, "chat_s": time.time() - t1, "reply": r["choices"][0]["message"].get("content"), "reasoning": r["choices"][0]["message"].get("reasoning_content"), "finish": r["choices"][0].get("finish_reason"), "usage": r.get("usage"), "swap_mb": swap[0]}
    except Exception as e:  # a request that fails is the smoke's finding
        return {"error": repr(e), "swap_mb": swap[0]}
    finally:
        stop.set()
        p.terminate()
        try:
            p.wait(30)
        except subprocess.TimeoutExpired:
            p.kill()
        out.close()


def main():
    binary, outdir = sys.argv[1], sys.argv[2]
    os.makedirs(outdir, exist_ok=True)
    fail, results = 0, {}
    for name, model, flags in CASES:
        before = sidecars(model)
        print(f"[{time.strftime('%H:%M:%S')}] {name}: sidecars before: {[(os.path.basename(k), v[0]) for k, v in before.items()]}", flush=True)
        a = run(binary, model, flags, os.path.join(outdir, f"{name}-first.log"))
        mid = sidecars(model)
        new = sorted(set(mid) - set(before))
        print(f"  first: {a}\n  written: {[(os.path.basename(k), mid[k][0]) for k in new]}", flush=True)
        time.sleep(3)
        b = run(binary, model, flags, os.path.join(outdir, f"{name}-second.log"))
        after = sidecars(model)
        print(f"  second: {b}\n  sidecars unchanged by the second load: {after == mid}", flush=True)
        problems = []
        for which, x in (("first", a), ("second", b)):
            if x.get("error"):
                problems.append(f"{which}: {x['error']}")
            elif not (x.get("reply") or "").strip():
                problems.append(f"{which}: an empty reply")
            if x.get("swap_mb", 0) > 512:
                problems.append(f"{which}: VmSwap {x['swap_mb']} MB")
        if after != mid:
            problems.append("the second load changed the sidecars")
        if a.get("reply") != b.get("reply"):
            problems.append(f"the replies differ: {a.get('reply')!r} and {b.get('reply')!r}")
        if "q4k" in name and not any(".q4k." in os.path.basename(p) and p.endswith(".giw") for p in new):
            problems.append(f"no .q4k sidecar written: {[os.path.basename(p) for p in new]}")
        results[name] = {"first": a, "second": b, "written": [os.path.basename(p) for p in new], "problems": problems}
        print(f"  {name}: {'PASS by the numbers' if not problems else 'PROBLEMS: ' + '; '.join(problems)}", flush=True)
        fail += bool(problems)
        time.sleep(3)
    json.dump(results, open(os.path.join(outdir, "results.json"), "w"), indent=1)
    print(f"VERDICT: {'FAIL' if fail else 'PASS'} ({fail} of {len(CASES)} cases with a problem)")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
