#!/usr/bin/env python3
"""KV5-6's G3c (docs/tasks/task-kv-architecture-2026-10.md, KV5's plan): at serve's defaults on this Mac, one conversation's cap is at
least twice main's planned per-slot context, read from both banners in one session, and a request of that length completes on Metal
with no CPU fallback and no 413.

For each model: start main's serve at its defaults (no -ctx; -require-backend, so a Metal decline refuses rather than serving on the
CPU), read its context line, stop it; start this build's serve the same way, read its context and KV plan lines; then send one chat
request whose prompt plus max_tokens is this build's cap, greedy, and read its status, usage and finish_reason. The prompt is a filler
text, its length calibrated against the server's own prompt_tokens; the request itself starts on another word, so it shares no prefix
with the calibration and prefills cold. PASS: main's context and this build's both read, this build's at least twice main's; the
request completing at least twice main's (status 200, prompt_tokens + completion_tokens, finish_reason length or stop); this build's
decode path on Metal; no line of its server log naming a fallback or a decline; and no swap-out during the request (vm_stat). Run under
the exclusive timing lock, on AC.

  python3 g3c.py SERVE_MAIN SERVE_NEW OUT.json [MODEL ...]   (models: 1.5B 7B; default both)
"""
import json, os, re, subprocess, sys, time, urllib.error, urllib.request

MODELS = {"1.5B": "~/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf", "7B": "~/models/qwen2.5-7b-instruct-q4_k_m.gguf"}
PORT, GEN = 18181, 8
BASE = f"http://127.0.0.1:{PORT}"


def say(s):
    print(time.strftime("%H:%M:%S ") + s, flush=True)


def swapouts():
    m = re.search(r"Swapouts:\s+(\d+)", subprocess.run(["vm_stat"], capture_output=True, text=True).stdout)
    return int(m.group(1)) if m else -1


def start(serve, path, log):
    f = open(log, "w")
    p = subprocess.Popen([serve, "-model", f"bench={path}", "-backend", "metal", "-addr", f"127.0.0.1:{PORT}", "-require-backend"],
                         stdout=f, stderr=subprocess.STDOUT)
    t0 = time.time()
    while time.time() - t0 < 600:
        if p.poll() is not None:
            return p, False
        try:
            with urllib.request.urlopen(BASE + "/v1/models", timeout=5) as r:
                if r.status == 200:
                    return p, True
        except Exception:
            pass
        time.sleep(1)
    return p, False


def stop(p):
    p.terminate()
    try:
        p.wait(30)
    except subprocess.TimeoutExpired:
        p.kill()
        p.wait()


def banner(log):
    text = open(log).read()
    get = lambda pat: (re.search(pat, text, re.M) or [None, ""])[0] or ""
    ctx = re.search(r"context: (\d+) tokens", text)
    return {"context_line": get(r"^.*context: .*$"), "kv_plan_line": get(r"^.*KV plan: .*$"), "kv_pool_line": get(r"^.*KV pool: .*$"),
            "decode_path_line": get(r"^.*decode path: .*$"), "context": int(ctx.group(1)) if ctx else 0}


def ask(prompt, max_tokens, timeout):
    body = json.dumps({"model": "bench", "temperature": 0, "max_tokens": max_tokens, "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(BASE + "/v1/chat/completions", body, {"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            out = json.load(r)
            return r.status, out, time.time() - t0
    except urllib.error.HTTPError as e:
        return e.code, {"error": e.read().decode()[:2000]}, time.time() - t0


WORDS = "river stone lantern copper meadow signal harbor violet engine timber orbit canyon ember falcon glacier pepper quartz saddle".split()


def filler(n, salt=0):
    return " ".join(f"{WORDS[(i + salt) % len(WORDS)]} {i % 997}" for i in range(n))


def main():
    serve_main, serve_new, out = sys.argv[1], sys.argv[2], sys.argv[3]
    models = sys.argv[4:] or list(MODELS)
    logdir = os.path.splitext(out)[0] + "-logs"
    os.makedirs(logdir, exist_ok=True)
    say(f"G3c: models {models}; power: {subprocess.run(['pmset', '-g', 'batt'], capture_output=True, text=True).stdout.splitlines()[0]}")
    results = []
    for key in models:
        path = os.path.expanduser(MODELS[key])
        row = {"model": key, "path": path}
        for arm, serve in [("main", serve_main), ("new", serve_new)]:
            log = f"{logdir}/{key}-{arm}.log"
            say(f"{key} {arm}: starting {serve}")
            p, ok = start(serve, path, log)
            row[arm] = banner(log)
            row[arm]["started"] = ok
            say(f"{key} {arm}: {'ready' if ok else 'DID NOT START'}; {row[arm]['context_line'].strip()}")
            if row[arm]["kv_plan_line"]:
                say(f"{key} {arm}: {row[arm]['kv_plan_line'].strip()}")
            if ok and arm == "new":
                cap = row["new"]["context"]
                st, o, _ = ask(filler(1000), 1, 600)
                per = o.get("usage", {}).get("prompt_tokens", 0) / 1000 if st == 200 else 0
                row["calibration"] = {"status": st, "prompt_tokens_per_item": per}
                target, got = cap - GEN, 0
                n = int(target / per) if per else 0
                for _ in range(3):  # tighten to the target within a few tokens; each try after the first reuses the last one's prefix
                    st, o, _ = ask(filler(n), 1, 900)
                    got = o.get("usage", {}).get("prompt_tokens", 0) if st == 200 else 0
                    if not got or abs(got - target) <= 4:
                        break
                    n += int((target - got) / per)
                s0 = swapouts()
                say(f"{key} new: the request, {n} filler items (calibrated {got} prompt tokens) + max_tokens {GEN}, against cap {cap}")
                st, o, wall = ask(filler(n, 7), GEN, 1800)
                row["request"] = {"status": st, "items": n, "usage": o.get("usage"), "finish_reason": (o.get("choices") or [{}])[0].get("finish_reason"),
                                  "error": o.get("error"), "wall_s": round(wall, 1), "swapout_pages": swapouts() - s0}
                say(f"{key} new: status {st}, usage {o.get('usage')}, finish {row['request']['finish_reason']}, wall {wall:.0f} s, "
                    f"swap-out pages {row['request']['swapout_pages']}")
            stop(p)
            text = open(log).read()
            row[arm]["cpu_fallback_lines"] = [l for l in text.splitlines() if re.search(r"fall(ing)? ?back|declin|on the CPU", l, re.I)][:10]
        m, n = row["main"]["context"], row["new"]["context"]
        req = row.get("request", {})
        u = req.get("usage") or {}
        done = u.get("prompt_tokens", 0) + u.get("completion_tokens", 0)
        row["verdict"] = ("PASS" if m and n >= 2 * m and req.get("status") == 200 and done >= 2 * m and req.get("finish_reason") in ("length", "stop")
                          and "metal" in row["new"]["decode_path_line"] and not row["new"]["cpu_fallback_lines"]
                          and req.get("swapout_pages") == 0 else "FAIL")
        for l in row["new"]["cpu_fallback_lines"]:
            say(f"{key} new, a fallback or decline line to read: {l.strip()[:200]}")
        say(f"{key}: main {m}, new {n} ({n / m if m else 0:.2f}x); {row['verdict']}")
        results.append(row)
        json.dump(results, open(out, "w"), indent=1)
    say("done")


if __name__ == "__main__":
    main()
