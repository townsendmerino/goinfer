#!/usr/bin/env python3
"""M2 served check (docs/tasks/task-minimax-code-compat-2026-09.md): start goinfer-serve on Metal with NO -ctx,
print the context lines of its banner, send one request of a coding agent's first-request size, and read the
machine's swap before the load and after the answer.

  m2_served_check.py <serve binary> <model path> <log dir> [--tokens 11137] [--extra-arg ARG ...] [--banner-only]
"""
import argparse, json, os, re, signal, subprocess, sys, time, urllib.request

PORT = 18477

def swap_used_mb():
    out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
    m = re.search(r"used = ([\d.]+)M", out)
    return float(m.group(1)) if m else float("nan")

def post(body, timeout):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")

def words(n):
    return " ".join(f"item{i % 977} holds value {i * 7 % 1013}," for i in range(n // 4))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("binary"); ap.add_argument("model"); ap.add_argument("logdir")
    ap.add_argument("--tokens", type=int, default=11137)
    ap.add_argument("--extra-arg", action="append", default=[])
    ap.add_argument("--banner-only", action="store_true")
    ap.add_argument("--load-timeout", type=int, default=420)
    a = ap.parse_args()
    os.makedirs(a.logdir, exist_ok=True)
    name = os.path.basename(a.model)
    logp = os.path.join(a.logdir, name + ".serve.log")
    swap0 = swap_used_mb()
    print(f"[{time.strftime('%T')}] {name}: swap used before the load {swap0:.0f} MB", flush=True)
    argv = [a.binary, "-model", f"bench={a.model}", "-backend", "metal", "-addr", f"127.0.0.1:{PORT}", "-require-backend"] + a.extra_arg
    print("  " + " ".join(argv), flush=True)
    log = open(logp, "w")
    proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    rc = 0
    try:
        t0 = time.time()
        up = False
        while time.time() - t0 < a.load_timeout:
            if proc.poll() is not None:
                break
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=2).read()
                up = True
                break
            except Exception:
                time.sleep(1)
        text = open(logp).read()
        for line in text.splitlines():
            if re.search(r"context:|harness:|KV plan|fit:|decode path|metal: |\[resident\]|refus|declin", line):
                print("  | " + line.strip()[:260], flush=True)
        if not up:
            print(f"  serve did not come up (exit {proc.poll()}) after {time.time() - t0:.0f} s; log {logp}", flush=True)
            return 1
        print(f"  up after {time.time() - t0:.0f} s; swap used {swap_used_mb():.0f} MB", flush=True)
        if a.banner_only:
            return 0
        # Calibrate tokens per generated word on a small request, then size the real one.
        st, small = post({"model": "bench", "messages": [{"role": "user", "content": words(800)}], "max_tokens": 1}, 300)
        per = small["usage"]["prompt_tokens"] / 800
        n = int((a.tokens + 60) / per)
        t1 = time.time()
        st, big = post({"model": "bench", "messages": [{"role": "user", "content": words(n)}], "max_tokens": 8, "temperature": 0}, 1500)
        dt = time.time() - t1
        swap1 = swap_used_mb()
        if st != 200:
            print(f"  REFUSED ({st}): {json.dumps(big)[:300]}", flush=True)
            rc = 1
        else:
            u = big["usage"]
            reply = big["choices"][0]["message"]["content"]
            print(f"  answered in {dt:.1f} s: prompt_tokens {u['prompt_tokens']}, completion_tokens {u['completion_tokens']}, reply {reply!r}", flush=True)
            if u["prompt_tokens"] < a.tokens:
                print(f"  NOTE: the prompt is {u['prompt_tokens']} tokens, under the {a.tokens} asked", flush=True)
                rc = 1
        print(f"  swap used after the answer {swap1:.0f} MB (before the load {swap0:.0f} MB, growth {swap1 - swap0:+.0f} MB)", flush=True)
        if swap1 - swap0 > 0:
            rc = 1
        return rc
    finally:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=30)
        except Exception:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:
                pass

if __name__ == "__main__":
    sys.exit(main())
