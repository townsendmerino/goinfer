#!/usr/bin/env python3
"""I4 census, Claude Code: serve a small model with a window nothing overflows and -log-requests, send one trivial headless
prompt from an empty directory, and print what the server saw. cc_census.py <serve> <model.gguf> <outdir> [VAR=VALUE ...]"""
import json, os, subprocess, sys, tempfile, time, urllib.request, signal
serve, model, out = sys.argv[1:4]
extra = dict(a.split("=", 1) for a in sys.argv[4:])
mode = extra.pop("CC_MODE", "bare")
PORT = 18482
os.makedirs(out, exist_ok=True)
log = open(os.path.join(out, "serve.log"), "w")
p = subprocess.Popen([serve, "-model", f"coder={model}", "-backend", "metal", "-ctx", "32768", "-addr", f"127.0.0.1:{PORT}",
                      "-require-backend", "-log-requests"], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
try:
    for _ in range(120):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=2).read(); break
        except Exception:
            time.sleep(1)
    else:
        sys.exit("serve did not come up")
    work = tempfile.mkdtemp(prefix="cc-census-")
    home = os.environ["HOME"] if mode == "bare" else tempfile.mkdtemp(prefix="cc-home-")
    env = {"HOME": home, "PATH": os.environ["PATH"], "ANTHROPIC_BASE_URL": f"http://127.0.0.1:{PORT}",
           "ANTHROPIC_MODEL": "coder"}
    env.update(extra)
    argv = ["claude"] + (["--bare"] if mode == "bare" else []) + ["-p", "Reply with the single word ok.", "--max-turns", "1", "--output-format", "json"]
    print("mode:", mode, "(full = no --bare, a fresh empty HOME)", flush=True)
    print("env:", {k: v for k, v in env.items() if k not in ("HOME", "PATH")}, flush=True)
    print("argv:", " ".join(argv), "(cwd: an empty temp dir)", flush=True)
    t0 = time.time()
    try:
        r = subprocess.run(argv, cwd=work, env=env, capture_output=True, text=True, timeout=400, stdin=subprocess.DEVNULL)
        print(f"claude exit {r.returncode} after {time.time()-t0:.1f} s", flush=True)
        try:
            j = json.loads(r.stdout)
            print("result:", json.dumps({k: j.get(k) for k in ("subtype", "is_error", "num_turns", "result", "usage")})[:600])
        except Exception:
            print("stdout:", r.stdout[:600]); print("stderr:", r.stderr[:600])
    except subprocess.TimeoutExpired:
        print("claude timed out after 400 s")
    time.sleep(1)
finally:
    try: os.killpg(p.pid, signal.SIGTERM)
    except Exception: pass
log.close()
print("--- what the server logged for requests:")
for line in open(os.path.join(out, "serve.log")):
    if any(k in line for k in ("/v1/", " 400", " 404", "req ", "POST", "GET")):
        print("  " + line.rstrip()[:300])
