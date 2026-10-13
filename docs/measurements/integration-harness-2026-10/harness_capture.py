#!/usr/bin/env python3
"""I4, capture phase: run each installed harness once, headless, against capture.py (which records every request and
answers 400), from an empty directory with a fresh HOME. No model answers, so no harness runs a tool.

  harness_capture.py <harness bin dir> <outdir> [harness ...]      harnesses: claude opencode codex
"""
import json, os, signal, subprocess, sys, tempfile, time
here = os.path.dirname(os.path.abspath(__file__))
bindir, out = sys.argv[1], sys.argv[2]
which = sys.argv[3:] or ["claude", "opencode", "codex"]
PORT = 18483
PROMPT = "Reply with the single word ok."

def setup(name, home, work):
    base = f"http://127.0.0.1:{PORT}"
    env = {"HOME": home, "PATH": bindir + ":/usr/bin:/bin:/usr/local/bin"}
    if name == "claude":
        env.update(ANTHROPIC_BASE_URL=base, ANTHROPIC_API_KEY="goinfer", ANTHROPIC_MODEL="coder")
        return env, ["claude", "-p", PROMPT, "--max-turns", "1", "--output-format", "json"]
    if name == "opencode":
        json.dump({"$schema": "https://opencode.ai/config.json", "provider": {"goinfer": {
            "npm": "@ai-sdk/openai-compatible", "name": "goinfer", "options": {"baseURL": base + "/v1", "apiKey": "goinfer"},
            "models": {"coder": {"name": "coder", "limit": {"context": 32768, "output": 4096}}}}}},
            open(os.path.join(work, "opencode.json"), "w"))
        return env, ["opencode", "run", "-m", "goinfer/coder", "--format", "json", PROMPT]
    if name == "codex":
        ch = os.path.join(home, ".codex"); os.makedirs(ch)
        open(os.path.join(ch, "config.toml"), "w").write(
            'model = "coder"\nmodel_provider = "goinfer"\n\n[model_providers.goinfer]\nname = "goinfer"\n'
            f'base_url = "{base}/v1"\nwire_api = "responses"\nenv_key = "GOINFER_KEY"\n')
        env.update(CODEX_HOME=ch, GOINFER_KEY="goinfer")
        return env, ["codex", "exec", "--json", "--skip-git-repo-check", PROMPT]
    sys.exit(f"unknown harness {name}")

for name in which:
    d = os.path.join(out, name); os.makedirs(d, exist_ok=True)
    cap = subprocess.Popen([sys.executable, os.path.join(here, "capture.py"), d, str(PORT)], start_new_session=True)
    time.sleep(1)
    home, work = tempfile.mkdtemp(prefix=f"{name}-home-"), tempfile.mkdtemp(prefix=f"{name}-work-")
    env, argv = setup(name, home, work)
    t0 = time.time()
    try:
        r = subprocess.run(argv, cwd=work, env=env, capture_output=True, text=True, timeout=240, stdin=subprocess.DEVNULL)
        rc, so, se = r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired as e:
        rc, so, se = "timeout", (e.stdout or b"").decode("utf-8", "replace") if isinstance(e.stdout, bytes) else (e.stdout or ""), ""
    finally:
        os.killpg(cap.pid, signal.SIGTERM)
    open(os.path.join(d, "harness.out"), "w").write(f"argv: {argv}\nexit: {rc} after {time.time()-t0:.1f}s\n--- stdout\n{so[-3000:]}\n--- stderr\n{se[-3000:]}\n")
    n = len([f for f in os.listdir(d) if f.startswith("req-")])
    print(f"{name}: exit {rc} after {time.time()-t0:.1f} s, {n} request(s) recorded", flush=True)
