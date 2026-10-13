#!/usr/bin/env python3
"""I4, capture phase: run each installed harness once, headless, against capture.py (which records every request and
answers 400), from an empty directory with a fresh HOME. No model answers, so no harness runs a tool.

  harness_capture.py <harness bin dir> <outdir> [harness ...]
    harnesses: claude opencode codex pi omp hermes openclaw cline goose aider kilo   (default: claude opencode codex)

Each harness gets its own capture port, so the recorder of one can never answer another. The 2026-10-13 additions
(pi omp hermes openclaw cline goose aider kilo) live in ~/harnesses: npm packages under node_modules/.bin, Python ones in
<name>-venv, goose as a release binary in goose/. Run the whole script under
`bwrap --bind / / --dev-bind /dev /dev --proc /proc --unshare-net` to prove a harness needs no network beyond the recorder.
"""
import json, os, signal, subprocess, sys, tempfile, time
here = os.path.dirname(os.path.abspath(__file__))
bindir, out = sys.argv[1], sys.argv[2]
which = sys.argv[3:] or ["claude", "opencode", "codex"]
PORTS = {n: 18483 + i for i, n in enumerate("claude opencode codex pi omp hermes openclaw cline goose aider kilo".split())}
HOME_H = os.path.expanduser("~/harnesses")
PROMPT = "Reply with the single word ok."

def setup(name, home, work):
    PORT = PORTS[name]
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
    if name == "pi":
        ad = os.path.join(home, ".pi", "agent"); os.makedirs(ad)
        json.dump({"providers": {"goinfer": {"baseUrl": base + "/v1", "api": "openai-completions", "apiKey": "goinfer",
            "models": [{"id": "coder", "name": "coder", "contextWindow": 32768, "maxTokens": 4096}]}}}, open(os.path.join(ad, "models.json"), "w"))
        return env, ["pi", "-p", "--provider", "goinfer", "--model", "coder", "--no-session", PROMPT]
    if name == "omp":
        ad = os.path.join(home, ".omp", "agent"); os.makedirs(ad)
        open(os.path.join(ad, "models.yml"), "w").write(
            f"providers:\n  goinfer:\n    baseUrl: {base}/v1\n    apiKey: goinfer\n    api: openai-completions\n    models:\n"
            "      - id: coder\n        name: coder\n        contextWindow: 32768\n        maxTokens: 4096\n")
        env["PATH"] = HOME_H + "/bun:" + env["PATH"]
        return env, ["omp", "-p", "--model", "goinfer/coder", "--no-session", PROMPT]
    if name == "hermes":
        hh = os.path.join(home, ".hermes"); os.makedirs(hh)
        open(os.path.join(hh, "config.yaml"), "w").write(
            f"model:\n  default: coder\n  provider: custom\n  base_url: {base}/v1\n  api_key: goinfer\n  context_length: 65536\n")
        env["PATH"] = HOME_H + "/hermes-venv/bin:" + env["PATH"]
        return env, ["hermes", "-z", PROMPT]
    if name == "openclaw":
        cd = os.path.join(home, ".openclaw"); os.makedirs(cd)
        json.dump({"models": {"mode": "replace", "providers": {"goinfer": {"baseUrl": base + "/v1", "apiKey": "goinfer", "api": "openai-completions",
            "models": [{"id": "coder", "name": "coder", "reasoning": False, "input": ["text"], "contextWindow": 32768, "maxTokens": 4096}]}}}},
            open(os.path.join(cd, "openclaw.json"), "w"))
        env["PATH"] = HOME_H + "/node-v24.21.0-linux-x64/bin:" + env["PATH"]
        return env, ["openclaw", "agent", "exec", "--config", os.path.join(cd, "openclaw.json"), "--model", "goinfer/coder", "--cwd", work, "--json", PROMPT]
    if name == "cline":
        subprocess.run(["cline", "auth", "-p", "openai-compatible", "-b", base + "/v1", "-k", "goinfer", "-m", "coder"],
                       cwd=work, env=env, capture_output=True, text=True, timeout=60, stdin=subprocess.DEVNULL)
        pj = os.path.join(home, ".cline", "data", "settings", "providers.json"); j = json.load(open(pj))
        j["providers"]["openai-compatible"]["settings"].update(contextWindow=32768, maxTokens=4096)
        json.dump(j, open(pj, "w"), indent=2)
        return env, ["cline", "-P", "openai-compatible", "-m", "coder", "--json", "--timeout", "60", PROMPT]
    if name == "goose":
        env.update(GOOSE_PROVIDER="openai", OPENAI_HOST=base, OPENAI_BASE_PATH="v1/chat/completions", OPENAI_API_KEY="goinfer",
                   GOOSE_MODEL="coder", GOOSE_CONTEXT_LIMIT="32768")
        env["PATH"] = HOME_H + "/goose:" + env["PATH"]
        return env, ["goose", "run", "--no-session", "-q", "--output-format", "stream-json", "-t", PROMPT]
    if name == "aider":
        json.dump({"openai/coder": {"max_input_tokens": 32768, "max_output_tokens": 4096, "litellm_provider": "openai", "mode": "chat"}},
                  open(os.path.join(work, ".aider.model.metadata.json"), "w"))
        env.update(OPENAI_API_BASE=base + "/v1", OPENAI_API_KEY="goinfer")
        env["PATH"] = HOME_H + "/aider-venv/bin:" + env["PATH"]
        return env, ["aider", "--model", "openai/coder", "--message", PROMPT, "--yes-always"]
    if name == "kilo":
        kd = os.path.join(home, ".config", "kilo"); os.makedirs(kd)
        json.dump({"$schema": "https://opencode.ai/config.json", "provider": {"goinfer": {
            "npm": "@ai-sdk/openai-compatible", "name": "goinfer", "options": {"baseURL": base + "/v1", "apiKey": "goinfer"},
            "models": {"coder": {"name": "coder", "limit": {"context": 32768, "output": 4096}}}}}}, open(os.path.join(kd, "kilo.json"), "w"))
        return env, ["kilo", "run", "-m", "goinfer/coder", "--format", "json", PROMPT]
    sys.exit(f"unknown harness {name}")

def reap_daemons(work):
    """cline leaves a detached `--cline-hub-daemon --cwd <work>` behind; stop the one this run started, by PID."""
    for p in os.listdir("/proc"):
        if not p.isdigit(): continue
        try: cmd = open(f"/proc/{p}/cmdline", "rb").read().split(b"\0")
        except OSError: continue
        if b"--cline-hub-daemon" in cmd and work.encode() in cmd:
            print(f"  stopping leftover daemon pid {p}", flush=True)
            os.kill(int(p), signal.SIGTERM)

for name in which:
    d = os.path.join(out, name); os.makedirs(d, exist_ok=True)
    cap = subprocess.Popen([sys.executable, os.path.join(here, "capture.py"), d, str(PORTS[name])], start_new_session=True)
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
        reap_daemons(work)
    open(os.path.join(d, "harness.out"), "w").write(f"argv: {argv}\nexit: {rc} after {time.time()-t0:.1f}s\n--- stdout\n{so[-3000:]}\n--- stderr\n{se[-3000:]}\n")
    n = len([f for f in os.listdir(d) if f.startswith("req-")])
    print(f"{name}: exit {rc} after {time.time()-t0:.1f} s, {n} request(s) recorded", flush=True)
