#!/usr/bin/env python3
"""Gate for `connect claude-code` writing CLAUDE_CODE_ATTRIBUTION_HEADER=0 (docs/measurements/kv3-cuda-2026-10-13.md §1).

Two Claude Code conversations, different first prompts, run as KV0's header probe ran them (unsandboxed, a fixed HOME and a fixed
non-repository working directory), but with the settings file written by `serve connect claude-code -write` against a running
server, and every request recorded by kv0_proxy.py (listen -> the server). A conversation is killed once its first request body is
on disk: only the request's prefix is read, not the answer.

  header_gate.py <serve binary> <model .gguf> <out dir> [--control]

--control removes CLAUDE_CODE_ATTRIBUTION_HEADER from the written settings, the same run without the setting.
"""
import json, os, signal, subprocess, sys, time, urllib.request

serve, model, out = sys.argv[1:4]
control = "--control" in sys.argv
os.makedirs(out, exist_ok=True)
HOME, WORK = "/tmp/kv3-hdr-home", "/tmp/kv3-hdr-work"
for d in (HOME, WORK):
    os.makedirs(d, exist_ok=True)
BIN = "/home/francis/harnesses/node_modules/.bin"
HERE = os.path.dirname(os.path.abspath(__file__))
srv = subprocess.Popen([serve, "--model", model, "--backend", "cuda", "--ctx", "32768", "--kv-sessions", "1", "--addr", "127.0.0.1:18090"],
                       stdout=open(out + "/serve.log", "w"), stderr=subprocess.STDOUT, start_new_session=True, env={**os.environ, "GOINFER_SWAP_GUARD": "4096"})
try:
    for _ in range(240):
        try:
            if json.load(urllib.request.urlopen("http://127.0.0.1:18090/v1/models", timeout=2)).get("data"):
                break
        except Exception:
            time.sleep(1)
    px = subprocess.Popen([sys.executable, HERE + "/kv0_proxy.py", "18095", "18090", out + "/wire.jsonl", out + "/bodies"], start_new_session=True)
    time.sleep(1)
    cfg = HOME + "/.claude/settings.json"
    if os.path.exists(cfg):
        os.remove(cfg)
    c = subprocess.run([serve, "connect", "claude-code", "-write", "-file", cfg, "http://127.0.0.1:18095"], capture_output=True, text=True)
    open(out + "/connect.out", "w").write(c.stdout + c.stderr)
    if control:
        j = json.load(open(cfg)); j["env"].pop("CLAUDE_CODE_ATTRIBUTION_HEADER", None); json.dump(j, open(cfg, "w"))
    env_block = json.load(open(cfg))["env"]
    print("settings env written:", sorted(env_block), "ATTRIBUTION_HEADER =", env_block.get("CLAUDE_CODE_ATTRIBUTION_HEADER"))
    for i, prompt in enumerate(["Reply with the single word ok.", "Say hello in French."], 1):
        n0 = len(os.listdir(out + "/bodies")) if os.path.isdir(out + "/bodies") else 0
        p = subprocess.Popen(["claude", "-p", prompt, "--max-turns", "1", "--output-format", "json"], cwd=WORK, start_new_session=True,
                             env={"HOME": HOME, "PATH": BIN + ":/usr/bin:/bin"}, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(120):
            time.sleep(1)
            if len(os.listdir(out + "/bodies")) > n0:
                break
        time.sleep(2)
        os.killpg(p.pid, signal.SIGTERM)
        time.sleep(3)
finally:
    for pr in (locals().get("px"), srv):
        if pr:
            try:
                os.killpg(pr.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
