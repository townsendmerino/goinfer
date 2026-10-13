#!/usr/bin/env python3
"""I2's live gate: for each agent, `connect <agent> -write` against a running server, one headless turn that must
succeed, then `connect <agent> -undo`, and the file byte-identical to before (or gone, if connect created it).

  connect_live.py <serve binary> <server url> <harness bin dir> <outdir> [agent ...]

Every run is in a throwaway HOME and work directory, and -write is given an explicit -file inside them, so no real
configuration file is ever named. Each agent runs twice: `fresh` (no file before, so -undo must remove it) and
`existing` (a file with unrelated settings, so -undo must give back its exact bytes).
"""
import hashlib, json, os, subprocess, sys, tempfile, time

serve, url, bindir, out = sys.argv[1:5]
agents = sys.argv[5:] or ["claude-code", "opencode", "codex"]
PROMPT = "Reply with the single word ok."
os.makedirs(out, exist_ok=True)


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest() if os.path.exists(p) else None


def model_id():
    import urllib.request
    d = json.load(urllib.request.urlopen(url + "/v1/models"))["data"]
    return [m["id"] for m in d if m.get("context_window")][0]


EXISTING = {
    "claude-code": ('{\n  "theme": "dark",\n  "env": {"KEEP_ME": "1"},\n  "permissions": {"allow": ["Bash(ls)"]}\n}\n'),
    "opencode": ('{\n  "$schema": "https://opencode.ai/config.json",\n  "theme": "tokyonight",\n  "provider": {"other": {"name": "other"}}\n}\n'),
    "codex": ('# my codex settings\nmodel_reasoning_effort = "low"\n\n[tui]\nnotifications = true\n'),
}


def layout(agent, home, work):
    """(config path, env for the agent, argv)"""
    env = {"HOME": home, "PATH": bindir + ":/usr/bin:/bin:/usr/local/bin"}
    m = model_id()
    if agent == "claude-code":
        return os.path.join(home, ".claude", "settings.json"), env, ["claude", "-p", PROMPT, "--output-format", "json"]
    if agent == "opencode":
        return os.path.join(work, "opencode.json"), env, ["opencode", "run", "-m", f"goinfer/{m}", "--format", "json", PROMPT]
    if agent == "codex":
        ch = os.path.join(home, ".codex")
        env.update(CODEX_HOME=ch, GOINFER_API_KEY="goinfer")
        return os.path.join(ch, "config.toml"), env, ["codex", "exec", "--json", "--skip-git-repo-check", PROMPT]
    sys.exit(f"unknown agent {agent}")


def reply_text(agent, stdout):
    """The agent's own final answer, read from its JSON output; "" when it printed none."""
    out = []
    for line in stdout.splitlines():
        try:
            j = json.loads(line)
        except ValueError:
            continue
        if agent == "claude-code" and j.get("type") == "result":
            out.append(str(j.get("result", "")))
        elif agent == "opencode" and j.get("type") == "text":
            out.append(j.get("part", {}).get("text", ""))
        elif agent == "codex" and j.get("type") == "item.completed" and j.get("item", {}).get("type") == "agent_message":
            out.append(j["item"].get("text", ""))
    return "".join(out)


def run(argv, env, cwd, timeout):
    t0 = time.time()
    try:
        r = subprocess.run(argv, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)
        return r.returncode, r.stdout, r.stderr, time.time() - t0
    except subprocess.TimeoutExpired as e:
        so = e.stdout.decode("utf-8", "replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        return "timeout", so, "", time.time() - t0


results = []
for agent in agents:
    for case in ("fresh", "existing"):
        home, work = tempfile.mkdtemp(prefix=f"live-{agent}-home-"), tempfile.mkdtemp(prefix=f"live-{agent}-work-")
        path, env, argv = layout(agent, home, work)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if case == "existing":
            open(path, "w").write(EXISTING[agent])
        before = sha(path)
        cm = [serve, "connect", agent, "-file", path]
        w = subprocess.run(cm + ["-write", url], capture_output=True, text=True, env=env)
        wrote = sha(path)
        rc, so, se, dt = run(argv, env, work, 900)
        u = subprocess.run(cm + ["-undo", url], capture_output=True, text=True, env=env)
        after = sha(path)
        leftovers = [f for f in os.listdir(os.path.dirname(path)) if "goinfer-" in f]
        reply = reply_text(agent, so or "")
        reply_ok = reply.strip().lower().strip(".!") == "ok"
        row = dict(agent=agent, case=case, write_rc=w.returncode, wrote_changed=wrote != before, turn_rc=rc,
                   turn_seconds=round(dt, 1), reply=reply[:80], reply_is_ok=reply_ok, undo_rc=u.returncode,
                   byte_identical=(after == before), leftovers=leftovers)
        row["pass"] = (w.returncode == 0 and rc == 0 and reply_ok and u.returncode == 0 and after == before and not leftovers)
        results.append(row)
        d = os.path.join(out, f"{agent}-{case}")
        os.makedirs(d, exist_ok=True)
        open(os.path.join(d, "connect-write.txt"), "w").write(w.stdout + w.stderr)
        open(os.path.join(d, "agent.out"), "w").write(f"argv: {argv}\nexit: {rc} after {dt:.1f}s\n--- stdout\n{so[-6000:]}\n--- stderr\n{se[-4000:]}\n")
        print(json.dumps(row), flush=True)
json.dump(results, open(os.path.join(out, "results.json"), "w"), indent=1)
sys.exit(0 if all(r["pass"] for r in results) else 1)
