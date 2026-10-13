#!/usr/bin/env python3
"""I2 live gate, Codex's two keys: does Codex 0.162.1 honour top-level model_context_window and
model_auto_compact_token_limit as `connect -write` sets them, and does it compact by itself before the window?

  codex_compact_probe.py <serve binary> <server url> <harness bin dir> <outdir> ["prompt"]

A throwaway HOME and CODEX_HOME; the config comes from `connect -write` and is not edited. Codex runs with its
own default read-only sandbox, so the shell commands it is steered into (large outputs) cannot write. The result
is read from Codex's own session log (token_count events carry the window it assumed; a compaction is a
`compacted` record) and from the goinfer request log the caller keeps.
"""
import glob, json, os, subprocess, sys, tempfile, time

serve, url, bindir, out = sys.argv[1:5]
prompt = sys.argv[5] if len(sys.argv) > 5 else (
    "Use your shell tool to run exactly this command: seq 1 8000   Then run the same command again, and again, "
    "three times in total, one call each. After the third, reply with the word done.")
os.makedirs(out, exist_ok=True)
home, work = tempfile.mkdtemp(prefix="cxp-home-"), tempfile.mkdtemp(prefix="cxp-work-")
ch = os.path.join(home, ".codex")
env = {"HOME": home, "CODEX_HOME": ch, "GOINFER_API_KEY": "goinfer", "PATH": bindir + ":/usr/bin:/bin:/usr/local/bin"}
w = subprocess.run([serve, "connect", "codex", "-write", url], capture_output=True, text=True, env=env)
print(w.stdout.splitlines()[0], flush=True)
t0 = time.time()
try:
    r = subprocess.run(["codex", "exec", "--json", "--skip-git-repo-check", prompt], cwd=work, env=env,
                       capture_output=True, text=True, timeout=900, stdin=subprocess.DEVNULL)
    rc, so, se = r.returncode, r.stdout, r.stderr
except subprocess.TimeoutExpired as e:
    rc, so, se = "timeout", (e.stdout or b"").decode("utf-8", "replace"), ""
dt = time.time() - t0
open(os.path.join(out, "codex.out"), "w").write(f"exit {rc} after {dt:.0f}s\n--- stdout\n{so}\n--- stderr\n{se[-3000:]}")
print(f"codex exit {rc} after {dt:.0f}s", flush=True)
for f in glob.glob(os.path.join(ch, "sessions", "**", "*.jsonl"), recursive=True):
    n = os.path.join(out, "session.jsonl")
    open(n, "w").write(open(f).read())
    rows = [json.loads(l) for l in open(f) if l.strip()]
    kinds = {}
    for x in rows:
        k = x.get("type", "?") + ("/" + x["payload"]["type"] if isinstance(x.get("payload"), dict) and "type" in x["payload"] else "")
        kinds[k] = kinds.get(k, 0) + 1
    print("session record kinds:", json.dumps(kinds, sort_keys=True))
    for x in rows:
        p = x.get("payload") or {}
        if p.get("type") == "token_count" and p.get("info"):
            i = p["info"]
            print("token_count: window", i.get("model_context_window"), "last_total", (i.get("last_token_usage") or {}).get("total_tokens"),
                  "total", (i.get("total_token_usage") or {}).get("total_tokens"))
print("home", home)
