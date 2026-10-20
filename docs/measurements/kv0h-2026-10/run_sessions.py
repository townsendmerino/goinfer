#!/usr/bin/env python3
"""KV0-H (kv0h-hybrid-traffic-2026-10.md), a copy of KV0's run_sessions.py with two changes: the swap-guard env is set only when the plan names a
swap_guard_mb (a build with the per-process swap guard takes the default, CLAUDE.md), and the server also writes --log-token-ids (the server's own
generated ids and prompt hash for every generation) beside its --log-requests log.

KV0 traffic study, collection: real harness sessions against goinfer-serve, every request recorded by kv0_proxy.py.

  run_sessions.py <plan.json> <out dir> <bodies dir> [--only <config>[,<config>]]

plan.json: {"serve": path, "port", "proxy_port", "swap_guard_mb", "configs": [{"name", "model", "ctx", "serve_args": [],
"sessions": [{"agent", "task", "rep", "prompt"?}], "run_timeout"}]}.

One session is one headless harness invocation (matrix/run_cell.py: a fresh fixture copy and HOME, the sandbox, a mechanical
check of the result). The server stays up across a configuration's sessions, so a session's first request meets whatever the
previous one left in the single KV slot, as a real user's next conversation would. The proxy is restarted per session so that
each session's bodies land in <bodies dir>/<session id>/ (outside the repository; the bodies carry each harness's own
prompts). Records: <out>/records.jsonl (run_cell's) and <out>/sessions.jsonl (id, config, agent, task, rep, times).
"""
import argparse, json, os, signal, subprocess, sys, threading, time

HERE = os.path.dirname(os.path.abspath(__file__))
KV0 = os.path.join(HERE, "..", "kv0-traffic-2026-10")  # the proxy and the run-cell overlay are KV0's, unchanged
MATRIX = os.path.join(HERE, "..", "integration-harness-2026-10", "matrix")
sys.path.insert(0, MATRIX)
import sandbox, matrix  # noqa: E402

T0 = time.time()
STATE = dict(done=0, total=0, current="-", t=time.time(), phase="start")


def say(msg):
    print(f"[{time.strftime('%H:%M:%S')} +{int(time.time() - T0) // 60}m] {msg}", flush=True)


def heartbeat():
    while True:
        time.sleep(30)
        say(f"heartbeat: phase={STATE['phase']} sessions {STATE['done']}/{STATE['total']} current={STATE['current']} (running {int(time.time() - STATE['t'])} s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("plan"); ap.add_argument("out"); ap.add_argument("bodies")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    plan = json.load(open(a.plan))
    os.makedirs(a.out, exist_ok=True)
    only = set(filter(None, a.only.split(",")))
    cfgs = [c for c in plan["configs"] if not only or c["name"] in only]
    STATE["total"] = sum(len(c["sessions"]) for c in cfgs)
    threading.Thread(target=heartbeat, daemon=True).start()
    port, pport = plan["port"], plan["proxy_port"]
    done = set()
    sess_path = os.path.join(a.out, "sessions.jsonl")
    if os.path.exists(sess_path):
        done = {json.loads(l)["id"] for l in open(sess_path) if l.strip()}
    tested = False
    for c in cfgs:
        STATE["phase"] = f"serve {c['name']}"
        logdir = os.path.join(a.out, "logs"); os.makedirs(logdir, exist_ok=True)
        argv = [plan["serve"], "--model", c["model"], "--backend", "cuda", "--ctx", str(c["ctx"]), "--addr", f"127.0.0.1:{port}", "--log-requests",
                "--log-token-ids", os.path.join(logdir, f"tokenids-{c['name']}.jsonl")] + c.get("serve_args", [])
        slog = open(os.path.join(logdir, f"serve-{c['name']}.log"), "a")
        srv = subprocess.Popen(argv, stdout=slog, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True,
                               env=({**os.environ, "GOINFER_SWAP_GUARD": str(plan["swap_guard_mb"])} if "swap_guard_mb" in plan else dict(os.environ)))
        try:
            t = time.time()
            matrix.wait_ready(port, srv, seconds=1800)
            say(f"config {c['name']}: server up in {time.time() - t:.0f} s: {' '.join(argv[1:])}")
            if not tested:
                ok, rep = sandbox.selftest(port, verbose=False)
                open(os.path.join(a.out, "sandbox-selftest.txt"), "w").write(rep)
                if not ok:
                    say("SANDBOX SELFTEST FAILED: no agent will run.\n" + rep)
                    return 3
                ok, rep = sandbox.selftest_mutants(port)
                open(os.path.join(a.out, "sandbox-mutants.txt"), "w").write(rep)
                if not ok:
                    say("the sandbox self-test did not catch a planted weakness: no agent will run.\n" + rep)
                    return 3
                say("sandbox self-test passed and catches its planted weaknesses")
                tested = True
            smi = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
            say(f"config {c['name']}: VRAM after load {smi}")
            for s in c["sessions"]:
                sid = f"{c['name']}-{s['agent']}-{s['task']}-r{s['rep']}"
                STATE.update(phase=f"session {sid}", current=sid, t=time.time())
                if sid in done:
                    STATE["done"] += 1
                    continue
                bdir = os.path.join(a.bodies, sid)
                wlog = os.path.join(a.out, f"wire-{sid}.jsonl")
                proxy = subprocess.Popen([sys.executable, os.path.join(KV0, "kv0_proxy.py"), str(pport), str(port), wlog, bdir],
                                         stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
                time.sleep(1)
                t0 = time.time()
                cmd = [sys.executable, os.path.join(KV0, "kv0_run_cell.py"), "--serve", plan["serve"], "--port", str(port), "--proxy-port", str(pport),
                       "--wire-log", wlog, "--out", a.out, "--agent", s["agent"], "--task", s["task"], "--rep", str(s["rep"]),
                       "--timeout", str(c.get("run_timeout", 1200)), "--label", f"config={c['name']}", "--label", f"session={sid}",
                       "--label", f"ctx={c['ctx']}", "--label", f"model={c['model']}"]
                if s.get("prompt"):
                    cmd += ["--prompt", s["prompt"]]
                try:
                    r = subprocess.run(cmd, capture_output=True, text=True)
                    say(f"{sid}: {(r.stdout.strip() or r.stderr.strip())[-400:]}")
                finally:
                    os.killpg(proxy.pid, signal.SIGTERM)
                nreq = len([f for f in os.listdir(bdir)]) if os.path.isdir(bdir) else 0
                with open(sess_path, "a") as f:
                    f.write(json.dumps(dict(id=sid, config=c["name"], model=c["model"], ctx=c["ctx"], agent=s["agent"], task=s["task"], rep=s["rep"],
                                            t_start=t0, t_end=time.time(), bodies=nreq)) + "\n")
                STATE["done"] += 1
        finally:
            try:
                os.killpg(srv.pid, signal.SIGTERM)
                srv.wait(timeout=60)
            except Exception:
                try:
                    os.killpg(srv.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            time.sleep(5)
    say(f"finished: {STATE['done']}/{STATE['total']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
