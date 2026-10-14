#!/usr/bin/env python3
"""The I3 matrix as one unattended job: for each configuration, serve the model, prove the sandbox, run its cells, and write
one record per run (records.jsonl) and a progress line every 30 s (count and elapsed; no ETA, the cells differ hugely in cost).

  matrix.py <plan.json> <out dir> [--only <config>[,<config>]] [--deadline-minutes N]

plan.json: {"serve": path, "port": N, "proxy_port": N, "box": "...", "commit": "...", "configs": [{"name", "model", "ctx",
"serve_args": [...], "agents": [...], "tasks": [...], "reps": 1, "label": {...}, "run_timeout": seconds,
"serve": path (optional, replaces the plan's for this configuration), "cell_args": [...] (optional, passed to run_cell.py)}]}.
Resumable: a (config, agent, task, rep) already in records.jsonl is skipped. Nothing here needs a person or a Claude session.
"""
import argparse, json, os, signal, subprocess, sys, threading, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sandbox

T0 = time.time()
STATE = dict(done=0, total=0, current="-", cell_start=time.time(), phase="start")


def say(msg):
    print(f"[{time.strftime('%H:%M:%S')} +{int(time.time() - T0)//60}m] {msg}", flush=True)


def heartbeat():
    while True:
        time.sleep(30)
        say(f"heartbeat: phase={STATE['phase']} cells {STATE['done']}/{STATE['total']} current={STATE['current']} (running {int(time.time() - STATE['cell_start'])} s)")


def wait_ready(port, proc, seconds=600):
    for _ in range(seconds):
        if proc.poll() is not None:
            raise RuntimeError("serve exited: rc %s" % proc.returncode)
        try:
            d = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=3))
            if d.get("data"):
                return d
        except Exception:
            time.sleep(1)
    raise RuntimeError("serve did not come up")


def done_keys(path):
    keys = set()
    if os.path.exists(path):
        for line in open(path):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if not r.get("harness_error") and not r.get("void_halted"):
                keys.add((r["labels"].get("config"), r["agent"], r["task"], r["rep"]))
    return keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("plan"); ap.add_argument("out")
    ap.add_argument("--only", default=""); ap.add_argument("--deadline-minutes", type=float, default=0)
    a = ap.parse_args()
    plan = json.load(open(a.plan))
    os.makedirs(a.out, exist_ok=True)
    records = os.path.join(a.out, "records.jsonl")
    deadline = T0 + a.deadline_minutes * 60 if a.deadline_minutes else None
    only = set(filter(None, a.only.split(",")))
    cfgs = [c for c in plan["configs"] if not only or c["name"] in only]
    STATE["total"] = sum(len(c["agents"]) * len(c["tasks"]) * c.get("reps", 1) for c in cfgs)
    threading.Thread(target=heartbeat, daemon=True).start()
    port, pport = plan["port"], plan["proxy_port"]
    say(f"swap guard threshold {plan.get('swap_guard_mb', 512)} MB (GOINFER_SWAP_GUARD)")
    say(f"matrix: {len(cfgs)} configurations, {STATE['total']} runs; goinfer {plan.get('commit')} on {plan.get('box')}")
    tested = False
    skipped = 0
    for c in cfgs:
        STATE["phase"] = f"serve {c['name']}"
        logdir = os.path.join(a.out, "logs"); os.makedirs(logdir, exist_ok=True)
        serve = c.get("serve", plan["serve"])
        argv = [serve, "--model", c["model"], "--backend", "cuda", "--ctx", str(c["ctx"]), "--addr", f"127.0.0.1:{port}", "--log-requests"] + c.get("serve_args", [])
        wirelog = os.path.join(a.out, f"wire-{c['name']}.jsonl")
        procs = []

        def up():
            slog = open(os.path.join(logdir, f"serve-{c['name']}.log"), "a")
            srv = subprocess.Popen(argv, stdout=slog, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True,
                                   env={**os.environ, "GOINFER_SWAP_GUARD": str(plan.get("swap_guard_mb", 512))})
            proxy = subprocess.Popen([sys.executable, os.path.join(HERE, "wire_proxy.py"), str(pport), str(port), wirelog],
                                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
            procs[:] = [proxy, srv]
            t = time.time()
            wait_ready(port, srv)
            say(f"config {c['name']}: server up in {time.time() - t:.0f} s: {' '.join(argv[1:])}")

        def down():
            for p in procs:
                try:
                    os.killpg(p.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            for p in procs:
                try:
                    p.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid, signal.SIGKILL)
            time.sleep(5)

        try:
            up()
            if not tested:
                STATE["phase"] = "sandbox self-test"
                ok, rep = sandbox.selftest(port, verbose=False)
                open(os.path.join(a.out, "sandbox-selftest.txt"), "w").write(rep)
                if not ok:
                    say("SANDBOX SELFTEST FAILED: no agent will run.\n" + rep)
                    return 3
                say("sandbox self-test passed")
                ok, rep = sandbox.selftest_mutants(port)
                open(os.path.join(a.out, "sandbox-mutants.txt"), "w").write(rep)
                if not ok:
                    say("the sandbox self-test did not catch a planted weakness: no agent will run.\n" + rep)
                    return 3
                say("sandbox self-test catches its planted weaknesses")
                tested = True
            smi = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
            say(f"config {c['name']}: VRAM after load {smi}")
            for agent in c["agents"]:
                for task in c["tasks"]:
                    for rep in range(1, c.get("reps", 1) + 1):
                        STATE.update(phase=f"run {c['name']}", current=f"{agent}/{task}/r{rep}", cell_start=time.time())
                        if (c["name"], agent, task, rep) in done_keys(records):
                            skipped += 1; STATE["done"] += 1
                            continue
                        if deadline and time.time() > deadline:
                            say(f"deadline: not run: {c['name']} {agent} {task} r{rep}")
                            STATE["done"] += 1
                            continue
                        cmd = [sys.executable, os.path.join(HERE, "run_cell.py"), "--serve", serve, "--port", str(port), "--proxy-port", str(pport),
                               "--wire-log", os.path.join(a.out, f"wire-{c['name']}.jsonl"), "--out", a.out, "--agent", agent, "--task", task, "--rep", str(rep),
                               "--timeout", str(c.get("run_timeout", 1500)), "--label", f"config={c['name']}", "--label", f"ctx={c['ctx']}",
                               "--label", f"box={plan.get('box')}", "--label", f"goinfer={plan.get('commit')}", "--label", f"model={c['model']}"]
                        for k, v in (c.get("label") or {}).items():
                            cmd += ["--label", f"{k}={v}"]
                        cmd += c.get("cell_args", [])
                        for attempt in range(3):
                            r = subprocess.run(cmd, capture_output=True, text=True)
                            say(f"{c['name']} {agent} {task} r{rep}: {r.stdout.strip()[-300:] or r.stderr.strip()[-300:]}")
                            last = json.loads(open(records).read().strip().splitlines()[-1])
                            if not last.get("void_halted"):
                                break
                            say("the swap guard halted goinfer during that run: restarting the server and running the cell again")
                            down(); up()
                        STATE["done"] += 1
        finally:
            down()
    say(f"matrix finished: {STATE['done']}/{STATE['total']} (skipped as already recorded: {skipped})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
