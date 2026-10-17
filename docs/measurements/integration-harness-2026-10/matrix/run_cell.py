#!/usr/bin/env python3
"""One real-harness run (I3): a fresh fixture copy and HOME, the harness's configuration written by `goinfer-serve
connect -write`, one headless task inside the sandbox, the result checked mechanically, and everything I3 asks to
have recorded written as one JSON record.

  run_cell.py --serve <serve binary> --port <goinfer port> --proxy-port <port> --wire-log <file> --out <dir> \
              --agent claude-code|opencode|codex --task read|search|edit|write|fix [--rep 1] [--timeout 1200]
              [--label k=v ...]   (model, quant, ctx, backend, box ... recorded as given)

The goinfer server and the wire proxy (wire_proxy.py <proxy port> <goinfer port> <wire log>) are the caller's.
"""
import argparse, json, os, re, shutil, subprocess, sys, threading, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sandbox, tasks
from agents import AGENTS, RO

MAIN_ROUTES = ("/v1/chat/completions", "/v1/messages", "/v1/responses")


def sh(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


class Vram(threading.Thread):
    """Peak GPU memory in use while a run is going (nvidia-smi, every 2 s)."""
    def __init__(self):
        super().__init__(daemon=True)
        self.peak, self.stop = 0, threading.Event()

    def run(self):
        while not self.stop.is_set():
            r = sh(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"])
            try:
                self.peak = max(self.peak, int(r.stdout.split()[0]))
            except (ValueError, IndexError):
                pass
            self.stop.wait(2)


def wire_records(path, t0, t1):
    out = []
    if os.path.exists(path):
        for line in open(path):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if t0 <= r.get("t", 0) <= t1 + 5:
                out.append(r)
    return out


def overflow_tokens(r):
    """The prompt size an overflow answer names: a 400's text ("resulted in N tokens", "too long: N tokens") or response.failed's usage."""
    m = re.search(r"resulted in ([\d,]+) tokens|too long: ([\d,]+) tokens", r.get("error_body") or "")
    if m:
        return int((m.group(1) or m.group(2)).replace(",", ""))
    return (r.get("usage") or {}).get("input_tokens") if "response.failed" in (r.get("events") or {}) else None


def summarize_wire(recs):
    main = [r for r in recs if r.get("method") == "POST" and r.get("path", "").split("?")[0] in MAIN_ROUTES]
    gen = [r for r in main if (r.get("tools") or 0) > 0] or main
    tok = lambda r: (lambda u: u.get("prompt_tokens", u["input_tokens"] + u.get("cache_read_input_tokens", 0) if "input_tokens" in u else None))(r.get("usage") or {})
    def overflow(r):
        b = (r.get("error_body") or "").lower()
        return r.get("status") == 400 and any(k in b for k in ("too long", "context_length", "maximum context", "shortening the conversation")) \
            or "response.failed" in (r.get("events") or {})
    prompts = [tok(r) for r in gen if tok(r)]
    # A compaction as the wire shows it: the next request's prompt is under 60% of the largest so far.
    wire_compactions, peak = 0, 0
    for p in prompts:
        if peak and p < 0.6 * peak:
            wire_compactions += 1
            peak = p
        peak = max(peak, p)
    dropped = sorted({v for r in recs for k, v in (r.get("goinfer_headers") or {}).items() if k.lower() == "x-goinfer-tools-dropped"})
    return dict(
        requests=len(recs), generation_requests=len(gen), side_requests=len(main) - len(gen),
        n_halted=sum(1 for r in recs if r.get("status") == 503 and '"halted"' in (r.get("error_body") or "")),
        n_429=sum(1 for r in recs if r.get("status") == 429), n_5xx=sum(1 for r in recs if (r.get("status") or 0) >= 500),
        n_other_4xx=sum(1 for r in recs if 400 <= (r.get("status") or 0) < 500 and r.get("status") != 429 and not overflow(r)),
        overflow_errors=sum(1 for r in main if overflow(r)), wire_compactions=wire_compactions, tools_dropped=dropped,
        per_request=[dict(path=r["path"].split("?")[0], status=r.get("status"), prompt_tokens=tok(r), completion_tokens=(r.get("usage") or {}).get("completion_tokens", (r.get("usage") or {}).get("output_tokens")),
                          ttfb=r.get("ttfb"), seconds=r.get("t_end"), tools=r.get("tools"), max_out=r.get("max_out"), overflow=overflow(r) or None,
                          overflow_tokens=overflow_tokens(r)) for r in main])


def chain_trace(key, events, stdout):
    """What the agent did with I1-v2's chain, in its own order: ["read", i] for a tool call naming chain file i (0 is start),
    ["decoy"] for one naming a decoy, ["compact"]; and whether any decoy's lines reached the agent's output."""
    idx = {n: i for i, n in enumerate(key["chain"])}
    decoys = set(key["decoys"])
    seq = []
    for e in events:
        if e[0] == "compact":
            seq.append(["compact"])
            continue
        for n in re.findall(r"data/([0-9a-f]{6}|start)\.txt", e[1]):
            seq.append(["read", idx[n]] if n in idx else ["decoy"] if n in decoys else ["other"])
    return dict(seq=seq, last=len(key["chain"]) - 1, decoy_content=any(f"{d} record " in stdout for d in decoys))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--serve", required=True); ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--proxy-port", type=int, required=True); ap.add_argument("--wire-log", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--agent", required=True, choices=sorted(AGENTS))
    ap.add_argument("--task", required=True, choices=sorted(tasks.TASKS)); ap.add_argument("--rep", type=int, default=1)
    ap.add_argument("--timeout", type=int, default=1200); ap.add_argument("--label", action="append", default=[])
    ap.add_argument("--prompt", help="override the task's prompt (overflow runs); the task's check still applies when it has one")
    ap.add_argument("--env", action="append", default=[], help="K=V added to the harness's environment (a deviation, recorded in the labels)")
    ap.add_argument("--keep", action="store_true", help="keep the run directory (always kept next to the record)")
    ap.add_argument("--told-window", type=int, default=0, help="I1-v2's measurement configuration: tell the agent this window instead of the one connect wrote")
    ap.add_argument("--expect-output-cap", type=int, default=0, help="refuse the run unless the configuration's output cap, read back, is this")
    a = ap.parse_args()
    agent, task = AGENTS[a.agent], tasks.TASKS[a.task]
    rid = f"{a.agent}-{a.task}-r{a.rep}-{time.strftime('%H%M%S')}"
    run = os.path.join(a.out, "runs", rid)
    work, home = os.path.join(run, "work"), os.path.join(run, "home")
    tasks.copy_fixture(work); os.makedirs(home)
    if task.get("setup"):
        task["setup"](work)
    fw, sock = sandbox.start_forwarder(a.proxy_port)
    rec = dict(extra_env=a.env, id=rid, agent=a.agent, task=a.task, rep=a.rep, labels=dict(kv.split("=", 1) for kv in a.label), started=time.strftime("%F %T %Z"))
    try:
        base = dict(sock=sock, work=work, home=home, port=a.port, ro_extra=RO)
        g = sandbox.run(**base, cmd=["sh", "-c", "git init -q && git add -A && git commit -qm fixture"], timeout=60)
        if g[0] != 0:
            raise RuntimeError("git init in the sandbox: " + g[2][-300:])
        if a.agent == "opencode":   # where opencode looks for the ripgrep it would otherwise download
            os.makedirs(os.path.join(home, ".cache", "opencode", "bin"))
            shutil.copy(os.path.join(sandbox.TOOLS, "rg"), os.path.join(home, ".cache", "opencode", "bin", "rg"))
        cfg = agent.config_path(home, work)
        c = sh([a.serve, "connect", agent.connect_name, "-write", "-file", cfg, f"http://127.0.0.1:{a.port}"])
        if c.returncode != 0:
            raise RuntimeError("connect: " + c.stdout + c.stderr)
        rec["connect"] = c.stdout.splitlines()[:2]
        told = agent.tell_window(cfg, a.told_window) if a.told_window else agent.tell_window(cfg, int(rec["labels"].get("ctx") or 0)) if a.expect_output_cap else None
        if told is not None:
            rec["told"] = told
            if a.told_window and told["window"] != a.told_window:
                raise RuntimeError(f"told window: the file says {told['window']}, not {a.told_window}")
            if a.expect_output_cap and told["output_cap"] != a.expect_output_cap:
                raise RuntimeError(f"output cap: the file says {told['output_cap']}, not {a.expect_output_cap}")
        model = next((l.split("model ")[1].split(",")[0] for l in c.stdout.splitlines() if "model " in l), "")
        argv = agent.argv(a.prompt or task["prompt"]) if a.agent != "opencode" else agent.argv(a.prompt or task["prompt"], f"goinfer/{model}")
        vram = Vram(); vram.start()
        t0 = time.time()
        rc, so, se, dt = sandbox.run(**base, cmd=argv, env={**agent.env("/home/agent"), **dict(kv.split("=", 1) for kv in a.env)}, timeout=a.timeout)
        t1 = time.time()
        vram.stop.set()
        open(os.path.join(run, "agent.stdout"), "w").write(so); open(os.path.join(run, "agent.stderr"), "w").write(se)
        p = agent.parse(so, se, home)
        rec.update(exit=rc, wall_seconds=round(dt, 1), vram_peak_mib=vram.peak, **{k: p[k] for k in ("turns", "tool_calls", "tool_errors", "compactions", "errors", "ended_on_error")})
        rec["answer"] = p["answer"][:300]
        rec["timed_out"] = rc == "timeout"
        rec["wire"] = summarize_wire(wire_records(a.wire_log, t0, t1))
        # goinfer's swap guard refused requests (503 halted): the run measured the box, not the harness, so it is voided and redone
        rec["void_halted"] = rec["wire"]["n_halted"] > 0 or None
        # the mechanical check, in the sandbox, on what the model left
        if task.get("keyed"):
            key = json.load(open(os.path.join(run, "chain-key.json")))
            rec["chain"] = chain_trace(key, p.get("events", []), so)
            rec["context_window_seen"] = p.get("context_window_seen")
            rec["passed"] = key["code"] in (p["answer"] or "")
        elif "check" in task:
            k = sandbox.run(**base, cmd=["sh", "-c", task["check"]], timeout=300)
            rec["check"] = dict(rc=k[0], out=(k[1] + k[2])[-400:])
            rec["passed"] = k[0] == 0
        else:
            # The agent's own reply only: the raw stream holds ids and tool output, where "5" or "store.go" match falsely.
            rec["passed"] = bool(task["answer"](p["answer"]))
        rec["diff"] = sandbox.run(**base, cmd=["sh", "-c", "git add -A -N && git diff --stat | tail -5"], timeout=60)[1]
    except Exception as e:
        rec["harness_error"] = f"{type(e).__name__}: {e}"
        rec["passed"] = False
    finally:
        fw.terminate()
    rec["finished"] = time.strftime("%F %T %Z")
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "records.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps({k: rec.get(k) for k in ("id", "passed", "wall_seconds", "turns", "tool_calls", "tool_errors", "compactions", "ended_on_error", "errors", "answer", "harness_error")}), flush=True)
    return 0 if rec["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
