#!/usr/bin/env python3
"""bench_j6_scheduling.py — J6: prefix-aware admission, the concurrent-load gate
(docs/tasks/task-work-queue-2026-09.md J6; pre-registered band in
docs/measurements/j6-prefix-scheduling-PREREGISTERED.md, written before any run).

WHAT IT MEASURES. Neither bench_compare.sh (in-process Go microbenchmarks, no HTTP, no
concurrency) nor bench_peer.py/bench_peer_transcript.py (real HTTP, but one request at a time,
server restarted between cells) can drive concurrent admission-queue contention — this is a new,
narrowly-scoped harness for exactly that.

M independent simulated agent-loop conversations run CONCURRENTLY against one `serve` process.
Each conversation is its own SEQUENTIAL loop (request turn N+1 only after turn N's reply) — that
is what an agent loop actually does — but M of them run at once, so at any moment several
conversations' "next turn" requests may be queued together waiting for the single decode worker.
-kv-sessions is set BELOW M on purpose: with fewer resident-session slots than concurrent
conversations, there is genuine LRU eviction pressure, and the two admission policies (FIFO vs
J6's prefix-aware pick) can produce a genuinely different outcome — which waiter is served next
affects both service order and which sessions survive eviction. Every conversation shares one long
system-prompt prefix (forked from scripts/w4_transcript_base.json's 12 real tool schemas, rendered
as prose — see build_system_prompt below); each conversation's own turn content diverges (the
"divergent tails" the task doc names).

A/B is via TWO BINARIES, matching bench_peer.py's own established BENCH_ENGINES=goinfer,
goinfer_old pattern (docs/benchmarks.md:193-201) rather than a new runtime flag — J6's own ground
rule 5 ("every throughput claim is measured before it ships") means the mechanism is not the new
silent default until this gate passes; the old and new binaries are literally two different
commits, run interleaved (session-to-session drift is ~3.5% on this box per this repo's own
measurement notes, so non-interleaved arms would silently absorb it into the ratio).

Usage:
  python3 scripts/bench_j6_scheduling.py out.json \
      --old ~/j6-bench/serve-old --new ~/j6-bench/serve-new \
      --model ~/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf \
      --conversations 6 --kv-sessions 4 --turns 4 --reps 3 --max-tokens 64
"""
import argparse, json, os, platform, random, signal, socket, subprocess, sys, threading, time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
PORT = 8099  # not run concurrently with bench_peer_prefill.py (8098) or bench_peer_transcript.py

BUGS = [
    ("internal/ratelimit/limiter.go", "the rate limiter isn't throttling bursts correctly"),
    ("internal/cache/lru.go", "the LRU cache evicts the wrong entry under concurrent access"),
    ("internal/queue/worker.go", "the worker pool deadlocks under high load"),
    ("internal/auth/session.go", "sessions expire early under clock skew"),
    ("internal/retry/backoff.go", "the exponential backoff never resets after a success"),
    ("internal/metrics/counter.go", "the counter drops updates under contention"),
]


def build_system_prompt():
    """A realistic ~600-token shared system prompt, forked from w4_transcript_base.json's 12 real
    tool schemas (already-built fixture, not reinvented) rendered as prose rather than a `tools`
    API field — see this file's own module docstring for why: the plain-chat path never renders
    `tools` into the prompt unless tool-calling is actually active, so prose is what actually
    creates a shared prefix on the path this benchmark exercises."""
    fixture = os.path.join(HERE, "w4_transcript_base.json")
    with open(fixture) as f:
        tools = json.load(f)["tools"]
    lines = ["You are a careful coding agent working in a large Go monorepo. You have access to "
             "the following tools (described here for context; call them by name when needed):"]
    for t in tools:
        fn = t["function"]
        props = ", ".join(fn["parameters"].get("properties", {}).keys())
        lines.append(f"- {fn['name']}({props}): {fn['description']}")
    lines.append(
        "Always read a file before editing it. Prefer the smallest correct change. Explain your "
        "reasoning briefly before proposing an edit. Never invent a file path you have not seen. "
        "When a bug report names a file, start by reading exactly that file before looking anywhere "
        "else, since the report is usually accurate about where the symptom lives even when the "
        "root cause is elsewhere. State your hypothesis before proposing a fix.")
    return "\n".join(lines)


SYSTEM_PROMPT = build_system_prompt()


def wait_port(port, timeout=240):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with socket.create_connection(("127.0.0.1", port), 1):
                return True
        except OSError:
            time.sleep(0.4)
    return False


class GoinferServer:
    """Launches one `serve` binary with -kv-sessions set below the conversation count on purpose
    (see module docstring) — CPU backend only this pass (no GPU-specific complexity for an
    admission-ordering question that has nothing to do with the backend)."""

    def __init__(self, binary, model, kv_sessions, max_queue):
        self.binary, self.model, self.kv_sessions, self.max_queue = binary, model, kv_sessions, max_queue
        self.proc = None

    def __enter__(self):
        argv = [self.binary, "-model", f"bench={self.model}", "-backend", "cpu",
                "-addr", f"127.0.0.1:{PORT}", "-quant", "int8int8",
                "-kv-sessions", str(self.kv_sessions), "-max-queue", str(self.max_queue)]
        self.proc = subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                      preexec_fn=os.setsid)
        if not wait_port(PORT):
            raise RuntimeError(f"goinfer ({self.binary}): server did not come up")
        time.sleep(1)  # let the model finish loading past the point the port opens
        self.url = f"http://127.0.0.1:{PORT}/v1/chat/completions"
        return self

    def __exit__(self, *a):
        try:
            os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
        except Exception:
            pass
        try:
            self.proc.wait(timeout=30)
        except Exception:
            self.proc.kill()
        time.sleep(1)


def post(url, payload, timeout=120):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                  headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def run_conversation(url, conv_id, bug, turns, max_tokens, results, errors):
    """One agent-loop conversation: sequential turns, each waiting for the previous reply — the
    per-conversation ordering an M-concurrent-conversation benchmark must preserve."""
    path, symptom = bug
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"There's a bug: {symptom}. Start by reading {path}."}]
    total_tokens = 0
    try:
        for turn in range(turns):
            body = json.loads(json.dumps({
                "model": "bench", "temperature": 0, "max_tokens": max_tokens,
                "messages": messages,
            }))
            resp = post(url, body)
            reply = resp["choices"][0]["message"]["content"]
            usage = resp.get("usage", {})
            total_tokens += usage.get("completion_tokens", 0)
            messages.append({"role": "assistant", "content": reply})
            if turn < turns - 1:
                messages.append({"role": "user", "content":
                                  f"Turn {turn+2}: now check whether the fix also handles the "
                                  f"case where conversation {conv_id} triggers it concurrently."})
    except Exception as e:
        errors.append(f"conv {conv_id}: {e}")
        return
    results.append(total_tokens)


def run_cell(url, n_conversations, turns, max_tokens):
    """Fires all N conversations concurrently (threads — these are blocking HTTP calls, not
    CPU-bound, so the GIL is not a bottleneck), each its own sequential turn loop. Returns
    (wall_seconds, aggregate_completion_tokens, errors)."""
    results, errors = [], []
    threads = []
    t0 = time.perf_counter()
    for i in range(n_conversations):
        bug = BUGS[i % len(BUGS)]
        th = threading.Thread(target=run_conversation,
                               args=(url, i, bug, turns, max_tokens, results, errors))
        threads.append(th)
        th.start()
    for th in threads:
        th.join()
    wall = time.perf_counter() - t0
    return wall, sum(results), errors


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        return ""


def loadavg():
    return sh("sysctl -n vm.loadavg 2>/dev/null || cat /proc/loadavg 2>/dev/null")


def machine_header():
    return {
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": platform.node(),
        "platform": platform.platform(),
        "goinfer_commit": sh(f"git -C {HERE}/.. rev-parse --short HEAD"),
        "goinfer_dirty": bool(sh(f"git -C {HERE}/.. status --porcelain")),
        "loadavg": loadavg(),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", help="output JSON path")
    ap.add_argument("--old", required=True, help="goinfer serve binary WITHOUT J6 (today's FIFO)")
    ap.add_argument("--new", required=True, help="goinfer serve binary WITH J6 (prefix-aware)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--conversations", type=int, default=6, help="M — concurrent agent loops")
    ap.add_argument("--kv-sessions", type=int, default=4, help="must be < --conversations for real contention")
    ap.add_argument("--max-queue", type=int, default=32)
    ap.add_argument("--turns", type=int, default=4, help="sequential turns per conversation")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--reps", type=int, default=3, help="paired reps per arm, interleaved")
    ap.add_argument("--starvation-bounds", default="", help="comma-separated GOINFER_TEST_* sweep — not yet wired, placeholder for the overnight sweep")
    args = ap.parse_args()

    if args.kv_sessions >= args.conversations:
        print(f"WARNING: --kv-sessions ({args.kv_sessions}) >= --conversations ({args.conversations})"
              f" — no LRU eviction pressure, the two policies may not differ at all", file=sys.stderr)

    header = machine_header()
    header["args"] = vars(args)
    print(json.dumps(header, indent=2), file=sys.stderr)

    arms = {"old_fifo": args.old, "new_prefix_aware": args.new}
    cells = []
    # Interleaved: rep 0 of old, rep 0 of new, rep 1 of old, rep 1 of new, ... — never all of one
    # arm before the other (this repo's own paired-differencing rule, docs/benchmarks.md).
    for rep in range(args.reps):
        for name, binary in arms.items():
            load_before = loadavg()
            print(f"=== rep {rep} arm {name} === loadavg {load_before}", file=sys.stderr)
            with GoinferServer(binary, args.model, args.kv_sessions, args.max_queue) as srv:
                wall, tokens, errors = run_cell(srv.url, args.conversations, args.turns, args.max_tokens)
            load_after = loadavg()
            if errors:
                print(f"  {len(errors)} conversation error(s): {errors[:3]}", file=sys.stderr)
            tps = tokens / wall if wall > 0 else 0.0
            print(f"  wall={wall:.2f}s tokens={tokens} agg_tok/s={tps:.2f} errors={len(errors)} loadavg_after={load_after}",
                  file=sys.stderr)
            cells.append({"rep": rep, "arm": name, "wall_s": wall, "tokens": tokens,
                           "agg_tok_s": tps, "errors": errors,
                           "loadavg_before": load_before, "loadavg_after": load_after})
            # Write incrementally so an interrupted run still leaves usable partial data —
            # the 24-rep VOID run this replaces (2026-09-15, spread 36.8%/59.7% from periodic
            # interference) was only diagnosable after the fact BECAUSE the interim log survived;
            # this makes the structured JSON itself survive an interruption the same way.
            with open(args.out, "w") as f:
                json.dump({"header": header, "cells": cells}, f, indent=2)

    print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
