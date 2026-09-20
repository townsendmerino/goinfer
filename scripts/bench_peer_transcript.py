#!/usr/bin/env python3
"""bench_peer_transcript.py — W4: the agent-turn transcript replay
(docs/tasks/task-peer-benchmarks.md §3/§7).

WHAT IT MEASURES. A scripted N-turn tool-calling conversation is replayed
against goinfer's /v1/chat/completions as strict prefix extensions — each
turn's request is the FULL message list so far, with the previous turn's
actual assistant reply and the fixture's canned tool result appended. Per
turn this records TTFT and how many leading prompt tokens the server's
resident cache reused (usage.prefill_reused_tokens — a goinfer vendor
extension added alongside this harness; see internal/serveapp/openai.go's
`usage` struct). This is the number an agent-loop user (Claude Code,
opencode, ...) actually feels turn to turn, as opposed to the single-shot
throughput numbers the rest of the peer matrix measures.

WHY TWO REQUESTS PER TURN, NOT bench_peer_prefill.py's STREAMING ttft().
Every turn here forces a NAMED tool_choice (see below for why), and
internal/serveapp/tools.go's serveChatToolsWith ALWAYS buffers the whole
generation before sending any SSE content frame once a call is grammar-
constrained from token 1 — there is no prose lead to stream incrementally,
even on an "incremental" chat template family. So "TTFT from the first
streamed content token" is unmeasurable as literally specified for a forced
tool turn; naively adapting bench_peer_prefill.py's ttft() would silently
measure time-to-full-generation instead of time-to-first-token. Fixed with
the same technique that script already uses for the same reason (Engine.
payload's max_tokens=1 comment): each turn issues

  1. PROBE  — max_tokens=1, non-streaming. Wall-clock around the blocking
     POST is this turn's TTFT; usage.prefill_reused_tokens/prompt_tokens
     from the body is this turn's reuse reading.
  2. REAL   — max_tokens=<--max-tokens>, non-streaming. Supplies the actual
     tool_calls reply used to build the NEXT turn's message list.

THE PROBE CONTAMINATES THE REAL REQUEST'S OWN REUSE READING, ON PURPOSE,
HARMLESSLY. The probe's generation commits to the resident cache on normal
exit (decoder/model.go: the commit fires on hitting max_tokens, not only a
clean stop) — so by the time the REAL request for the same turn runs, its
prompt is a 100%-byte match against what the probe just prefilled, and its
own prefill_reused_tokens reads artificially fully-warm. That's fine for
what it's used for (only its tool_calls reply matters) but means only
PROBE.prefill_reused_tokens is this turn's genuine cold/warm signal — the
real request's reuse field is recorded too but labelled non-authoritative.

WHY EVERY TURN FORCES A NAMED tool_choice, INCLUDING THE LAST.
internal/serveapp/openai.go's handleChat routes to the tool-rendering path
(RenderToolsSegments) only when tools is non-empty AND tool_choice != "none";
a turn that answered in free prose instead would silently switch prompt-
rendering code paths, breaking the strict-prefix-extension property this
whole fixture exists to test. A named choice also sidesteps N-18 (`auto`/
`required` doesn't reliably force a call with 2+ tools) since
constrainForcedTool forces reliably on a named choice regardless of tool
count (docs/integrations/claude-code.md's own known-open-issues list).

WHY THE TWO VARIANTS (base / edited_turn6) RUN AS SEPARATE SERVER PROCESSES.
Resident reuse is a single global per-model slot, keyed on raw token
content, not on conversation identity (decoder/resident_reuse.go). Running
edited_turn6 right after base in the same live process would read its
turns 1-5 as "already resident" from the PREVIOUS variant's run rather than
from a fresh session — exactly the cross-cell contamination
bench_peer_prefill.py's own per-cell Engine restart already exists to
avoid. So each variant gets a fresh server.

  GOINFER_SERVE_CUDA=~/bench-cur/serve-cuda \\
    python3 scripts/bench_peer_transcript.py out.json --model 7B --backend cuda

  GOINFER_SERVE_METAL=~/bench-cur/serve-metal \\
    python3 scripts/bench_peer_transcript.py out.json --model 7B --backend metal

R12 (docs/tasks/red-october.md), 2026-09: extended to metal (mirrors bench_peer_prefill.py's own
cuda/metal SERVE-path pattern — no change to the request shape, tool-forcing, or probe/real
two-request-per-turn technique above, all of which are backend-agnostic).
"""
import argparse, json, os, platform, signal, socket, subprocess, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
GPORT = 8098  # same value bench_peer_prefill.py uses; not run concurrently with it
SERVE_CUDA = os.environ.get("GOINFER_SERVE_CUDA", os.path.expanduser("~/bench-cur/serve-cuda"))
SERVE_METAL = os.environ.get("GOINFER_SERVE_METAL", os.path.expanduser("~/bench-cur/serve-metal"))
SERVE = {"cuda": SERVE_CUDA, "metal": SERVE_METAL}
# Only the model cell this pass validates against (docs/tasks/task-peer-benchmarks.md's "D7"
# cell) — add more MODELS entries here if a later pass extends coverage.
MODELS = {
    "7B": (os.path.expanduser("~/models/qwen2.5-7b-instruct-q4_k_m.gguf"), "q7b"),
    # Added for W7 (R12): the 7B cell's concurrent multi-session memory footprint proved unsafe
    # on a 16GB Mac (measured directly -- two near-total memory exhaustions, see the record), so
    # W7's concurrency sweep moved to this smaller cell rather than keep forcing 7B.
    "1.5B": (os.path.expanduser("~/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"), "q15"),
}
# W7 (R12, docs/tasks/red-october.md): the --clients sweep restarts a fresh 7B-model server
# several times in a tight loop, and darwin's reclaim of a just-exited large process's resident
# pages is not instantaneous -- measured directly, a restart that outran it drove this machine
# from ~6 GB free to 58 MB free in under 20s. Refuse to start (or continue) rather than risk it;
# this machine has hit a real kernel panic from exactly this class of over-commitment before.
MIN_FREE_MB_BEFORE_NEXT_SERVER = int(os.environ.get("BENCH_MIN_FREE_MB", "3000"))


def free_mb():
    """Free physical memory in MB (darwin vm_stat; page size 16384 on Apple Silicon). Returns
    None if vm_stat isn't available (non-darwin) -- callers must treat that as "unknown, don't
    gate on it" rather than "zero"."""
    try:
        out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return None
    for line in out.splitlines():
        if line.startswith("Pages free:"):
            pages = int(line.split(":")[1].strip().rstrip("."))
            return pages * 16384 / 1048576
    return None


def wait_free_memory_mb(min_mb, timeout=60):
    """Blocks until free_mb() >= min_mb or timeout elapses, polling every 2s. No-op (returns
    immediately) if free_mb() can't be read at all (non-darwin) -- see free_mb's own note."""
    if free_mb() is None:
        return
    t0 = time.time()
    while time.time() - t0 < timeout:
        mb = free_mb()
        if mb is None or mb >= min_mb:
            return
        print(f"[w7] waiting for memory to recover: {mb:.0f} MB free, want >= {min_mb} MB "
              f"({time.time()-t0:.0f}s/{timeout}s)", file=sys.stderr)
        time.sleep(2)
    print(f"[w7] WARNING: proceeding after {timeout}s without reaching {min_mb} MB free "
          f"(have {free_mb():.0f} MB) -- the next server load may be under real pressure",
          file=sys.stderr)


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
    """A single-engine analogue of bench_peer_prefill.py's Engine — this pass validates
    goinfer alone (docs/tasks/task-peer-benchmarks.md scoped this workload's first landing to
    one engine); a peer arm is future work, not dropped by oversight. cuda and metal both
    supported (R12, docs/tasks/red-october.md); the request/probe shape above is unchanged
    either way, only which binary and -backend flag get launched differs."""

    def __init__(self, model_key, backend="cuda"):
        if model_key not in MODELS:
            raise RuntimeError(f"no MODELS entry for {model_key!r}")
        if backend not in SERVE:
            raise RuntimeError(f"backend {backend!r} not supported — known: {sorted(SERVE)}")
        self.path, self.tag = MODELS[model_key]
        self.backend = backend
        self.proc = None

    def __enter__(self):
        wait_free_memory_mb(MIN_FREE_MB_BEFORE_NEXT_SERVER, timeout=60)
        argv = [SERVE[self.backend], "-model", f"bench={self.path}", "-backend", self.backend,
                "-addr", f"127.0.0.1:{GPORT}", "-quant", "int4"]
        self.proc = subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                      preexec_fn=os.setsid)
        if not wait_port(GPORT):
            raise RuntimeError("goinfer: server did not come up")
        self.url = f"http://127.0.0.1:{GPORT}/v1/chat/completions"
        return self

    def __exit__(self, *a):
        try:
            os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
        except Exception:
            pass
        try:
            self.proc.wait(timeout=45)
        except Exception:
            self.proc.kill()
        # W7 (R12): a fixed 2s "let VRAM settle" (bench_peer_prefill.py's own idiom, fine for its
        # single-server-per-invocation shape) is NOT enough margin for a --clients sweep, which
        # restarts a FRESH 7B-model server several times in a tight loop -- measured directly: a
        # concurrency-level restart that started before the outgoing process's memory was actually
        # reclaimed drove this machine from ~6 GB free to 58 MB free in under 20s (darwin's own
        # reclaim of a large process's resident pages is not instantaneous on SIGTERM). Poll for
        # real headroom instead of trusting a constant.
        wait_free_memory_mb(MIN_FREE_MB_BEFORE_NEXT_SERVER, timeout=60)


def post(url, payload, timeout=120):
    """Blocking, non-streaming POST. Returns (elapsed_seconds, parsed_json_body)."""
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                  headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = json.loads(r.read())
    return time.perf_counter() - t0, body


def named_choice(name):
    return {"type": "function", "function": {"name": name}}


def _reused_fraction(usage):
    p, r = usage.get("prompt_tokens"), usage.get("prefill_reused_tokens")
    if not p:
        return None
    return round((r or 0) / p, 4)


def run_transcript(url, fixture, max_tokens, temperature=0.0, nonce=None):
    """Replays one fixture end to end against one live server. Returns the list
    of per-turn records (see the module docstring for what probe vs real means).

    nonce: prepended to the FIRST user turn only (docs/tasks/task-peer-benchmarks.md's own
    "nonce-prefixed prompts to defeat [the engine's] KV prefix cache" idiom, same one
    bench_peer.py/bench_peer_prefill.py already use). Every later turn is a strict prefix
    extension of turn 1, so a unique first turn is enough to make an entire replay's token
    content unique -- required for run_concurrent (W7): N clients replaying identical fixture
    content concurrently would otherwise let one client's requests warm the cache for another,
    which is not what "independent conversations" means. None for run_transcript's own
    single-conversation (W4) callers, where no such collision exists."""
    messages = []
    out = []
    first_user_turn = True
    for turn in fixture["turns"]:
        if turn["user"] is not None:
            text = turn["user"]
            if nonce is not None and first_user_turn:
                text = f"[{nonce}] {text}"
            first_user_turn = False
            messages.append({"role": "user", "content": text})
        base_payload = {
            "model": "bench", "temperature": temperature, "stream": False,
            "messages": messages, "tools": fixture["tools"],
            "tool_choice": named_choice(turn["tool_choice"]),
        }
        probe_t, probe_body = post(url, {**base_payload, "max_tokens": 1})
        real_t, real_body = post(url, {**base_payload, "max_tokens": max_tokens})

        probe_usage = probe_body.get("usage") or {}
        real_usage = real_body.get("usage") or {}
        choices = real_body.get("choices") or [{}]
        msg = choices[0].get("message") or {}
        calls = msg.get("tool_calls") or [{}]
        call = calls[0] or {}
        fn = call.get("function") or {}

        out.append({
            "n": turn["n"],
            "tool_choice": turn["tool_choice"],
            "probe": {
                "ttft_ms": round(probe_t * 1000, 1),
                "prompt_tokens": probe_usage.get("prompt_tokens"),
                "prefill_reused_tokens": probe_usage.get("prefill_reused_tokens"),
                "reused_fraction": _reused_fraction(probe_usage),
            },
            "real": {
                "latency_ms": round(real_t * 1000, 1),
                "prompt_tokens": real_usage.get("prompt_tokens"),
                "completion_tokens": real_usage.get("completion_tokens"),
                "prefill_reused_tokens": real_usage.get("prefill_reused_tokens"),
                "note": ("post-probe: this request's own prompt was just fully prefilled by "
                         "the probe above, so its prefill_reused_tokens reads artificially "
                         "fully-warm — it is NOT this turn's cold/warm signal. Read probe.* "
                         "for that; see the module docstring."),
            },
            "cache_state": "warm" if (probe_usage.get("prefill_reused_tokens") or 0) > 0 else "cold",
            "assistant_tool_call": {"name": fn.get("name"), "arguments": fn.get("arguments")},
        })

        messages.append(msg)
        messages.append({
            "role": "tool",
            "name": fn.get("name"),
            "tool_call_id": call.get("id"),
            "content": turn["tool_result"],
        })
    return out


def run_concurrent(url, fixture, max_tokens, temperature, n_clients):
    """W7 (docs/tasks/red-october.md, R12 item iv): n_clients independent replays of the SAME
    fixture running CONCURRENTLY against one live server — each client keeps its own message
    history from scratch (genuinely independent conversations, not shared state; goinfer's
    resident reuse is keyed on raw token content, so two clients replaying the identical
    fixture will each see their OWN turns as reused, not each other's — same isolation
    run_transcript's own module docstring already relies on for the base/edited_turn6 split).

    aggregate_tok_s is total completion_tokens summed across every client's every REAL turn,
    divided by the WALL-CLOCK duration of the whole concurrent batch — not the sum of each
    client's own duration, which would double-count the overlapping time and is not what
    "aggregate throughput" means for a concurrency measurement."""
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n_clients) as ex:
        futures = [
            ex.submit(run_transcript, url, fixture, max_tokens, temperature, f"w7-client{i}-{t0}")
            for i in range(n_clients)
        ]
        per_client = [f.result() for f in futures]
    wall_s = time.perf_counter() - t0
    total_completion_tokens = sum(
        (turn["real"]["completion_tokens"] or 0) for client_turns in per_client for turn in client_turns
    )
    return {
        "n_clients": n_clients,
        "wall_s": round(wall_s, 3),
        "total_completion_tokens": total_completion_tokens,
        "aggregate_tok_s": round(total_completion_tokens / wall_s, 2) if wall_s > 0 else None,
        "per_client_ttft_ms": [
            [turn["probe"]["ttft_ms"] for turn in client_turns] for client_turns in per_client
        ],
        "per_client": per_client,
    }


def machine_header(backend="cuda"):
    def sh(cmd):
        try:
            return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=20).stdout.strip()
        except Exception:
            return ""
    # stat's -c (GNU/Linux) vs -f (BSD/darwin) flag differs; try GNU first, fall back to BSD —
    # same idiom bench_peer_prefill.py's own header uses for the same reason.
    serve_path = SERVE.get(backend, SERVE_CUDA)
    serve_mtime = sh("stat -c %%y %s 2>/dev/null" % serve_path) or sh("stat -f %%Sm %s 2>/dev/null" % serve_path)
    return {
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": platform.node(),
        "driver": sh("nvidia-smi --query-gpu=driver_version --format=csv,noheader"),
        "gpu": sh("nvidia-smi --query-gpu=name --format=csv,noheader"),
        "kernel": platform.release(),
        "distro": sh(". /etc/os-release 2>/dev/null && echo $PRETTY_NAME"),
        "goinfer_commit": sh("git -C %s rev-parse --short HEAD" % HERE),
        "goinfer_dirty": bool(sh("git -C %s status --porcelain" % HERE)),
        "serve_mtime": serve_mtime,
        "loadavg": sh("cat /proc/loadavg") or sh("uptime"),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", help="output JSON path")
    ap.add_argument("--model", default="7B", choices=sorted(MODELS))
    ap.add_argument("--backend", default="cuda", choices=sorted(SERVE))
    ap.add_argument("--variants", default="base,edited_turn6",
                     help="comma-separated fixture variant names; each resolves to "
                          "scripts/w4_transcript_<variant>.json")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--clients", default=None,
                     help="W7 (R12): comma-separated concurrent-client counts to sweep, e.g. "
                          "1,2,4 -- n independent replays of the SAME fixture run CONCURRENTLY "
                          "against one live server, reporting aggregate tok/s and per-client "
                          "TTFT. Omit for W4's original single-conversation behavior (unchanged).")
    a = ap.parse_args()

    hdr = machine_header(a.backend)
    results = {}
    for variant in a.variants.split(","):
        fixture_path = os.path.join(HERE, f"w4_transcript_{variant}.json")
        fixture = json.load(open(fixture_path))
        if a.clients:
            levels = [int(n) for n in a.clients.split(",")]
            print(f"[w7] variant={variant!r} turns={len(fixture['turns'])} clients={levels}", file=sys.stderr)
            by_level = {}
            for n in levels:
                # A FRESH server per concurrency level, not one shared across the sweep: every
                # level replays the SAME fixture content, so a shared server would let level 2
                # inherit level 1's now-resident turns (goinfer's reuse is keyed on raw token
                # content, not client identity) and read faster for a reason that has nothing to
                # do with concurrency -- measured directly: a shared-server smoke test read
                # MORE aggregate tok/s at clients=2 than clients=1, backwards for a single-worker
                # server under real contention. Same fresh-server discipline the variant loop
                # above already uses, for the same reason.
                print(f"[w7] variant={variant!r} clients={n} starting fresh server", file=sys.stderr)
                with GoinferServer(a.model, a.backend) as srv:
                    by_level[str(n)] = run_concurrent(srv.url, fixture, a.max_tokens, a.temperature, n)
                r = by_level[str(n)]
                print(f"[w7] variant={variant!r} clients={n}: wall={r['wall_s']}s "
                      f"aggregate={r['aggregate_tok_s']} tok/s", file=sys.stderr)
            results[variant] = by_level
        else:
            print(f"[w4] variant={variant!r} turns={len(fixture['turns'])} — starting fresh server", file=sys.stderr)
            with GoinferServer(a.model, a.backend) as srv:
                results[variant] = run_transcript(srv.url, fixture, a.max_tokens, a.temperature)
        print(f"[{'w7' if a.clients else 'w4'}] variant={variant!r} done", file=sys.stderr)

    out = {
        "header": hdr,
        "model": a.model,
        "backend": a.backend,
        "max_tokens": a.max_tokens,
        "temperature": a.temperature,
        "clients": a.clients,
        "variants": results,
    }
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
