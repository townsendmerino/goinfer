#!/usr/bin/env python3
"""bench_peer_transcript.py — W4: the agent-turn transcript replay
(docs/task-peer-benchmarks.md §3/§7).

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
"""
import argparse, json, os, platform, signal, socket, subprocess, sys, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
GPORT = 8098  # same value bench_peer_prefill.py uses; not run concurrently with it
SERVE_CUDA = os.environ.get("GOINFER_SERVE_CUDA", os.path.expanduser("~/bench-cur/serve-cuda"))
# Only the model cell this pass validates against (docs/task-peer-benchmarks.md's "D7"
# cell) — add more MODELS entries here if a later pass extends coverage.
MODELS = {
    "7B": (os.path.expanduser("~/models/qwen2.5-7b-instruct-q4_k_m.gguf"), "q7b"),
}


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
    """A single-engine, CUDA-only analogue of bench_peer_prefill.py's Engine —
    this pass validates goinfer alone (docs/task-peer-benchmarks.md scoped this
    workload's first landing to one engine, one box); a peer arm is future work,
    not dropped by oversight."""

    def __init__(self, model_key, backend="cuda"):
        if model_key not in MODELS:
            raise RuntimeError(f"no MODELS entry for {model_key!r}")
        if backend != "cuda":
            raise RuntimeError(f"backend {backend!r} not validated for W4 yet — cuda only this pass")
        self.path, self.tag = MODELS[model_key]
        self.backend = backend
        self.proc = None

    def __enter__(self):
        argv = [SERVE_CUDA, "-model", f"bench={self.path}", "-backend", "cuda",
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
        time.sleep(2)  # let VRAM settle before the next variant loads (bench_peer_prefill.py idiom)


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


def run_transcript(url, fixture, max_tokens, temperature=0.0):
    """Replays one fixture end to end against one live server. Returns the list
    of per-turn records (see the module docstring for what probe vs real means)."""
    messages = []
    out = []
    for turn in fixture["turns"]:
        if turn["user"] is not None:
            messages.append({"role": "user", "content": turn["user"]})
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


def machine_header():
    def sh(cmd):
        try:
            return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=20).stdout.strip()
        except Exception:
            return ""
    return {
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": platform.node(),
        "driver": sh("nvidia-smi --query-gpu=driver_version --format=csv,noheader"),
        "gpu": sh("nvidia-smi --query-gpu=name --format=csv,noheader"),
        "kernel": platform.release(),
        "distro": sh(". /etc/os-release 2>/dev/null && echo $PRETTY_NAME"),
        "goinfer_commit": sh("git -C %s rev-parse --short HEAD" % HERE),
        "goinfer_dirty": bool(sh("git -C %s status --porcelain" % HERE)),
        "serve_mtime": sh("stat -c %%y %s 2>/dev/null" % SERVE_CUDA),
        "loadavg": sh("cat /proc/loadavg"),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", help="output JSON path")
    ap.add_argument("--model", default="7B", choices=sorted(MODELS))
    ap.add_argument("--backend", default="cuda", choices=["cuda"])
    ap.add_argument("--variants", default="base,edited_turn6",
                     help="comma-separated fixture variant names; each resolves to "
                          "scripts/w4_transcript_<variant>.json")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0)
    a = ap.parse_args()

    hdr = machine_header()
    results = {}
    for variant in a.variants.split(","):
        fixture_path = os.path.join(HERE, f"w4_transcript_{variant}.json")
        fixture = json.load(open(fixture_path))
        print(f"[w4] variant={variant!r} turns={len(fixture['turns'])} — starting fresh server", file=sys.stderr)
        with GoinferServer(a.model, a.backend) as srv:
            results[variant] = run_transcript(srv.url, fixture, a.max_tokens, a.temperature)
        print(f"[w4] variant={variant!r} done", file=sys.stderr)

    out = {
        "header": hdr,
        "model": a.model,
        "backend": a.backend,
        "max_tokens": a.max_tokens,
        "temperature": a.temperature,
        "variants": results,
    }
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
