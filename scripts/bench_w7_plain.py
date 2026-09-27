#!/usr/bin/env python3
"""bench_w7_plain.py — W7 (docs/tasks/red-october.md R12 item iv), simplified: goinfer's single
worker vs llama-server's slots under N concurrent conversations, on plain multi-turn chat (no
tool-calling).

WHY NOT THE W4 TRANSCRIPT FIXTURE (scripts/w4_transcript_*.json, bench_peer_transcript.py). That
fixture forces a named tool_choice every turn, which needs a model whose chat template AND
training actually support tool-calling. The memory-safe local model (qwen2.5-coder-1.5b) does
not reliably satisfy llama-server's tool-call parser even with the real Qwen tool-template
supplied directly from a sibling checkpoint's own GGUF metadata (verified directly: the model
emits the right JSON content but not the <tool_call> wrapper tags the parser requires) — a model-
capability gap, not a config problem. The 7B instruct model does satisfy it, but caused two near-
total memory exhaustions on this machine earlier in this session under concurrent load (see
bench_peer_transcript.py's own MODELS dict comment). This script sidesteps both: plain multi-turn
text, no tools, on the 1.5B model — it answers the core W7 question (concurrent decode throughput,
goinfer's single worker vs llama-server's per-client slots) without either blocker, at the cost of
not being the exact tool-calling shape the original W7 definition specified.

Same memory-safety and isolation discipline as bench_peer_transcript.py: wait_free_memory_mb
before every server (re)start, a FRESH server per (engine, concurrency level) so no level inherits
a warmer cache than clients=1 saw, and a nonce prefixed onto each client's first turn so N clients
replaying the same fixture don't share cache credit.

  python3 scripts/bench_w7_plain.py out.json --clients 1,2,4

MC0 of docs/tasks/task-concurrency-2026-09.md (added 2026-09-26): every turn now records
usage.prefill_reused_tokens (goinfer's vendor extension; absent from llama-server), and
--engines / --backend select goinfer-only runs and the CPU control. `--backend cpu` records under
the engine key "goinfer_cpu", so a Metal and a CPU sweep can share one results file.

  python3 scripts/bench_w7_plain.py mc0.json --clients 1,2 --engines goinfer --backend metal
  python3 scripts/bench_w7_plain.py mc0.json --clients 1,2 --engines goinfer --backend cpu
"""
import argparse, json, os, platform, shlex, signal, socket, subprocess, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
GPORT = 8098
LPORT = 8097
SERVE_CPU_METAL = os.environ.get("GOINFER_SERVE_CPU", os.path.expanduser("~/bench-cur/serve-metal"))
LLAMA_BIN = os.environ.get("LLAMA_SERVER_BIN", "llama-server")
MODEL_PATH = os.path.expanduser(
    os.environ.get("BENCH_W7_MODEL", "~/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"))

# Same guard bench_peer_transcript.py carries, for the same reason (see that file's own
# comment): a --clients sweep restarts servers in a loop, and darwin's reclaim of a just-exited
# process's resident pages is not instantaneous.
MIN_FREE_MB_BEFORE_NEXT_SERVER = int(os.environ.get("BENCH_MIN_FREE_MB", "3000"))


def free_mb():
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
    if free_mb() is None:
        return
    t0 = time.time()
    while time.time() - t0 < timeout:
        mb = free_mb()
        if mb is None or mb >= min_mb:
            return
        print(f"[w7-plain] waiting for memory: {mb:.0f} MB free, want >= {min_mb} MB "
              f"({time.time()-t0:.0f}s/{timeout}s)", file=sys.stderr)
        time.sleep(2)
    print(f"[w7-plain] WARNING: proceeding after {timeout}s without reaching {min_mb} MB free "
          f"(have {free_mb():.0f} MB)", file=sys.stderr)


def wait_port(port, timeout=240):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with socket.create_connection(("127.0.0.1", port), 1):
                return True
        except OSError:
            time.sleep(0.4)
    return False


def wait_llama_health(port, timeout=240):
    """llama-server accepts TCP connections as soon as it starts, well before the model finishes
    loading -- a request in that window gets HTTP 503 {"error":{"message":"Loading model",...}}
    (confirmed directly). wait_port alone is not readiness for this server; poll /health for the
    real {"status":"ok"} it returns once loaded."""
    if not wait_port(port, timeout):
        return False
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


class GoinferServer:
    def __init__(self, backend="metal", extra=""):
        self.backend = backend
        self.extra = shlex.split(extra)

    def __enter__(self):
        wait_free_memory_mb(MIN_FREE_MB_BEFORE_NEXT_SERVER, timeout=60)
        argv = [SERVE_CPU_METAL, "-model", f"bench={MODEL_PATH}", "-backend", self.backend,
                "-addr", f"127.0.0.1:{GPORT}", "-quant", "int4"] + self.extra
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
        wait_free_memory_mb(MIN_FREE_MB_BEFORE_NEXT_SERVER, timeout=60)


class LlamaServer:
    def __init__(self, n_slots):
        self.n_slots = n_slots

    def __enter__(self):
        wait_free_memory_mb(MIN_FREE_MB_BEFORE_NEXT_SERVER, timeout=60)
        argv = [LLAMA_BIN, "-m", MODEL_PATH, "--port", str(LPORT), "-np", str(self.n_slots),
                "-cb", "--host", "127.0.0.1", "-c", str(4096 * self.n_slots)]
        self.proc = subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                      preexec_fn=os.setsid)
        if not wait_llama_health(LPORT):
            raise RuntimeError("llama-server: server did not come up")
        self.url = f"http://127.0.0.1:{LPORT}/v1/chat/completions"
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
        wait_free_memory_mb(MIN_FREE_MB_BEFORE_NEXT_SERVER, timeout=60)


def post(url, payload, timeout=120):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                  headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = json.loads(r.read())
    return time.perf_counter() - t0, body


# A small, fixed set of plain user turns — no tools, each turn a genuinely new question so the
# conversation keeps growing (a real multi-turn shape, not N repeats of one message).
TURNS = [
    "What's a good strategy for testing a rate limiter under burst load?",
    "How would that change if the limiter needs to be distributed across multiple servers?",
    "What are the tradeoffs between a token bucket and a sliding window log for this?",
    "Sketch the interface for a Go implementation of whichever approach you'd pick.",
    "What edge cases would you specifically write tests for?",
    "How would you benchmark it to make sure the implementation doesn't add latency under load?",
]


def run_transcript(url, max_tokens, temperature, nonce):
    messages = []
    out = []
    for i, turn in enumerate(TURNS):
        text = f"[{nonce}] {turn}" if i == 0 else turn
        messages.append({"role": "user", "content": text})
        body = {"model": "bench", "messages": messages, "max_tokens": max_tokens,
                "temperature": temperature, "stream": False}
        t, resp = post(url, body)
        choices = resp.get("choices") or [{}]
        content = (choices[0].get("message") or {}).get("content") or ""
        usage = resp.get("usage") or {}
        out.append({
            "n": i + 1,
            "latency_ms": round(t * 1000, 1),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "prefill_reused_tokens": usage.get("prefill_reused_tokens"),
        })
        messages.append({"role": "assistant", "content": content})
    return out


def run_concurrent(url, max_tokens, temperature, n_clients):
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n_clients) as ex:
        futures = [ex.submit(run_transcript, url, max_tokens, temperature, f"w7p-c{i}-{t0}")
                   for i in range(n_clients)]
        per_client = [f.result() for f in futures]
    wall_s = time.perf_counter() - t0
    total_completion = sum((t["completion_tokens"] or 0) for c in per_client for t in c)
    return {
        "n_clients": n_clients,
        "wall_s": round(wall_s, 3),
        "total_completion_tokens": total_completion,
        "aggregate_tok_s": round(total_completion / wall_s, 2) if wall_s > 0 else None,
        "per_client_latency_ms": [[t["latency_ms"] for t in c] for c in per_client],
        "per_client": per_client,
    }


def machine_header():
    def sh(cmd):
        try:
            return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=20).stdout.strip()
        except Exception:
            return ""
    return {
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": platform.node(),
        "model": MODEL_PATH,
        "goinfer_commit": sh(f"git -C {HERE}/.. rev-parse --short HEAD"),
        "goinfer_dirty": bool(sh(f"git -C {HERE}/.. status --porcelain")),
        "goinfer_serve_mtime": sh("stat -f %%Sm %s 2>/dev/null" % SERVE_CPU_METAL),
        "llama_server_version": sh(f"{LLAMA_BIN} --version 2>&1 | head -1"),
        "uptime": sh("uptime"),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--clients", default="1,2,4")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--engines", default="goinfer,llamacpp", help="comma list: goinfer, llamacpp")
    ap.add_argument("--backend", default="metal", help="goinfer's -backend (cpu records as goinfer_cpu)")
    ap.add_argument("--serve-args", default="", help="extra goinfer serve flags, e.g. '-max-concurrent 4'")
    ap.add_argument("--key", default="", help="results key for this goinfer run (default goinfer / goinfer_<backend>) — "
                    "lets two serve builds share one results file")
    a = ap.parse_args()
    engines = [e.strip() for e in a.engines.split(",") if e.strip()]
    gkey = a.key or ("goinfer" if a.backend == "metal" else f"goinfer_{a.backend}")

    levels = [int(x) for x in a.clients.split(",")]
    hdr = machine_header()
    results = {"goinfer": {}, "llamacpp": {}}
    results.setdefault(gkey, {})
    if os.path.exists(a.out):
        try:
            prev = json.load(open(a.out))
            results = prev.get("results", results)
            results.setdefault(gkey, {})
            results.setdefault("llamacpp", {})
            print(f"[w7-plain] resuming: {sum(len(v) for v in results.values())} cell(s) already in {a.out}",
                  file=sys.stderr)
        except Exception:
            pass

    def save():
        out = {"header": hdr, "clients": levels, "max_tokens": a.max_tokens,
               "temperature": a.temperature, "results": results}
        with open(a.out, "w") as f:
            json.dump(out, f, indent=2)

    for n in levels if "goinfer" in engines else []:
        if str(n) in results[gkey]:
            print(f"[w7-plain] {gkey} clients={n}: already done, skipping", file=sys.stderr)
            continue
        print(f"[w7-plain] {gkey} clients={n} starting fresh server", file=sys.stderr)
        with GoinferServer(a.backend, a.serve_args) as srv:
            r = run_concurrent(srv.url, a.max_tokens, a.temperature, n)
        results[gkey][str(n)] = r
        save()
        reuse = [[t.get("prefill_reused_tokens") for t in c] for c in r["per_client"]]
        print(f"[w7-plain] {gkey} clients={n}: wall={r['wall_s']}s aggregate={r['aggregate_tok_s']} tok/s; "
              f"prefill_reused per client per turn {reuse}", file=sys.stderr)

    for n in levels if "llamacpp" in engines else []:
        if str(n) in results["llamacpp"]:
            print(f"[w7-plain] llamacpp clients={n}: already done, skipping", file=sys.stderr)
            continue
        print(f"[w7-plain] llamacpp clients={n} starting fresh server (np={n})", file=sys.stderr)
        with LlamaServer(n_slots=n) as srv:
            r = run_concurrent(srv.url, a.max_tokens, a.temperature, n)
        results["llamacpp"][str(n)] = r
        save()
        print(f"[w7-plain] llamacpp clients={n}: wall={r['wall_s']}s aggregate={r['aggregate_tok_s']} tok/s",
              file=sys.stderr)

    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
