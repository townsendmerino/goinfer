#!/usr/bin/env python3
"""bench_peer.py — the COMMITTED goinfer-vs-peer harness. Both sides, one method.

It exists because the previous arrangement could not produce a defensible comparison:
scripts/bench_compare.sh measures goinfer with in-process Go benchmarks and never drives the peer,
so the two columns of docs/benchmarks.md were produced by different methods — a kernel throughput
divided by an end-to-end throughput. That ratio was retired on 2026-08-09.

WHAT IT GUARANTEES
  - both engines driven over their OWN HTTP server (goinfer /v1/chat/completions; Ollama /api/chat)
  - DECODE-ONLY: inter-token rate timed client-side from the FIRST streamed token onward, so
    prefill is excluded on both sides by construction, and neither engine's self-reported
    accounting is trusted
  - INTERLEAVED cell by cell, with a server RESTART between cells
  - sampling sent EXPLICITLY to both engines and echoed into the results file (never assumed)
  - the same WEIGHTS on both sides, verified per-tensor by scripts/gguf_same_weights.py.
    NOT by file md5: `ollama create` repacks the container in a different tensor ORDER, so the
    file hash always differs even when every tensor is bit-identical (measured 2026-08-22 on all
    three models here: 339/339, 339/339, 291/291 tensors identical, whole-file md5 different)

  decode tok/s = (n_tokens - 1) / (t_last_token - t_first_token)

PROMPT DEPTHS ARE TOKEN-CALIBRATED (prompts.json, built by scripts/bench_prompts_calibrate.py).
Word-count filler is not a depth axis: "item1234" tokenizes to ~4 tokens, which once turned an
intended 2048-token prompt into 9094 and silently blew the context window.

  GOINFER_SERVE=/path/to/cuda-serve OLLAMA_BIN=~/ollama-0325/bin/ollama \
    python3 scripts/bench_peer.py results.json
"""
import json, os, signal, socket, subprocess, sys, time, urllib.request, statistics

GOINFER = os.environ.get("GOINFER_SERVE", "./goinfer-serve")
OLLAMA = os.environ.get("OLLAMA_BIN", os.path.expanduser("~/ollama-0325/bin/ollama"))
OLLAMA_MODELS = os.environ.get("OLLAMA_MODELS", os.path.expanduser("~/ollama-0325/models"))
MODELS = {
    "0.5B": (os.path.expanduser("~/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf"), "q05"),
    "1.5B": (os.path.expanduser("~/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"), "q15"),
    # 7B added 2026-08-22. Both other rows are TINY, and this repo has already been burned by
    # that exactly once: CUDA graphs measured 1.4-1.7x on a small model and 1.01x at real size,
    # because CPU dispatch overlaps GPU compute once the model is big enough. A peer number
    # published off 0.5B/1.5B alone does not transfer.
    "7B":   (os.path.expanduser("~/models/qwen2.5-7b-instruct-q4_k_m.gguf"), "q7b"),
}

# One goinfer binary per backend. Ollama has no WebGPU build, so the webgpu row is compared
# against ollama's CUDA row and MUST be labelled cross-backend rather than presented as a
# like-for-like peer cell.
SERVE = {
    "cpu":    os.environ.get("GOINFER_SERVE_CPU",    "/home/francis/bench-v0.15.0/serve-cpu"),
    "cuda":   os.environ.get("GOINFER_SERVE_CUDA",   "/home/francis/bench-v0.15.0/serve-cuda"),
    "webgpu": os.environ.get("GOINFER_SERVE_WEBGPU", "/home/francis/bench-v0.15.0/serve-webgpu"),
}
GPORT, OPORT = 8099, 11499
NGEN = 64          # tokens generated per completion
NCOMP = 8          # completions per run  (>= 8 required)
NRUNS = 2          # runs per cell        (>= 2 required, spread reported)

def wait_port(port, timeout=180):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return True
        except OSError:
            time.sleep(0.5)
    return False

def post_stream(url, payload, parse):
    """POST a streaming request; return (n_tokens, t_first, t_last)."""
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    n, t_first, t_last = 0, None, None
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line:
                continue
            tok = parse(line)
            if tok is None:
                continue
            now = time.perf_counter()
            if t_first is None:
                t_first = now
            else:
                n += 1          # count INTER-token intervals only
                t_last = now
    return n, t_first, t_last

def parse_openai(line):
    if not line.startswith("data:"):
        return None
    body = line[5:].strip()
    if body == "[DONE]":
        return None
    try:
        d = json.loads(body)
        c = d.get("choices", [{}])[0].get("delta", {}).get("content")
        return c if c else None
    except Exception:
        return None

def parse_ollama(line):
    try:
        d = json.loads(line)
        if d.get("done"):
            return None
        c = d.get("message", {}).get("content")
        return c if c else None
    except Exception:
        return None

_PROMPTS = json.load(open(os.environ.get("BENCH_PROMPTS",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts.json"))))

def prompt_for_depth(depth, model_key):
    """Token-CALIBRATED prompt (calib.py): measured via goinfer's usage.prompt_tokens so the
    depth axis is real token depth, not a word count. The first attempt used "itemN" filler,
    which tokenizes ~4 tokens/word — "depth 2048" was actually 9094 tokens and blew the 4096
    context window, 400ing both engines identically."""
    return _PROMPTS[f"{model_key}:{depth}"]["text"]

def prompt_tokens(depth, model_key):
    return _PROMPTS[f"{model_key}:{depth}"]["tokens"]

def goinfer_payload(model_path, prompt, cfg):
    p = {"model": "bench", "stream": True, "max_tokens": NGEN,
         "messages": [{"role": "user", "content": prompt}]}
    p.update(cfg.get("goinfer", {}))
    return p

def ollama_payload(tag, prompt, cfg, backend="cuda"):
    p = {"model": tag, "stream": True,
         "messages": [{"role": "user", "content": prompt}],
         "options": {"num_predict": NGEN, "num_ctx": 4096}}
    if backend == "cpu":
        p["options"]["num_gpu"] = 0     # force CPU; ollama defaults to CUDA when present
    p["options"].update(cfg.get("ollama", {}))
    return p

# Sampling configurations. Each records EXACTLY what is sent to each side.
CONFIGS = {
    "greedy": {
        "goinfer": {"temperature": 0},
        "ollama": {"temperature": 0, "seed": 1},
        "note": "temperature=0 both sides (deterministic)",
    },
    "temp0.8_topp0.95": {
        "goinfer": {"temperature": 0.8, "top_p": 0.95, "seed": 1},
        "ollama": {"temperature": 0.8, "top_p": 0.95, "seed": 1},
        "note": "temperature=0.8, top_p=0.95 both sides",
    },
    "temp0.8_topk40": {
        "goinfer": {"temperature": 0.8, "top_k": 40, "seed": 1},
        "ollama": {"temperature": 0.8, "top_k": 40, "seed": 1},
        "note": "temperature=0.8, top_k=40 both sides",
    },
    "own_defaults": {
        "goinfer": {},
        "ollama": {},
        "note": "UNMATCHED — no sampling params sent; each side uses its own defaults",
    },
}

def run_cell(engine, model_key, depth, cfg_name, backend="cuda"):
    """Restart the server, do NRUNS runs of NCOMP completions, return per-run rates.

    `backend` selects the goinfer binary AND, for the ollama side, whether the model is forced
    onto the CPU (options.num_gpu=0). Without that force ollama silently uses CUDA, which would
    have made the 'CPU' row a GPU number on both engines and nobody would have seen it."""
    path, tag = MODELS[model_key]
    cfg = CONFIGS[cfg_name]
    prompt = prompt_for_depth(depth, model_key)
    proc = None
    try:
        if engine == "goinfer":
            proc = subprocess.Popen(
                [SERVE[backend], "-model", f"bench={path}", "-backend", backend,
                 "-addr", f"127.0.0.1:{GPORT}", "-quant", "int4"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                preexec_fn=os.setsid)
            port, url, parse, mk = GPORT, f"http://127.0.0.1:{GPORT}/v1/chat/completions", parse_openai, \
                (lambda: goinfer_payload(path, prompt, cfg))
        else:
            env = dict(os.environ, OLLAMA_MODELS=OLLAMA_MODELS,
                       OLLAMA_HOST=f"127.0.0.1:{OPORT}")
            proc = subprocess.Popen([OLLAMA, "serve"], env=env,
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                    preexec_fn=os.setsid)
            port, url, parse, mk = OPORT, f"http://127.0.0.1:{OPORT}/api/chat", parse_ollama, \
                (lambda: ollama_payload(tag, prompt, cfg, backend))
        if not wait_port(port):
            return None, "server did not come up"
        # warm: one discarded completion (model load + first-run outlier)
        try:
            post_stream(url, mk(), parse)
        except Exception as e:
            return None, f"warmup failed: {e}"

        run_rates = []
        for _ in range(NRUNS):
            rates = []
            for _ in range(NCOMP):
                n, tf, tl = post_stream(url, mk(), parse)
                if n >= 2 and tf and tl and tl > tf:
                    rates.append(n / (tl - tf))
            if rates:
                run_rates.append(statistics.mean(rates))
        return run_rates, None
    except Exception as e:
        return None, str(e)
    finally:
        if proc:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                proc.wait(timeout=30)
            except Exception:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except Exception:
                    pass
            time.sleep(3)  # let VRAM settle before the next engine loads

def main():
    """Plan. Phase A is the headline BACKEND table -- every backend at one depth, so the
    cross-backend row is apples-to-apples. Phase B is the depth curve, CUDA only, because
    running four depths x five backend-cells x three models does not fit in a release window
    and the depth axis is about the engine's prefill/attention scaling, not the backend.

    RESUMABLE: results are keyed and reloaded from the output file, so a killed or
    disconnected run is restarted with the same command and skips completed cells."""
    outpath = sys.argv[1]
    out = []
    if os.path.exists(outpath):
        try:
            out = json.load(open(outpath))
            print(f"# resuming: {len(out)} cells already recorded in {outpath}", flush=True)
        except Exception:
            out = []
    done = {(r["phase"], r["engine"], r.get("backend"), r["model"], r["depth"], r["config"])
            for r in out if r.get("runs")}

    plan = []
    # A) the backend table, greedy, one depth. goinfer on all three backends; ollama on the
    #    two it actually has. webgpu has NO ollama counterpart -- it is scored against the
    #    ollama CUDA row and labelled cross-backend in the writeup, never as a peer cell.
    for mk in ["0.5B", "1.5B", "7B"]:
        for eng, be in [("goinfer","cpu"), ("ollama","cpu"),
                        ("goinfer","cuda"), ("ollama","cuda"),
                        ("goinfer","webgpu")]:
            plan.append(("A", eng, be, mk, 128, "greedy"))
    # B) depth curve, CUDA only, both engines
    for mk in ["0.5B", "1.5B", "7B"]:
        for d in [512, 2048, 3900]:
            for eng in ["goinfer", "ollama"]:
                plan.append(("B", eng, "cuda", mk, d, "greedy"))

    print(f"# {len(plan)} cells planned, {len(done)} already done", flush=True)
    for phase, engine, backend, mk, depth, cfg in plan:
        key = (phase, engine, backend, mk, depth, cfg)
        if key in done:
            print(f"# skip (done): {key}", flush=True)
            continue
        t0 = time.time()
        rates, err = run_cell(engine, mk, depth, cfg, backend)
        rec = {"phase": phase, "engine": engine, "backend": backend, "model": mk,
               "depth": depth, "prompt_tokens": prompt_tokens(depth, mk),
               "config": cfg, "sent": CONFIGS[cfg].get(engine, {}),
               "note": CONFIGS[cfg]["note"], "runs": rates, "error": err,
               "secs": round(time.time() - t0, 1)}
        if rates:
            rec["mean"] = round(statistics.mean(rates), 1)
            rec["spread"] = round(max(rates) - min(rates), 1)
        out.append(rec)
        print(json.dumps(rec), flush=True)
        with open(outpath, "w") as f:
            json.dump(out, f, indent=1)

if __name__ == "__main__":
    main()
