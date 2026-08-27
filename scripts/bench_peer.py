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
  - PROVENANCE AND MACHINE STATE stamped into the results file itself (added 2026-08-26): a
    header record with driver, distro, kernel, goinfer commit + tree-dirty, peer version and
    serve-binary mtimes, plus load average and GPU temperature recorded at the start of every
    cell. It also REFUSES to start on a box that is not idle. Before this, the file carried
    numbers and nothing that said what produced them, so an operator had to remember the driver
    -- which is precisely how the August 2026 CUDA re-anchor became necessary
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
    # phi3-mini and gemma3-1b carry §B5's STALE cells (the sampled rows, and every cell of these
    # two), so re-anchoring §B5 after the 2026-08-25 driver/distro upgrade needs them addressable
    # here. Both were already in the peer store as tags p3m/g31b; only this table was missing them,
    # which is why the §B5 re-measure looked like it needed new assets and did not.
    #
    # gemma3-1b's GGUF is the ollama blob copied into ~/models under a real name. It was in neither
    # ~/models nor the archive: the geometry is not optional -- its sliding window is the case that
    # exposed the split-KV gate testing nKeys instead of the window-clamped nWin.
    "phi3-mini":  (os.path.expanduser("~/models/phi3-mini-4k-gguf/Phi-3-mini-4k-instruct-q4.gguf"), "p3m"),
    "gemma3-1b":  (os.path.expanduser("~/models/gemma3-1b-q4_k_m.gguf"), "g31b"),
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

# --- MACHINE STATE ---------------------------------------------------------------------------
# Added 2026-08-26. docs/benchmarks.md requires a "verified-idle box, with the machine state
# recorded beside the number", and this harness recorded NONE of it: the results file carried
# phase/engine/model/depth/config/runs and nothing about the machine, the driver, or the build.
# The provenance lived only in whatever the operator happened to name the log file.
#
# That is the same defect this repo just spent a day on from the other end. The CUDA rows were
# re-anchored in August 2026 because a driver version had not been attached to the numbers it
# governed; the instrument that produced those numbers could not attach it either. A
# provenance-gated page cannot be fed by an instrument that emits anonymous numbers.
#
# Two levels, and they answer different questions:
#   - the PROVENANCE header, once per file: what stack produced this, recorded automatically so
#     it cannot be forgotten or mistyped after the fact.
#   - per-cell MACHINE state: load average and GPU temperature at the moment that cell started.
#
# Read the per-cell numbers as a RECORD, not as an idle check. After the first cell the load is
# mostly this harness's own servers, so a high figure there is expected and means nothing. The
# idle question is only answerable BEFORE any work starts, which is what preflight() gates.

ARCHIVE_ROOTS = ("/srv/models", "/Volumes/")

def check_bench_disk(paths):
    """A timed run must read its checkpoint from the local bench set, never from the archive.

    Prose could not carry this one. The rule lived in three documents and was complete in one:
    the other two named only /Volumes/, which does not exist on Linux -- and /srv/models is a
    LOCAL mount on the very box that measures every CUDA row, so "benchmark from local disk" reads
    as permission for it. Reading a checkpoint off the 5400 rpm SMR archive does not error; it
    returns a plausible, wrong number, and the resulting row is indistinguishable afterwards from
    a good one. So the check goes where the path is used, not only where the rule is written."""
    bad = [p for p in paths
           if any(os.path.realpath(p).startswith(r) or p.startswith(r) for r in ARCHIVE_ROOTS)]
    if bad:
        sys.exit("REFUSED: these checkpoints are on the ARCHIVE, which is not a bench surface on "
                 "either machine:\n  " + "\n  ".join(bad) +
                 "\nThe archive is a 5400 rpm SMR disk (and over SMB from the MacBook). A run that "
                 "reads it measures that disk, not the engine, and does so WITHOUT ERRORING. "
                 "Copy to the local bench set first (`models-pull <name>`) and re-run. "
                 "See docs/benchmarks.md, 'Model storage'.")

def _sh(cmd, default=""):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=15).stdout.strip() or default
    except Exception:
        return default

def _loadavg():
    try:
        return list(os.getloadavg())
    except Exception:
        return None

def _gpu_state():
    out = _sh(["nvidia-smi", "--query-gpu=driver_version,name,temperature.gpu,memory.used,memory.total",
               "--format=csv,noheader,nounits"])
    if not out:
        return None
    f = [x.strip() for x in out.splitlines()[0].split(",")]
    if len(f) < 5:
        return None
    def num(x):
        try:
            return int(x)
        except Exception:
            return None
    return {"driver": f[0], "name": f[1], "temp_c": num(f[2]),
            "mem_used_mib": num(f[3]), "mem_total_mib": num(f[4])}

def _gpu_compute_apps():
    out = _sh(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"])
    return [l.strip() for l in out.splitlines() if l.strip()]

def _ollama_version():
    """`ollama --version` prints a "could not connect" warning line when no server is up, and the
    warning also carries a version. Pull the first version-shaped token rather than a line, so the
    recorded peer version does not depend on whether a server happened to be running."""
    import re
    m = re.search(r"\d+\.\d+\.\d+", _sh([OLLAMA, "--version"]))
    return m.group(0) if m else None

def provenance():
    """Everything a reader needs to know whether these numbers are comparable to another set."""
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    git = lambda *a: _sh(["git", "-C", repo] + list(a))
    return {
        "kind": "provenance",
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": socket.gethostname(),
        "uname": _sh(["uname", "-srm"]),
        "distro": _sh(["sh", "-c", ". /etc/os-release 2>/dev/null && echo \"$NAME $VERSION_ID\""]),
        "goinfer_commit": git("rev-parse", "--short", "HEAD"),
        "goinfer_tree_dirty": bool(git("status", "--porcelain")),
        "peer": {"ollama_bin": OLLAMA, "version": _ollama_version()},
        "serve_binaries": {k: {"path": v, "mtime": (time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                                  time.gmtime(os.path.getmtime(v)))
                                                   if os.path.exists(v) else None)}
                           for k, v in SERVE.items()},
        "gpu": _gpu_state(),
        "cpu_count": os.cpu_count(),
        "loadavg_at_start": _loadavg(),
        "gpu_compute_apps_at_start": _gpu_compute_apps(),
        "sampling": "sent explicitly per cell; see each record's `sent`",
    }

def machine_state():
    """Recorded at the START of each cell. See the note above: a record, not a gate."""
    return {"loadavg": _loadavg(), "gpu": _gpu_state()}

def preflight():
    """The idle check, and the ONLY place it is answerable -- before this harness loads the box.

    Refuses rather than warns. A warning printed into a 39-minute detached run is read, if at
    all, after the numbers already exist and have already been believed."""
    cap = float(os.environ.get("BENCH_MAX_LOADAVG", "1.0"))
    la = _loadavg()
    if la and la[0] > cap:
        sys.exit(f"REFUSED: 1-min load average {la[0]:.2f} exceeds {cap:.2f}. The box is not idle, "
                 f"and a number measured on a busy box is not distinguishable afterwards from a "
                 f"number measured on a quiet one. Wait, or raise BENCH_MAX_LOADAVG deliberately.")
    check_bench_disk([p for p, _ in MODELS.values()])
    apps = _gpu_compute_apps()
    if len(apps) > 1:
        sys.exit("REFUSED: %d compute processes already hold the GPU (1 = the compositor, expected):\n  %s"
                 % (len(apps), "\n  ".join(apps)))
    print(f"# preflight OK: loadavg {la} · gpu {_gpu_state()} · compute apps {apps}", flush=True)

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
    # §B5's "temp-only" rows: temperature 1.0 with NO truncation, which is goinfer's own default.
    # Sent EXPLICITLY to both sides rather than relying on own_defaults, because the two engines'
    # defaults differ — own_defaults measures "each side's default", which is a different question
    # and would not reproduce the §B5 row.
    "temp1.0_notrunc": {
        "goinfer": {"temperature": 1.0},
        "ollama": {"temperature": 1.0, "seed": 1},
        "note": "temperature=1.0, no truncation, both sides",
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

def plan_models():
    """The models the sweep runs. DEFAULTS TO THE HISTORICAL LIST, unchanged.

    BENCH_MODELS overrides it, which is how §B5's stale cells get re-measured: those are the
    sampled rows plus every phi3-mini and gemma3-1b cell, and both models are in MODELS but were
    never in the plan. Parameterised rather than edited so the release sweep keeps producing the
    same rows by default -- silently widening a release sweep is not a thing a re-measure should do.

        BENCH_MODELS=phi3-mini,gemma3-1b python3 scripts/bench_peer.py ...
    """
    raw = os.environ.get("BENCH_MODELS", "").strip()
    if not raw:
        return ["0.5B", "1.5B", "7B"]
    picked = [m.strip() for m in raw.split(",") if m.strip()]
    unknown = [m for m in picked if m not in MODELS]
    if unknown:
        sys.exit(f"BENCH_MODELS: unknown model key(s) {unknown}; known: {sorted(MODELS)}")
    return picked


def plan_configs():
    """Sampling configurations to sweep, beyond greedy. EMPTY by default, so the release sweep is
    unchanged: adding a phase that runs by default would silently make every future sweep longer."""
    raw = os.environ.get("BENCH_CONFIGS", "").strip()
    if not raw:
        return []
    picked = [c.strip() for c in raw.split(",") if c.strip()]
    unknown = [c for c in picked if c not in CONFIGS]
    if unknown:
        sys.exit(f"BENCH_CONFIGS: unknown config(s) {unknown}; known: {sorted(CONFIGS)}")
    return picked


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

    # The provenance header is element 0 and is REFRESHED on every (re)start, with the earlier
    # ones kept in `resumed_from`. A resumed run can cross a reboot, a driver change or a rebuild
    # -- exactly the events that make two halves of one file incomparable -- so the file has to
    # be able to say "these cells were not all produced by the same stack".
    preflight()
    prov = provenance()
    if out and isinstance(out[0], dict) and out[0].get("kind") == "provenance":
        prev = out[0].pop("resumed_from", [])
        # Keep whichever identifying keys the previous header actually had. A reconstructed
        # header (one written by hand for a run that predates this capture) uses different names,
        # and dropping it silently would lose exactly the "these halves differ" signal this exists
        # for -- so fall back to the whole header rather than to nothing.
        keys = [k for k in ("utc", "utc_run_start", "goinfer_commit", "gpu", "uname",
                            "instrument_read", "RECONSTRUCTED") if k in out[0]]
        snap = {k: out[0][k] for k in keys} if keys else dict(out[0])
        prov["resumed_from"] = prev + [snap]
        out[0] = prov
    else:
        out.insert(0, prov)
    with open(outpath, "w") as f:
        json.dump(out, f, indent=1)

    plan = []
    # A) the backend table, greedy, one depth. goinfer on all three backends; ollama on the
    #    two it actually has. webgpu has NO ollama counterpart -- it is scored against the
    #    ollama CUDA row and labelled cross-backend in the writeup, never as a peer cell.
    for mk in plan_models():
        for eng, be in [("goinfer","cpu"), ("ollama","cpu"),
                        ("goinfer","cuda"), ("ollama","cuda"),
                        ("goinfer","webgpu")]:
            plan.append(("A", eng, be, mk, 128, "greedy"))
    # B) depth curve, CUDA only, both engines
    for mk in plan_models():
        for d in [512, 2048, 3900]:
            for eng in ["goinfer", "ollama"]:
                plan.append(("B", eng, "cuda", mk, d, "greedy"))

    # Phase C is the SAMPLING axis, and it is empty unless asked for. §B5's stale set is "the
    # sampled (temp / temp+top_p) rows"; CONFIGS has carried those definitions all along but no
    # phase ever scheduled them, so those rows had no reproducible path — which is why they could
    # not simply be re-run after the 2026-08-25 re-anchor.
    #
    #     BENCH_CONFIGS=temp0.8_topp0.95,temp0.8_topk40 python3 scripts/bench_peer.py ...
    for cfg in plan_configs():
        for mk in plan_models():
            for eng in ["goinfer", "ollama"]:
                plan.append(("C", eng, "cuda", mk, 128, cfg))

    print(f"# {len(plan)} cells planned, {len(done)} already done", flush=True)
    for phase, engine, backend, mk, depth, cfg in plan:
        key = (phase, engine, backend, mk, depth, cfg)
        if key in done:
            print(f"# skip (done): {key}", flush=True)
            continue
        t0 = time.time()
        machine = machine_state()
        rates, err = run_cell(engine, mk, depth, cfg, backend)
        rec = {"phase": phase, "engine": engine, "backend": backend, "model": mk,
               "depth": depth, "prompt_tokens": prompt_tokens(depth, mk),
               "config": cfg, "sent": CONFIGS[cfg].get(engine, {}),
               "note": CONFIGS[cfg]["note"], "runs": rates, "error": err,
               "machine": machine,
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
