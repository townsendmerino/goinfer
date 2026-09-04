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
# llama-server (docs/task-peer-benchmarks.md §1): verified live 2026-09-04 that its
# /v1/chat/completions stream is byte-for-byte the shape parse_openai already handles --
# same role/content/finish_reason delta framing, same usage.completion_tokens final chunk --
# so it reuses parse_openai directly rather than a third parser.
LLAMACPP = os.environ.get("LLAMACPP_BIN", "llama-server")
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
    # M35/M26 added 2026-09-04 (docs/task-peer-benchmarks.md §2, tier 1). The GGUF path here is
    # what ollama and llama-server load -- Q4_K_M, same quant family as every other row. Neither
    # checkpoint had a Q4_K_M GGUF on this box: the archive only carried M35 as Q8_0 and M26 as a
    # legacy Q4_0, so both were REQUANTIZED locally with llama-quantize --allow-requantize (source
    # was already quantized, not f32/f16 -- llama-quantize refuses a q8_0 source without that flag
    # for exactly this reason). M35's source was q8_0 (near-lossless), so Q4_K_M off it is close to
    # quantizing from full precision; M26's source was already q4_0 (lossy), so its Q4_K_M here is a
    # DOUBLE quantization and is NOT the same provenance as a real from-f16 Q4_K_M -- flag this
    # wherever M26's row gets quoted for quality, not just speed. See GOINFER_MOE_PATH below: goinfer
    # itself does NOT load this GGUF -- it runs its own kind-4 .giw bundle (the shipped-default path
    # for these two MoE models), so the GGUF here is the PEER-ONLY artifact.
    "M35": (os.path.expanduser("~/models/qwen3.6-35b-a3b-q4_k_m.gguf"), "m35q4km"),
    "M26": (os.path.expanduser("~/models/gemma4-26b-q4_k_m.gguf"), "m26q4km"),
    # G20 added 2026-09-05 (docs/task-peer-benchmarks.md §2/§5, tier 2: "fits the Mac, not the
    # card: a resident cell on one box and an offload cell on the other"). gpt-oss-20b's OWN
    # shipped MXFP4 quant (OpenAI's native format) -- the only GGUF for it on this box, and the
    # SAME file all three engines load (unlike M35/M26 below): decoder/gguf.go's gptOssArchitecture
    # routes expert weights through stackedExperts/RowDequantizer regardless of the harness's
    # `-quant` flag, so there is no separate pre-quantized .giw bundle for goinfer here and no
    # quant-flag refusal either -- confirmed live 2026-09-05 with a smoke test (`-quant int4
    # -moe-cache-experts` loaded it CUDA-resident at 7269 MiB VRAM in ~90s). 20B total / ~3.6B
    # active MoE -- past the 8 GB card, hence MOE_MODELS membership below, but it is NOT in
    # GOINFER_MOE_PATH (no path substitution, no quant-flag omission -- see the is_moe split in
    # run_cell's goinfer branch).
    "G20": (os.path.expanduser("~/models/gpt-oss-20b-MXFP4.gguf"), "g20"),
}

# MoE cells whose VRAM footprint exceeds this box's 8 GB card: goinfer needs -moe-cache-experts
# (host<->VRAM expert streaming) or it silently declines resident CUDA and falls back to the CPU
# path -- a "cuda" row that is actually CPU, with nothing in the response to say so (measured
# 2026-09-04: without the flag the server logs the decline and keeps serving on CPU). Also drives
# the longer load timeout and (on llama.cpp) dropping -ngl so its own --fit can place layers.
# NOTE this is a SUPERSET of GOINFER_MOE_PATH's keys: M35/M26 additionally load through a
# pre-quantized .giw bundle that BAKES ITS OWN QUANT (so goinfer's `-quant` flag must be omitted
# for those two, not just changed -- "cannot apply to the prequantized .giw bundle ... baked at
# int4mix"); G20 has no such bundle and takes the harness's normal `-quant int4` unmodified (see
# the MODELS/G20 comment above) while still needing everything else in this set.
MOE_MODELS = {"M35", "M26", "G20"}

# goinfer's OWN path for the MOE_MODELS set, distinct from MODELS[key][0] above (which is the
# Q4_K_M GGUF ollama/llama-server load). goinfer runs its native kind-4 .giw bundle instead --
# the shipped-default configuration for these two models (docs/task-peer-benchmarks.md §2) -- so
# the two engines are NOT reading the same file for these cells, only the same nominal quant tier.
GOINFER_MOE_PATH = {
    "M35": os.path.expanduser("~/models/qwen3.6-35b-a3b-int4.giw"),
    "M26": os.path.expanduser("~/models/gemma4-26b-int4.giw"),
}
# NOTE (merge 2026-09-05): the Mac box does NOT have qwen3.6-35b-a3b-q4_k_m.gguf,
# gemma4-26b-q4_k_m.gguf, or the two GOINFER_MOE_PATH .giw files above -- it only has
# Qwen3.5-35B-A3B-Q4_K_M.gguf and a Q4_0 gemma4-26b GGUF (see the Mac-side results already
# recorded in docs/measurements/peer-matrix-2026-09/mac-m1pro-m35m26-tier1-2026-09-04.json,
# which carry their own provenance notes for those paths). M35/M26 are parked on the Mac
# (kernel panic + swap-thrashing incident, 2026-09-04/05 -- see docs/task-peer-benchmarks.md
# session notes) so this canonical (nobara-quality, real Q4_K_M) path set is not expected to
# resolve there; a Mac re-pull/requantize is a separate follow-up if M35/M26 work resumes on
# that box.

# One goinfer binary per backend. Ollama has no WebGPU build, so the webgpu row is compared
# against ollama's CUDA row and MUST be labelled cross-backend rather than presented as a
# like-for-like peer cell.
SERVE = {
    "cpu":    os.environ.get("GOINFER_SERVE_CPU",    "/home/francis/bench-v0.15.0/serve-cpu"),
    "cuda":   os.environ.get("GOINFER_SERVE_CUDA",   "/home/francis/bench-v0.15.0/serve-cuda"),
    "webgpu": os.environ.get("GOINFER_SERVE_WEBGPU", "/home/francis/bench-v0.15.0/serve-webgpu"),
    # Metal, added 2026-09-04 for the Mac side of docs/task-peer-benchmarks.md. Built from the
    # `metal/` submodule (M-19: the root cmd/serve builds no backend) -- `go build
    # github.com/townsendmerino/goinfer/metal/cmd/serve`, darwin-only, no build tag needed. No
    # Linux-box default exists (there is nothing to default to), so this path is DELIBERATELY
    # wrong/absent unless GOINFER_SERVE_METAL is set -- it will only ever be selected by
    # BENCH_BACKENDS=metal on darwin.
    "metal":  os.environ.get("GOINFER_SERVE_METAL",  "/nonexistent/serve-metal"),
}
GPORT, OPORT, LPORT = 8099, 11499, 8098
# DEEP CONTEXT (§B7). BENCH_DEEP_CTX sets the resident/served context cap in positions; 0 keeps
# the shallow protocol untouched. A 32k prefill costs orders of magnitude more than the decode being
# measured, so deep cells deliberately use FEWER requests with MORE decode tokens each — the same
# protocol difference §B7 recorded, not a shortcut.
DEEP_CTX = int(os.environ.get("BENCH_DEEP_CTX", "0"))

NGEN = int(os.environ.get("BENCH_NGEN", "64"))  # tokens generated per completion
# BENCH_NGEN exists for spec/10's window-variance question: one completion IS a decode window at
# ~fixed KV depth, so the window size under test is the completion length. Default unchanged.
NCOMP = 8          # completions per run  (>= 8 required)
NRUNS = 2          # runs per cell        (>= 2 required, spread reported)


def gen_params():
    """(tokens per completion, completions per run, runs) — deep cells use 128 x 2 x 2.

    BENCH_RUNS overrides the run count, which is how a cell gets compared against an anchor that
    used a different one. §B5's phi3-mini rows were taken at 5 runs x 8 completions while this
    harness defaults to 2 x 8, and that cell is recorded moving 112.4 -> 116.6 on re-measurement
    alone — so a 2-run figure and a 5-run figure are not the same measurement (G26)."""
    ngen, ncomp, nruns = (128, 2, 2) if DEEP_CTX else (NGEN, NCOMP, NRUNS)
    if v := os.environ.get("BENCH_RUNS", "").strip():
        nruns = int(v)
    return (ngen, ncomp, nruns)

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

def _llamacpp_version():
    """`llama-server --version` prints e.g. 'version: 0.3.0 (build 10621, commit c1d0e7a00)' --
    to STDERR, not stdout. Verified live 2026-09-04: reusing `_sh()` (stdout-only) here silently
    returned null. ollama's --version is on stdout, so _ollama_version's `_sh()` reuse is fine;
    llama-server's stream choice is just different, hence the separate capture here.

    Captures the WHOLE 'version: ...' line, not just the X.Y.Z prefix (V-08,
    docs/review-2026-09-04.md): llama.cpp cuts frequent builds under the same release tag, so two
    different daily builds can share "0.3.0" while differing in exactly the build number and commit
    this line also carries -- dropping them made two different binaries record identical
    provenance. A build whose version string doesn't even start with X.Y.Z (some llama.cpp builds
    print 'version: NNNN (sha)') used to record null from the old digit.digit.digit-only regex;
    this falls back to the raw line so provenance is never silently empty for a real binary."""
    import re
    try:
        out = subprocess.run([LLAMACPP, "--version"], capture_output=True, text=True,
                             timeout=15).stderr
    except Exception:
        return None
    m = re.search(r"version:\s*(.+)", out)
    return m.group(1).strip() if m else (out.strip() or None)

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
        "peer_llamacpp": {"bin": LLAMACPP, "version": _llamacpp_version()},
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
    check_bench_disk([p for p, _ in MODELS.values()] + list(GOINFER_MOE_PATH.values()))
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

def wait_llamacpp_ready(port, timeout=180):
    """llama-server binds its port and starts answering /health with a 503 "Loading model" WHILE
    the checkpoint is still loading -- verified live 2026-09-04 on the 1.5B (503 for ~2s). The
    generic wait_port() above only checks that the socket accepts a connection, which is already
    true at that point; sending run_cell's warmup completion against a still-loading llama-server
    would 503, land in the try/except as "warmup failed", and misreport a working peer as broken.
    On the larger Tier-1 cells (M35, H27) the load window is much longer than 1.5B's ~2s, so this
    polls /health for status=="ok" instead of merely a connectable socket."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                if json.loads(r.read()).get("status") == "ok":
                    return True
        except Exception:
            pass
        time.sleep(0.5)
    return False

def post_stream(url, payload, parse):
    """POST a streaming request; return (intervals, t_first, t_last, chunks, reported).

    COUNTS COME FROM THE ENGINE, NOT FROM THE STREAM. This used to increment once per SSE event
    and call the result n_tokens, which is a CHUNK count — and chunks are not tokens on goinfer's
    side: `streamTokens` (internal/serveapp/openai.go) emits only when `end > printed`, so a token
    held back for an incomplete UTF-8 rune or a trailing partial stop-string match produces no
    chunk at all, and the token that resolves the holdback produces one chunk carrying several
    tokens' bytes. chunks <= tokens, always in that direction, and only on our side of a paired
    comparison — so the error under-reported goinfer's own decode rate.

    `reported` is the engine's own completion-token count (OpenAI usage.completion_tokens via
    stream_options.include_usage; Ollama eval_count on the done message). `chunks` is kept so the
    tokens/chunks ratio can be recorded per cell as evidence rather than assumption.

    Timing is unchanged and still client-side: `intervals` counts INTER-event gaps for the t_first
    .. t_last window. Empty deltas are no longer discarded for timing — an empty delta is still a
    stream event and still marks an interval, and the holdback path is exactly what produces one.
    """
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    intervals, chunks, t_first, t_last, reported = 0, 0, None, None, None
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line:
                continue
            ev = parse(line)
            if ev is None:
                continue            # not a stream event at all (framing, [DONE])
            if ev.get("count") is not None:
                reported = ev["count"]      # terminal report; not an inter-token interval
                continue
            if ev.get("text"):
                chunks += 1
            now = time.perf_counter()
            if t_first is None:
                t_first = now
            else:
                intervals += 1      # count INTER-token intervals only
                t_last = now
    return intervals, t_first, t_last, chunks, reported

def parse_openai(line):
    """-> {"text": str} for a stream event, {"count": int} for the usage chunk, or None."""
    if not line.startswith("data:"):
        return None
    body = line[5:].strip()
    if body == "[DONE]":
        return None
    try:
        d = json.loads(body)
    except Exception:
        return None
    u = d.get("usage")
    if u and u.get("completion_tokens") is not None:
        return {"count": u["completion_tokens"]}    # stream_options.include_usage final chunk
    ch = d.get("choices") or []
    if not ch:
        return None
    c0 = ch[0]
    # FRAMING vs TOKEN EVENTS. The opening role chunk and the closing finish_reason chunk are
    # protocol framing, not token boundaries: counting them would add two spurious intervals and,
    # worse, put t_last on the finish chunk and stretch the timing window past the last token.
    if c0.get("finish_reason") is not None:
        return None
    dl = c0.get("delta", {})
    if dl.get("role") is not None and not dl.get("content"):
        return None
    # An EMPTY content delta that is NOT framing is a real token boundary — it is what the UTF-8 /
    # stop-string holdback produces — so it is kept rather than dropped.
    return {"text": dl.get("content") or ""}

def parse_ollama(line):
    """-> {"text": str} for a stream event, {"count", "eval_ns"} for the done message, or None."""
    try:
        d = json.loads(line)
    except Exception:
        return None
    if d.get("done"):
        # The done message carries eval_count / eval_duration and used to be discarded outright.
        # eval_count is the peer's own token count; eval_duration is kept as a cross-check against
        # our client-side timing.
        if d.get("eval_count") is not None:
            return {"count": d["eval_count"], "eval_ns": d.get("eval_duration")}
        return None
    return {"text": d.get("message", {}).get("content") or ""}

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
    ngen, _, _ = gen_params()
    p = {"model": "bench", "stream": True, "max_tokens": ngen,
         "stream_options": {"include_usage": True},   # authoritative completion_tokens
         "messages": [{"role": "user", "content": prompt}]}
    p.update(cfg.get("goinfer", {}))
    return p

def ollama_payload(tag, prompt, cfg, backend="cuda"):
    p = {"model": tag, "stream": True,
         "messages": [{"role": "user", "content": prompt}],
         "options": {"num_predict": gen_params()[0], "num_ctx": DEEP_CTX or 4096}}
    if backend == "cpu":
        p["options"]["num_gpu"] = 0     # force CPU; ollama defaults to CUDA when present
    p["options"].update(cfg.get("ollama", {}))
    return p

def llamacpp_payload(prompt, cfg):
    """llama-server's /v1/chat/completions takes the identical field names goinfer's does
    (temperature/top_p/top_k/seed, stream_options.include_usage) -- verified live 2026-09-04 --
    so this mirrors goinfer_payload rather than inventing a third shape. Sampling values are
    shared with goinfer's CONFIGS entry (see the setdefault loop below CONFIGS) rather than
    duplicated per config."""
    ngen, _, _ = gen_params()
    p = {"model": "bench", "stream": True, "max_tokens": ngen,
         "stream_options": {"include_usage": True},
         "messages": [{"role": "user", "content": prompt}]}
    p.update(cfg.get("llamacpp", {}))
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
    # G26 follow-up: the temp-only ladder. optimistic forward (6a4e0ae) runs on sampled decode only
    # and its hit rate falls with temperature (98% at T=0.2, 55.6% at T=1.0 by its own gates), so
    # its on/off value is a FUNCTION of temperature and two endpoints cannot locate the crossover.
    # Same shape as temp1.0_notrunc -- temperature only, no truncation, sent explicitly to both.
    "temp0.2_notrunc": {
        "goinfer": {"temperature": 0.2},
        "ollama": {"temperature": 0.2, "seed": 1},
        "note": "temperature=0.2, no truncation, both sides",
    },
    "temp0.4_notrunc": {
        "goinfer": {"temperature": 0.4},
        "ollama": {"temperature": 0.4, "seed": 1},
        "note": "temperature=0.4, no truncation, both sides",
    },
    "temp0.6_notrunc": {
        "goinfer": {"temperature": 0.6},
        "ollama": {"temperature": 0.6, "seed": 1},
        "note": "temperature=0.6, no truncation, both sides",
    },
    "temp0.8_notrunc": {
        "goinfer": {"temperature": 0.8},
        "ollama": {"temperature": 0.8, "seed": 1},
        "note": "temperature=0.8, no truncation, both sides",
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
# llama-server speaks goinfer's own OpenAI sampling dialect (same field names), so every config
# above gets a free "llamacpp" alias of its "goinfer" entry instead of nine duplicated blocks.
for _cfg in CONFIGS.values():
    _cfg.setdefault("llamacpp", _cfg.get("goinfer", {}))

def gate_cell_idle():
    """Re-check that the box is idle, BEFORE EVERY CELL. Returns nothing; exits on failure.

    preflight() checks once at sweep start, and that is not enough: a sweep runs for tens of
    minutes, and load arriving at minute ten is invisible to a gate that fired at minute zero.
    MEASURED 2026-08-27 -- a window-variance sweep passed preflight at loadavg 0.75, then ran five
    cells at 1.00-2.43 because another job started on the box. Every cell was contaminated and the
    sweep had to be discarded. The per-cell machine state recorded in each row is what caught it;
    this gate is so it does not have to be caught after the fact.

    ON TIMEOUT THIS REFUSES RATHER THAN PROCEEDING, which is the whole design. A settle loop that
    gives up and measures anyway is WORSE than no loop at all: it produces numbers that look
    entirely normal and are silently taken on a loaded box. Refusing loses a sweep; proceeding
    loses the ability to tell which rows were real.
    """
    cap = float(os.environ.get("BENCH_MAX_LOADAVG", "1.0"))
    waited, limit = 0, int(os.environ.get("BENCH_IDLE_WAIT", "600"))
    while True:
        la = _loadavg()
        if not la or la[0] <= cap:
            return
        if waited >= limit:
            sys.exit(f"REFUSED mid-sweep: 1-min load average {la[0]:.2f} still exceeds {cap:.2f} "
                     f"after waiting {waited}s. Another job is on the box. NOT measuring anyway — "
                     f"a cell measured under contention is indistinguishable afterwards from a "
                     f"clean one, which is how a whole sweep gets silently voided. Re-run when the "
                     f"box is free, or raise BENCH_MAX_LOADAVG deliberately.")
        print(f"# cell gate: loadavg {la[0]:.2f} > {cap:.2f}, waiting ({waited}/{limit}s)", flush=True)
        time.sleep(20)
        waited += 20


def run_cell(engine, model_key, depth, cfg_name, backend="cuda"):
    """Restart the server, do NRUNS runs of NCOMP completions, return per-run rates.

    `backend` selects the goinfer binary AND, for the ollama side, whether the model is forced
    onto the CPU (options.num_gpu=0). Without that force ollama silently uses CUDA, which would
    have made the 'CPU' row a GPU number on both engines and nobody would have seen it."""
    path, tag = MODELS[model_key]
    cfg = CONFIGS[cfg_name]
    prompt = prompt_for_depth(depth, model_key)
    # M35/M26 load a 16-26 GB checkpoint (mmap page-in + CUDA expert-cache fill for goinfer;
    # weight load for llama-server); the generic 180s default (sized for 0.5B-7B) measured a real
    # M35 goinfer load STILL not listening at 359s on this box (2026-09-04 smoke test) -- so a
    # cell using it would misreport a working peer as "server did not come up". MOE_MODELS gets a
    # longer wait; 900s matches this repo's other MoE-scale timeout (bench_prompts_calibrate.py's
    # CALIB_TIMEOUT) rather than a number nobody has measured against.
    load_timeout = 900 if model_key in MOE_MODELS else 180
    proc = None
    try:
        if engine == "goinfer":
            # MOE_MODELS: `-moe-cache-experts` is added for every model in this set so a >8GB-VRAM
            # MoE actually runs resident-with-streaming on CUDA instead of silently declining to
            # the CPU path. GOINFER_MOE_PATH is the NARROWER subset (M35/M26) that also load
            # goinfer's OWN kind-4 .giw bundle instead of the peer GGUF in `path` -- that bundle
            # bakes its own quant, so `-quant int4` is dropped for those two (it refuses to start
            # otherwise). G20 is in MOE_MODELS but not GOINFER_MOE_PATH: it loads the SAME GGUF as
            # the peers and takes the normal `-quant int4` unmodified -- see the MODELS/G20 comment.
            has_own_bundle = model_key in GOINFER_MOE_PATH
            gpath = GOINFER_MOE_PATH[model_key] if has_own_bundle else path
            quant_args = [] if has_own_bundle else ["-quant", "int4"]
            moe_args = ["-moe-cache-experts"] if (model_key in MOE_MODELS and backend == "cuda") else []
            proc = subprocess.Popen(
                [SERVE[backend], "-model", f"bench={gpath}", "-backend", backend,
                 "-addr", f"127.0.0.1:{GPORT}"] + quant_args + moe_args
                + (["-ctx", str(DEEP_CTX)] if DEEP_CTX else []),
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                preexec_fn=os.setsid)
            port, url, parse, mk = GPORT, f"http://127.0.0.1:{GPORT}/v1/chat/completions", parse_openai, \
                (lambda: goinfer_payload(path, prompt, cfg))
        elif engine == "ollama":
            env = dict(os.environ, OLLAMA_MODELS=OLLAMA_MODELS,
                       OLLAMA_HOST=f"127.0.0.1:{OPORT}")
            if DEEP_CTX:
                # §B7's recorded peer configuration. Flash attention OFF because that is what the
                # anchor used; leaving it default would compare against a different engine.
                env["OLLAMA_FLASH_ATTENTION"] = "false"
            proc = subprocess.Popen([OLLAMA, "serve"], env=env,
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                    preexec_fn=os.setsid)
            port, url, parse, mk = OPORT, f"http://127.0.0.1:{OPORT}/api/chat", parse_ollama, \
                (lambda: ollama_payload(tag, prompt, cfg, backend))
        elif engine == "llamacpp":
            # V-08 (docs/review-2026-09-04.md): this branch used to launch with NO -ngl at all,
            # so llama-server fell back to its own auto-offload default -- Metal on a Mac, meaning
            # a "cpu" cell silently ran on GPU while goinfer's cpu cell and ollama's num_gpu=0 cell
            # did not, and the backend column lied. -ngl now mirrors goinfer's own -backend switch:
            # 0 (CPU-only) for "cpu", full offload for anything else. --ctx-size is now sent
            # UNCONDITIONALLY (matching ollama_payload's own `num_ctx: DEEP_CTX or 4096`, always
            # set) rather than only under DEEP_CTX -- llama-server's own default is "load from the
            # model", which for some checkpoints is nowhere near 4096 and would silently change the
            # KV footprint being compared. -fa off under DEEP_CTX mirrors ollama's own
            # OLLAMA_FLASH_ATTENTION=false for the same §B7 peer-configuration reason.
            # MOE_MODELS on cuda: an explicit -ngl 99 DEFEATS llama-server's own --fit (on by
            # default, verified live 2026-09-04 via --help on this build, commit 427291b) --
            # --fit only adjusts arguments that are UNSET, and -ngl 99 forces full offload of a
            # 20GB/16.8GB model into an 8GB card regardless. Measured: with -ngl 99 forced,
            # M35 and M26 both ran the full 900s load-wait and never came up. Dropping -ngl for
            # this set (cuda backend only -- the "cpu" forcing below is untouched, V-08) lets
            # --fit place layers automatically; this is the zero-flag mode docs/task-peer-
            # benchmarks.md §1 calls "--fit on" and docs/task-fit-to-hardware.md's own subject.
            if backend == "cpu":
                ngl_args = ["-ngl", "0"]
            elif model_key in MOE_MODELS:
                ngl_args = []
            else:
                ngl_args = ["-ngl", "99"]
            proc = subprocess.Popen(
                [LLAMACPP, "--model", path, "--port", str(LPORT), "--host", "127.0.0.1",
                 "--ctx-size", str(DEEP_CTX or 4096)] + ngl_args
                + (["-fa", "off"] if DEEP_CTX else []),
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                preexec_fn=os.setsid)
            port, url, parse, mk = LPORT, f"http://127.0.0.1:{LPORT}/v1/chat/completions", \
                parse_openai, (lambda: llamacpp_payload(prompt, cfg))
        ready = wait_llamacpp_ready(port, timeout=load_timeout) if engine == "llamacpp" \
            else wait_port(port, timeout=load_timeout)
        if not ready:
            return None, "server did not come up", None
        # warm: one discarded completion (model load + first-run outlier)
        try:
            post_stream(url, mk(), parse)
        except Exception as e:
            return None, f"warmup failed: {e}", None

        _, ncomp, nruns = gen_params()
        run_rates, comp_rates, tok_total, chunk_total = [], [], 0, 0
        for _ in range(nruns):
            rates = []
            for _ in range(ncomp):
                intervals, tf, tl, chunks, reported = post_stream(url, mk(), parse)
                if intervals < 2 or not tf or not tl or tl <= tf:
                    continue
                # Numerator is the ENGINE's count where it gives one; the interval count is the
                # fallback and is what the old code always used. Timing is untouched.
                n = reported if reported else intervals
                # reported counts all generated tokens; the window starts at the FIRST event, so
                # one token precedes it.
                n = n - 1 if reported else n
                rates.append(n / (tl - tf))
                tok_total += (reported or 0)
                chunk_total += chunks
            if rates:
                run_rates.append(statistics.mean(rates))
                # Keep the per-completion rates too. Each is a decode WINDOW of NGEN tokens at
                # ~fixed depth, and spec/10's kill gate turns on their spread -- which this loop has
                # always computed and then discarded, reporting only the mean of 8.
                comp_rates.extend(rates)
        ratio = (tok_total / chunk_total) if chunk_total else None
        return run_rates, None, {"tokens": tok_total, "chunks": chunk_total, "tokens_per_chunk": ratio,
                                 "completion_rates": comp_rates, "ngen": gen_params()[0]}
    except Exception as e:
        return None, str(e), None
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


def plan_depths():
    """Phase B depths. BENCH_DEPTHS overrides; default is the shallow curve. BENCH_DEPTHS=none
    drops phase B entirely, which a targeted single-cell investigation wants and a release sweep
    never does -- there was previously no way to express "no depth curve" short of editing this."""
    raw = os.environ.get("BENCH_DEPTHS", "").strip()
    if not raw:
        return [512, 2048, 3900]
    if raw.lower() == "none":
        return []
    return [int(d) for d in raw.split(",") if d.strip()]


def plan_engines():
    """Engines to run. BENCH_ENGINES narrows them; default is both.

    For an INTERNAL A/B — the same binary against itself under an env flag — the peer cell measures
    nothing and doubles the sweep. USING THIS FORFEITS THE DRIFT CONTROL, which is the whole reason
    a peer cell sits in a comparison, so it is legitimate ONLY where no cross-engine claim is being
    made. Never use it for a row that quotes a ratio."""
    raw = os.environ.get("BENCH_ENGINES", "").strip()
    if not raw:
        return ["goinfer", "ollama"]
    picked = [e.strip() for e in raw.split(",") if e.strip()]
    known = ("goinfer", "ollama", "llamacpp")
    unknown = [e for e in picked if e not in known]
    if unknown:
        sys.exit(f"BENCH_ENGINES: unknown engine(s) {unknown}; known: {list(known)}")
    return picked


def plan_backends():
    """Phase A backends. BENCH_BACKENDS narrows them; default is all three, so the release sweep's
    backend table is unchanged.

    This exists because phase A is where a targeted re-measure spends its time without learning
    anything: the CPU cell alone runs ~550 s and webgpu adds another, so isolating ONE cuda cell
    used to cost a ~40-minute sweep dominated by cells the question does not touch. That cost is
    what deterred bisecting G26.

        BENCH_BACKENDS=cuda BENCH_DEPTHS=none BENCH_MODELS=phi3-mini python3 scripts/bench_peer.py ...

    NARROWING THE PLAN IS NOT NARROWING THE PROTOCOL. Each cell that does run is measured exactly
    as it would be in a full sweep -- same warmup, same interleaving, same run count. A cell
    measured under this filter is comparable to the same cell from a release sweep; what is NOT
    comparable is a cross-BACKEND claim, since the backends the filter drops were never run."""
    raw = os.environ.get("BENCH_BACKENDS", "").strip()
    if not raw:
        return ["cpu", "cuda", "webgpu"]
    picked = [b.strip() for b in raw.split(",") if b.strip()]
    unknown = [b for b in picked if b not in SERVE]
    if unknown:
        sys.exit(f"BENCH_BACKENDS: unknown backend(s) {unknown}; known: {sorted(SERVE)}")
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
    # A) the backend table, greedy, one depth. goinfer on all three backends; ollama and llamacpp
    #    on the two they actually have (both now -ngl/num_gpu forced, not left to auto-offload --
    #    V-08). webgpu has NO ollama/llamacpp counterpart -- it is scored against the CUDA row and
    #    labelled cross-backend in the writeup, never as a peer cell.
    for mk in plan_models():
        bes, engs = plan_backends(), plan_engines()
        for eng, be in [("goinfer","cpu"), ("ollama","cpu"), ("llamacpp","cpu"),
                        ("goinfer","cuda"), ("ollama","cuda"), ("llamacpp","cuda"),
                        ("goinfer","webgpu"),
                        # Metal (Mac), added 2026-09-04. "ollama"/"metal" and "llamacpp"/"metal"
                        # map to the SAME code path as their "cuda" siblings (ollama_payload only
                        # special-cases backend=="cpu"; the llama-server ngl switch only
                        # special-cases "cpu" too) -- on darwin that default IS each peer's Metal
                        # backend, so no new branch was needed in run_cell, only this pairing and
                        # the "metal" SERVE entry above. Inert unless BENCH_BACKENDS includes
                        # "metal" (plan_backends() default is cpu/cuda/webgpu), so this changes
                        # nothing for the Linux release sweep.
                        ("goinfer","metal"), ("ollama","metal"), ("llamacpp","metal")]:
            if be in bes and eng in engs:
                plan.append(("A", eng, be, mk, 128, "greedy"))
    # B) depth curve, CUDA only, all engines. "cuda" here is no longer just a label (V-08): every
    #    engine's launch now maps it to a real GPU-offload flag (goinfer -backend cuda, ollama's
    #    default un-forced GPU, llamacpp -ngl 99), so a Phase-B cell is genuinely GPU on all three.
    for mk in plan_models():
        for d in plan_depths():
            for eng in plan_engines():
                plan.append(("B", eng, "cuda", mk, d, "greedy"))

    # Phase C is the SAMPLING axis, and it is empty unless asked for. §B5's stale set is "the
    # sampled (temp / temp+top_p) rows"; CONFIGS has carried those definitions all along but no
    # phase ever scheduled them, so those rows had no reproducible path — which is why they could
    # not simply be re-run after the 2026-08-25 re-anchor.
    #
    #     BENCH_CONFIGS=temp0.8_topp0.95,temp0.8_topk40 python3 scripts/bench_peer.py ...
    for cfg in plan_configs():
        for mk in plan_models():
            for eng in plan_engines():
                plan.append(("C", eng, "cuda", mk, 128, cfg))

    print(f"# {len(plan)} cells planned, {len(done)} already done", flush=True)
    for phase, engine, backend, mk, depth, cfg in plan:
        key = (phase, engine, backend, mk, depth, cfg)
        if key in done:
            print(f"# skip (done): {key}", flush=True)
            continue
        gate_cell_idle()
        t0 = time.time()
        machine = machine_state()
        rates, err, counts = run_cell(engine, mk, depth, cfg, backend)
        rec = {"phase": phase, "engine": engine, "backend": backend, "model": mk,
               "depth": depth, "prompt_tokens": prompt_tokens(depth, mk),
               "config": cfg, "sent": CONFIGS[cfg].get(engine, {}),
               "note": CONFIGS[cfg]["note"], "runs": rates, "error": err,
               "machine": machine,
               "secs": round(time.time() - t0, 1)}
        if counts:
            # tokens/chunks is the DIAGNOSTIC for the chunk-counting bug: 1.000 means chunking cost
            # this cell nothing and the old numbers were right here; >1 means the old harness
            # under-counted, and by how much.
            rec["counts"] = counts
            r = counts.get("tokens_per_chunk")
            if r:
                print(f"#   tokens/chunks = {r:.4f}  ({counts['tokens']} tok / {counts['chunks']} chunks)",
                      flush=True)
        if rates:
            rec["mean"] = round(statistics.mean(rates), 1)
            rec["spread"] = round(max(rates) - min(rates), 1)
        out.append(rec)
        print(json.dumps(rec), flush=True)
        with open(outpath, "w") as f:
            json.dump(out, f, indent=1)

if __name__ == "__main__":
    main()
