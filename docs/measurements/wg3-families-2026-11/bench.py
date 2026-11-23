#!/usr/bin/env python3
"""WG3 families' nobara cells (registration.md): scripts/bench_peer.py (decode) or WG0's wg0_ttft.py (TTFT), with this
registration's three models added to their model tables and BENCH_PROMPTS (the calibrated copy) as the prompt source.

  BENCH_PROMPTS=<prompts.json copy> bench.py decode <results.json>          (bench_peer's environment otherwise)
  BENCH_PROMPTS=<prompts.json copy> WG0_REPO=<worktree> bench.py ttft <out.json> --arm head=<serve>:webgpu --arm base=... --models <key>
"""
import json, os, signal, sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("WG0_REPO", os.path.abspath(os.path.join(HERE, "..", "..", "..")))
sys.path.insert(0, os.path.join(REPO, "scripts"))
M = os.path.expanduser("~/models")
MODELS = {
    "glm-ocr": (os.path.join(M, "glm-ocr"), "glm-ocr-none"),
    "ministral3": (os.path.join(M, "ministral3-3b-bf16"), "ministral3-none"),
    "cr7b": (os.path.join(M, "command-r7b"), "cr7b-none"),
}
mode, rest = sys.argv[1], sys.argv[2:]
prompts = json.load(open(os.environ["BENCH_PROMPTS"]))
if mode == "decode":
    sys.argv = [os.path.join(REPO, "scripts", "bench_peer.py")] + rest
    import bench_peer as bp  # noqa: E402
    bp.MODELS.update(MODELS)
    import timing_lock  # noqa: E402
    signal.signal(signal.SIGTERM, bp._term_exits)
    signal.signal(signal.SIGINT, bp._term_exits)
    with timing_lock.hold(f"bench.py decode {' '.join(rest)}", wait_s=float(os.environ.get("BENCH_LOCK_WAIT", "0"))):
        bp.main()
elif mode == "ttft":
    sys.argv = [sys.argv[0]] + rest
    sys.path.insert(0, os.path.join(HERE, "..", "webgpu-baseline-2026-11"))
    import wg0_ttft  # noqa: E402
    wg0_ttft.bpp.MODELS.update(MODELS)
    wg0_ttft.bpp.PROMPTS = prompts
    for k in MODELS:
        wg0_ttft.PROMPT_KEY[k] = k + ":1024"
    _popen = wg0_ttft.subprocess.Popen

    def popen(argv, *a, **kw):  # the base arm runs on the CPU by design, so no -require-backend
        if isinstance(argv, list) and "-model" in argv:
            argv = [x for x in argv if x != "-require-backend"]
        return _popen(argv, *a, **kw)

    wg0_ttft.subprocess.Popen = popen
    _run_cell = wg0_ttft.run_cell

    def run_cell(*a, **kw):  # records the argv the server actually ran
        rec = _run_cell(*a, **kw)
        rec["argv"] = [x for x in rec["argv"] if x != "-require-backend"]
        return rec

    wg0_ttft.run_cell = run_cell
    wg0_ttft.main()
else:
    sys.exit(f"mode {mode!r}: want decode or ttft")
