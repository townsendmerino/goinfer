#!/usr/bin/env python3
"""WG3-phi's TTFT cell: WG0's wg0_ttft.py with this registration's flags (registration.md): the phi3-mini:1024 prompt,
-quant int8int8, and no -require-backend on the base arm (WG3_BASE_SERVE), which runs on the CPU by design.

  WG0_REPO=<worktree> ttft.py out.json --arm head=<serve>:webgpu --arm base=<serve>:webgpu --models phi3-mini --quads 2 --n 6
"""
import os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "webgpu-baseline-2026-11"))
import wg0_ttft  # noqa: E402

wg0_ttft.PROMPT_KEY["phi3-mini"] = "phi3-mini:1024"
wg0_ttft.bpp.MODELS.setdefault("phi3-mini", (os.path.expanduser("~/models/phi3-mini-4k-gguf/Phi-3-mini-4k-instruct-q4.gguf"), "p3m-local"))
_popen = wg0_ttft.subprocess.Popen


BASE = os.environ["WG3_BASE_SERVE"]  # the base arm's binary: it runs on the CPU by design, the only arm without -require-backend


def popen(argv, *a, **kw):
    if isinstance(argv, list) and argv and "-model" in argv:
        argv = fix_argv(argv)
    return _popen(argv, *a, **kw)


def fix_argv(argv):
    argv = list(argv)
    if "-quant" in argv:
        argv[argv.index("-quant") + 1] = "int8int8"
    if argv[0] == BASE:
        argv = [x for x in argv if x != "-require-backend"]
    return argv


wg0_ttft.subprocess.Popen = popen
_run_cell = wg0_ttft.run_cell


def run_cell(*a, **kw):  # records the argv the server actually ran
    rec = _run_cell(*a, **kw)
    rec["argv"] = fix_argv(rec["argv"])
    return rec


wg0_ttft.run_cell = run_cell
if __name__ == "__main__":
    wg0_ttft.main()
