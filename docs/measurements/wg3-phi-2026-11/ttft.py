#!/usr/bin/env python3
"""WG3-phi's TTFT cell: WG0's wg0_ttft.py with this registration's flags (registration.md): the phi3-mini:1024 prompt,
-quant int8int8, and no -require-backend, since the base arm runs on the CPU by design.

  WG0_REPO=<worktree> ttft.py out.json --arm head=<serve>:webgpu --arm base=<serve>:webgpu --models phi3-mini --quads 2 --n 6
"""
import os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "webgpu-baseline-2026-11"))
import wg0_ttft  # noqa: E402

wg0_ttft.PROMPT_KEY["phi3-mini"] = "phi3-mini:1024"
_popen = wg0_ttft.subprocess.Popen


def popen(argv, *a, **kw):
    if isinstance(argv, list) and argv and "-model" in argv:
        argv = fix_argv(argv)
    return _popen(argv, *a, **kw)


def fix_argv(argv):
    argv = list(argv)
    if "-quant" in argv:
        argv[argv.index("-quant") + 1] = "int8int8"
    return [x for x in argv if x != "-require-backend"]


wg0_ttft.subprocess.Popen = popen
_run_cell = wg0_ttft.run_cell


def run_cell(*a, **kw):  # records the argv the server actually ran
    rec = _run_cell(*a, **kw)
    rec["argv"] = fix_argv(rec["argv"])
    return rec


wg0_ttft.run_cell = run_cell
if __name__ == "__main__":
    wg0_ttft.main()
