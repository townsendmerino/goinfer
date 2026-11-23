#!/usr/bin/env python3
"""TTFT A B B A for the int4 prefill GEMM's slab prefetch (README beside this file): wg0_ttft.py's cells, request helpers and void
rule at the 1.5B's 3,900-token prompt, two serve builds as the two arms.

  ttft_ab.py out.json --arm base=<serve>:metal --arm prefetch=<serve>:metal --models 1.5B --quads 5 --n 6 --ctx 4096
"""
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "webgpu-baseline-2026-11"))
import wg0_ttft  # noqa: E402

wg0_ttft.PROMPT_KEY = {"1.5B": "1.5B:3900"}

if __name__ == "__main__":
    wg0_ttft.main()
