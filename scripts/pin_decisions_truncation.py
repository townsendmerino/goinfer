#!/usr/bin/env python3
"""jev_core's state truncation, pinned (D4 follow-up, docs/tasks/task-constrained-confidence.md).

    JEV_TOKENIZER=~/models/JEV-9B python3 scripts/pin_decisions_truncation.py  ->  testdata/decisions/truncation_golden.json

A state over MAX_STATE_TOKENS (1024) tokens is cut to its first 60% and last 40% of tokens and decoded back to text
(skip_special_tokens=True) before the bare-v1 prompt is rendered: truncate_state and build_decision_prompt, imported
from pin_decisions_d0.py so this is the same port D0 used. None of the pinned corpus's 26,824 states is that long (the
longest is 541 tokens), so the corpus cannot test it; these states are built to. They put multi-byte characters (CJK,
emoji, accented Latin) where the cuts land, so HF's byte-level decode must replace a split character with U+FFFD, and
one puts bytes on both sides of the seam. Each case records the state, the truncated state, and the whole prompt's
token ids.
"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("JEV_MODEL_PATH", os.path.expanduser(os.environ.get("JEV_TOKENIZER", "~/models/JEV-9B")))
import transformers  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from pin_decisions_d0 import MAX_STATE_TOKENS, build_decision_prompt, truncate_state  # noqa: E402

TOK_DIR = os.environ["JEV_MODEL_PATH"]
OUT = os.path.join(HERE, "..", "testdata", "decisions", "truncation_golden.json")
tok = AutoTokenizer.from_pretrained(TOK_DIR)

def ntok(s):
    return len(tok(s, add_special_tokens=False)["input_ids"])

def grow(unit, target):
    """unit repeated until the text is at least target tokens."""
    s = ""
    while ntok(s) < target:
        s += unit
    return s

def exactly(unit, n):
    """the longest prefix of repeated unit with at most n tokens, then padded with ASCII to exactly n tokens."""
    s = grow(unit, n + 50)
    lo, hi = 0, len(s)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if ntok(s[:mid]) <= n:
            lo = mid
        else:
            hi = mid - 1
    s = s[:lo]
    while ntok(s) < n:
        s += "x"
    return s

states = {
    "ascii-json": grow('{"order": 1043, "status": "pending", "items": ["widget", "gadget"], "note": "call before noon"} ', 1600),
    "cjk": grow("订单已发货，预计三天内送达。客户要求更改地址。", 2000),
    "emoji": grow("status 🚚 shipped 📦 then 🏠 delivered ✅; ", 1800),
    "accents": grow("Café crème brûlée à la façon de São Paulo, naïve résumé; ", 1700),
    "rare-cjk": grow("𠀀𪚥𰻞 龘 䨻 𩙰 ", 1900),  # rare ideographs tokenize to raw bytes, so the cuts split them
    "mixed-seam": grow("abc 你好 😀 déjà ", 1100) + grow("终 末 🎉 fin ", 900),
    "exactly-1024": exactly("word ", MAX_STATE_TOKENS),
    "exactly-1025": exactly("word ", MAX_STATE_TOKENS + 1),
}
cases = []
for name, state in states.items():
    kept, truncated, n_state = truncate_state(tok, state)
    prompt, _ = build_decision_prompt("noul", kept, "Is the order late?", ["false", "true"])
    cases.append({"name": name, "state": state, "state_tokens": n_state, "truncated": truncated, "kept": kept,
                  "prompt_ids": tok(prompt, add_special_tokens=False)["input_ids"], "has_replacement": "�" in kept})
    print(f"{name:13s} state {n_state:5d} tokens, truncated {truncated}, kept {ntok(kept):5d} tokens, U+FFFD in kept: {'�' in kept}")
json.dump({"max_state_tokens": MAX_STATE_TOKENS, "tokenizer": "autotrust/JEV-9B (Qwen3.5)", "transformers": transformers.__version__,
           "cases": cases}, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("wrote", OUT)
