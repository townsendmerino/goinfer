#!/usr/bin/env python
"""Golden for chat.Harmony()'s conversation rendering: what gpt-oss's own chat template makes of a multi-turn conversation.

    ~/g4venv/bin/python scripts/pin_harmony_history.py gpt-oss.jinja ~/models/gpt-oss-20b-hf testdata/chat_think_goldens/harmony_history.json

gpt-oss-20b-hf ships no chat_template (its tokenizer_config has none), so the template is the one inside the GGUF; dump it with
goinfer's own reader (a throwaway test in ./chat: os.WriteFile(out, []byte(tokenizer.LoadGGUF(path).ChatTemplate()), 0o644)) and
pass the file. The template text is stored in the golden, as in the other chat_think goldens.

HuggingFace's apply_chat_template(add_generation_prompt=True) renders each conversation; the live date (strftime_now) is masked
as {DATE} so the golden does not expire. What it pins: the developer block, and — the defect this was written for — that an
assistant turn BEFORE the final one is rendered on the `final` channel (<|start|>assistant<|channel|>final<|message|>…<|end|>),
with its reasoning dropped however it was supplied. And that `reasoning_effort` (low | medium | high) is written on the `Reasoning:` line.
"""
import json
import re
import sys
import warnings

warnings.filterwarnings("ignore")
from transformers import AutoTokenizer

tmpl_path, tok_dir, out = sys.argv[1:4]
template = open(tmpl_path, encoding="utf-8").read()
tok = AutoTokenizer.from_pretrained(tok_dir)

S = lambda c: {"role": "system", "content": c}
U = lambda c: {"role": "user", "content": c}
def A(c, thinking=None):
    m = {"role": "assistant", "content": c}
    if thinking is not None:
        m["thinking"] = thinking
    return m

CASES = {
    "user_only":            [U("Hi")],
    "system_user":          [S("Answer in French."), U("Hi")],
    "one_exchange":         [U("Hi"), A("Hello!"), U("And you?")],
    "system_one_exchange":  [S("Be terse."), U("Hi"), A("Hello!"), U("And you?")],
    "two_exchanges":        [U("Hi"), A("Hello!"), U("Name a colour."), A("Blue."), U("Another?")],
    "reasoning_dropped":    [U("Hi"), A("Hello!", thinking="plan A"), U("And you?")],
    "content_whitespace":   [U("Hi"), A("  Hello!\n"), U("And you?")],
}
# Template kwargs: gpt-oss's template takes `reasoning_effort` (low | medium | high; "medium" when absent) for its `Reasoning:` line.
KWARG_CASES = {
    "effort_low":          ([U("Hi")], {"reasoning_effort": "low"}),
    "effort_high":         ([U("Hi")], {"reasoning_effort": "high"}),
    "effort_high_history": ([S("Be terse."), U("Hi"), A("Hello!"), U("And you?")], {"reasoning_effort": "high"}),
}
date = re.compile(r"Current date: \d{4}-\d{2}-\d{2}")
cases = []
for name, msgs in CASES.items():
    p = tok.apply_chat_template(msgs, chat_template=template, add_generation_prompt=True, tokenize=False)
    cases.append({"name": name, "messages": msgs, "prompt": date.sub("Current date: {DATE}", p)})
for name, (msgs, kwargs) in KWARG_CASES.items():
    p = tok.apply_chat_template(msgs, chat_template=template, add_generation_prompt=True, tokenize=False, **kwargs)
    cases.append({"name": name, "messages": msgs, "kwargs": kwargs, "prompt": date.sub("Current date: {DATE}", p)})
json.dump({"checkpoint": "gpt-oss-20b", "chat_template": template, "cases": cases}, open(out, "w"), indent=1, ensure_ascii=False)
print(f"{len(cases)} cases -> {out}")
for c in cases[:1] + cases[2:3]:
    print(repr(c["prompt"][-160:]))
