#!/usr/bin/env python
"""Golden for the EARLIER Gemma 4 chat template (the one the E2B GGUF carries; the canonical 2026-07-09 template is pinned by
pin_chat_think_history.py).

    ~/g4venv/bin/python scripts/pin_gemma4_old.py ~/models/gemma-4-E2B_q4_0-it.gguf ~/models/gemma-4-E2B-unq testdata/chat_think_goldens/gemma4_old.json

The template text is read from the GGUF's tokenizer.chat_template (the HF export of this checkpoint ships none), rendered by
HuggingFace's apply_chat_template with the E2B tokenizer, in enable_thinking unset / False / True. What it pins: the thinking-off
generation prompt is a bare `<|turn>model\\n` — the canonical template's closed scaffold does not exist here.
"""
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
from gguf import GGUFReader
from transformers import AutoTokenizer

TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "Weather for a city",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
U = lambda c: {"role": "user", "content": c}
S = lambda c: {"role": "system", "content": c}
def A(c, r=None, tc=None):
    m = {"role": "assistant", "content": c}
    if r is not None: m["reasoning_content"] = r
    if tc: m["tool_calls"] = tc
    return m
TC = lambda city: [{"type": "function", "function": {"name": "get_weather", "arguments": {"city": city}}}]
T = lambda c: {"role": "tool", "name": "get_weather", "content": c}
CASES = {
    "single":            [U("Hi")],
    "system":            [S("Be brief."), U("Hi")],
    "history":           [U("Hi"), A("Hello!"), U("And?")],
    "history_reasoning": [U("Hi"), A("Hello!", "plan A"), U("And?")],
    "answer_last":       [U("Hi"), A("Hello!", "plan A")],
    "channel_in_content": [U("Hi"), A("<|channel>thought\nplan A\n<channel|>Hello!"), U("And?")],
}
TOOL_CASES = {
    "declared":  [U("Weather in Paris?")],
    "tool_loop": [U("Weather in Paris?"), A("", "need the weather", tc=TC("Paris")), T("18C sunny")],
    "tool_loop_two":   [U("Weather?"), A("", "first Paris", tc=TC("Paris")), T("18C"), A("", "now Rome", tc=TC("Rome")), T("24C")],
    # a calling turn's reasoning is kept only while no later user turn exists; the same loop followed by a user turn drops it
    # reasoning on a turn that makes no call, after the last user turn, is dropped (the template keeps only a calling turn's)
    "tool_answer_reasoned": [U("Weather in Paris?"), A("", "need the weather", tc=TC("Paris")), T("18C sunny"), A("It is 18C.", "got it")],
    "tool_loop_then_user": [U("Weather in Paris?"), A("", "need the weather", tc=TC("Paris")), T("18C sunny"), A("It is 18C.", "got it"), U("thanks")],
    "tool_done": [U("Weather in Paris?"), A("", "need the weather", tc=TC("Paris")), T("18C sunny"), A("It is 18C."), U("And Rome?")],
}
MODES = {"unset": {}, "false": {"enable_thinking": False}, "true": {"enable_thinking": True}}

def main():
    gguf_path, tok_dir, out = os.path.expanduser(sys.argv[1]), os.path.expanduser(sys.argv[2]), sys.argv[3]
    f = GGUFReader(gguf_path).fields["tokenizer.chat_template"]
    tmpl = bytes(f.parts[f.data[0]]).decode()
    tok = AutoTokenizer.from_pretrained(tok_dir)
    tok.chat_template = tmpl
    cases = []
    for cname, msgs in list(CASES.items()) + list(TOOL_CASES.items()):
        tools = cname in TOOL_CASES
        c = {"name": cname, "tools": tools, "messages": msgs, "prompts": {}}
        for mname, kw in MODES.items():
            c["prompts"][mname] = tok.apply_chat_template(msgs, tools=TOOLS if tools else None, add_generation_prompt=True, tokenize=False, **kw)
        cases.append(c)
    json.dump([{"checkpoint": "gemma-4-E2B-old-template", "chat_template": tmpl, "cases": cases}], open(out, "w"), indent=1)
    print("wrote", out, len(cases), "cases")

if __name__ == "__main__":
    main()
