#!/usr/bin/env python
"""Golden for the ChatML Hermes tool renderer against Qwen2.5-Coder's own template, over conversations with SEVERAL tool results.

    ~/g4venv/bin/python scripts/pin_qwen25_tools.py qwen25-coder.jinja ~/models/qwen2.5-0.5b-instruct testdata/chat_goldens/tools_qwen25_grouped.json

The template (the one inside Qwen2.5-Coder-0.5B-Instruct's GGUF; dump it with goinfer's own reader: a throwaway test that writes
tokenizer.LoadGGUF(path).ChatTemplate()) is stored in the golden. What it pins: that a RUN of consecutive tool results is written as ONE user
turn — `<|im_start|>user`, one `<tool_response>` block per result, one `<|im_end|>` — because the template opens the user turn only when the
previous message was not a tool message. goinfer wrote a user turn per result.
"""
import json, sys, warnings
warnings.filterwarnings("ignore")
from transformers import AutoTokenizer

tmpl_path, tok_dir, out = sys.argv[1:4]
template = open(tmpl_path, encoding="utf-8").read()
tok = AutoTokenizer.from_pretrained(tok_dir)
SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "Get the weather for a city", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
def call(i, city): return {"id": i, "type": "function", "function": {"name": "get_weather", "arguments": {"city": city}}}
S = {"role": "system", "content": SYSTEM}
U = lambda c: {"role": "user", "content": c}
def A(c, calls): return {"role": "assistant", "content": c, "tool_calls": calls}
T = lambda c: {"role": "tool", "content": c}
CASES = {
    "one_result":       [S, U("Weather in Paris?"), A("", [call("a", "Paris")]), T("18C")],
    "two_results":      [S, U("Weather in Paris and Rome?"), A("", [call("a", "Paris"), call("b", "Rome")]), T("18C"), T("24C")],
    "three_results":    [S, U("Three cities?"), A("", [call("a", "Paris"), call("b", "Rome"), call("c", "Oslo")]), T("18C"), T("24C"), T("5C")],
    "results_then_answer_then_more": [S, U("Paris and Rome?"), A("", [call("a", "Paris"), call("b", "Rome")]), T("18C"), T("24C"), {"role": "assistant", "content": "18C and 24C."}, U("And Oslo?")],
    "two_loops":        [S, U("Weather?"), A("", [call("a", "Paris")]), T("18C"), A("", [call("b", "Rome")]), T("24C")],
}
# A case with its own tool list: descriptions with the characters HTML-escaping would touch, an enum, a default, a nested object, key order
# that is not alphabetical, and a tool with no parameters. What it pins is the BYTES of a declaration (tool | tojson), not the conversation.
SPECIAL = [
    {"type": "function", "function": {"name": "say", "description": "Say \"hi\" & <b>don't</b> panic", "parameters": {"type": "object", "properties": {"text": {"type": "string", "description": "It's <the> text & more"}, "mode": {"type": "string", "enum": ["a", "b"], "default": "a"}}, "required": ["text"]}}},
    {"type": "function", "function": {"name": "zeta", "description": "Keys out of alphabetical order", "parameters": {"required": ["z"], "type": "object", "properties": {"z": {"type": "integer"}, "a": {"type": "object", "properties": {"y": {"type": "boolean"}, "b": {"type": "array", "items": {"type": "string"}}}}}}}},
    {"type": "function", "function": {"name": "ping", "description": "No parameters"}},
]
cases = []
for name, msgs in CASES.items():
    p = tok.apply_chat_template(msgs, tools=TOOLS, chat_template=template, add_generation_prompt=True, tokenize=False)
    cases.append({"name": name, "messages": msgs, "prompt": p})
cases.append({"name": "declare_special", "messages": [S, U("hi")], "tools": SPECIAL,
              "prompt": tok.apply_chat_template([S, U("hi")], tools=SPECIAL, chat_template=template, add_generation_prompt=True, tokenize=False)})
json.dump({"checkpoint": "Qwen2.5-Coder-0.5B-Instruct", "chat_template": template, "tools": TOOLS, "cases": cases}, open(out, "w"), indent=1, ensure_ascii=False)
print(len(cases), "cases ->", out)
