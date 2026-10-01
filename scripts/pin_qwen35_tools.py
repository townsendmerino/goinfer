#!/usr/bin/env python
"""Golden for chat's Qwen3.5 NATIVE tool rendering: what the Qwen3.5 chat template makes of tool declarations and tool-loop conversations.

    ~/g4venv/bin/python scripts/pin_qwen35_tools.py ~/models/qwen3.5-9b testdata/chat_think_goldens/qwen35_tools.json

HuggingFace's apply_chat_template(add_generation_prompt=True) over conversations that exercise each branch of the template's tool handling:
the `<tools>` block (Python-spaced JSON per tool) and its XML-format instructions, the system text appended after them (trimmed), a call written
as `<tool_call><function=NAME><parameter=K>VALUE</parameter></function></tool_call>` (a string argument raw, a number as Python str(), a bool
as True/False, null as None, a list or dict as JSON), the newline rule between text and a call and between calls, several results grouped in
one user turn, and the think block of a turn after the last query. Each of the loop cases in enable_thinking unset / False / True.
The template is stored in the golden, as in the other chat_think goldens.
"""
import json
import sys
import warnings

warnings.filterwarnings("ignore")
from transformers import AutoTokenizer

tok_dir, out = sys.argv[1:3]
tok = AutoTokenizer.from_pretrained(tok_dir)
template = tok.chat_template

def fn(name, description, parameters=None):
    f = {"name": name}
    if description is not None:
        f["description"] = description
    if parameters is not None:
        f["parameters"] = parameters
    return {"type": "function", "function": f}

WEATHER = fn("get_weather", "Weather for a city", {"type": "object", "properties": {"city": {"type": "string", "description": "City name"}}, "required": ["city"]})
SEARCH = fn("search", "Search the web", {"type": "object", "properties": {
    "query": {"type": "string"}, "max_results": {"type": "integer", "default": 10}, "safe": {"type": "boolean"},
    "lang": {"type": "string", "enum": ["en", "fr"]}, "tags": {"type": "array", "items": {"type": "string"}},
}, "required": ["query"]})
NODESC = fn("ping", None, {"type": "object", "properties": {}})
QUOTED = fn("say", "Say \"hi\" & <b>don't</b> panic", {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]})

U = lambda c: {"role": "user", "content": c}
S = lambda c: {"role": "system", "content": c}
def A(c="", r=None, calls=None):
    m = {"role": "assistant", "content": c}
    if r is not None:
        m["reasoning_content"] = r
    if calls:
        m["tool_calls"] = [{"type": "function", "function": {"name": n, "arguments": a}} for n, a in calls]
    return m
T = lambda c: {"role": "tool", "content": c}

CASES = {
    "declare_one":          ([U("Weather in Paris?")], [WEATHER]),
    "declare_system":       ([S("  Be terse.  "), U("Weather in Paris?")], [WEATHER]),
    "declare_several":      ([U("Do stuff")], [WEATHER, SEARCH, NODESC]),
    "declare_quoted":       ([U("say hi")], [QUOTED]),
    "no_tools_system":      ([S("Be terse."), U("Hi")], None),
    "loop_after_call":      ([U("Weather in Paris?"), A("", None, [("get_weather", {"city": "Paris"})]), T("18C sunny")], [WEATHER]),
    "loop_with_reasoning":  ([U("Weather in Paris?"), A("", "need the weather", [("get_weather", {"city": "Paris"})]), T("18C sunny")], [WEATHER]),
    "loop_text_and_call":   ([U("Weather in Paris?"), A("Let me check.", "need it", [("get_weather", {"city": "Paris"})]), T("18C sunny")], [WEATHER]),
    "loop_parallel":        ([U("Weather in Paris and Rome?"), A("", "two cities", [("get_weather", {"city": "Paris"}), ("get_weather", {"city": "Rome"})]), T("18C"), T("24C")], [WEATHER]),
    "loop_sequential":      ([U("Weather?"), A("", "first", [("get_weather", {"city": "Paris"})]), T("18C"), A("", "second", [("get_weather", {"city": "Rome"})]), T("24C")], [WEATHER]),
    "finished_loop":        ([U("Weather in Paris?"), A("", "need it", [("get_weather", {"city": "Paris"})]), T("18C sunny"), A("It is 18C.", "got it"), U("And Rome?")], [WEATHER]),
    "arg_types":            ([U("go"), A("", None, [("search", {"query": "go 1.24", "max_results": 3, "ratio": 1.5, "safe": True, "off": False, "none": None, "tags": ["a", "b c"], "opts": {"k": [1, 2], "z": "x"}, "multi": "line1\nline2", "quoted": "say \"hi\" <b>it's</b>"})]), T("done")], [SEARCH]),
    "result_whitespace":    ([U("go"), A("", None, [("get_weather", {"city": "Paris"})]), T("  \n18C sunny \n\n")], [WEATHER]),
    "result_json_text":     ([U("go"), A("", None, [("get_weather", {"city": "Paris"})]), T('{"temp": 18, "sky": "sunny"}')], [WEATHER]),
    "tools_with_history":   ([S("Be terse."), U("Hi"), A("Hello!", "greet"), U("Weather in Paris?")], [WEATHER]),
}
cases = []
for name, (msgs, tools) in CASES.items():
    for mode, kw in (("unset", {}), ("false", {"enable_thinking": False}), ("true", {"enable_thinking": True})):
        if mode != "unset" and name not in ("declare_one", "loop_after_call", "loop_with_reasoning", "finished_loop"):
            continue
        kwargs = dict(kw)
        if tools:
            kwargs["tools"] = tools
        p = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, **kwargs)
        cases.append({"name": name, "mode": mode, "messages": msgs, "tools": tools or [], "prompt": p})
json.dump({"checkpoint": "qwen3.5-9b", "chat_template": template, "cases": cases}, open(out, "w"), indent=1, ensure_ascii=False)
print(f"{len(cases)} cases -> {out}")
