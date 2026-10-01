#!/usr/bin/env python
"""Golden for chat.Harmony()'s TOOL rendering: what gpt-oss's own chat template makes of tool declarations and tool-loop conversations.

    ~/g4venv/bin/python scripts/pin_harmony_tools.py gpt-oss.jinja ~/models/gpt-oss-20b-hf testdata/chat_think_goldens/harmony_tools.json

Same inputs as scripts/pin_harmony_history.py (the template comes out of the GGUF; the live date is masked as {DATE}). Each case is a
conversation plus a tool list, rendered by HuggingFace's apply_chat_template(add_generation_prompt=True). What it pins: the TypeScript-like
`namespace functions` block in the developer message (the template's render_typescript_type, over schemas that exercise each of its
branches: strings and enums, number/integer/boolean, arrays, nested objects, optional and defaulted parameters, nullable, oneOf),
and how a call and its result replay — `<|start|>assistant to=functions.NAME<|channel|>commentary json<|message|>ARGS<|call|>` then
`<|start|>functions.NAME to=assistant<|channel|>commentary<|message|>"RESULT"<|end|>` — including that an assistant turn's text or
reasoning beside a call is an ANALYSIS message, kept only while no later assistant turn has answered.
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

def fn(name, description, parameters=None):
    f = {"name": name, "description": description}
    if parameters is not None:
        f["parameters"] = parameters
    return {"type": "function", "function": f}

WEATHER = fn("get_weather", "Weather for a city", {"type": "object", "properties": {"city": {"type": "string", "description": "City name"}}, "required": ["city"]})
SEARCH = fn("search", "Search the web", {"type": "object", "properties": {
    "query": {"type": "string", "description": "What to look for"},
    "topn": {"type": "integer", "default": 10},
    "safe": {"type": "boolean"},
    "lang": {"type": "string", "enum": ["en", "fr", "de"], "default": "en"},
    "tags": {"type": "array", "items": {"type": "string"}},
    "ratio": {"type": "number"},
}, "required": ["query"]})
NESTED = fn("create_event", "Create a calendar event", {"type": "object", "properties": {
    "title": {"type": "string"},
    "when": {"type": "object", "properties": {"date": {"type": "string"}, "hour": {"type": "integer"}}, "required": ["date"]},
    "guests": {"type": "array", "items": {"type": "object", "properties": {"name": {"type": "string"}}}},
    "notes": {"type": "string", "nullable": True},
    "anything": {"oneOf": [{"type": "string"}, {"type": "integer"}]},
    "free": {"type": "object"},
    "loose": {},
}, "required": ["title", "when"]})
NOARGS = fn("ping", "Check the connection")
QUOTED = fn("say", "Say \"hi\" & <b>don't</b> panic", {"type": "object", "properties": {"text": {"type": "string", "description": "It's <the> text & more"}}, "required": ["text"]})

S = lambda c: {"role": "system", "content": c}
U = lambda c: {"role": "user", "content": c}
def A(c="", thinking=None):
    m = {"role": "assistant", "content": c}
    if thinking is not None:
        m["thinking"] = thinking
    return m
def C(name, args, c="", thinking=None):
    m = A(c, thinking)
    m["tool_calls"] = [{"type": "function", "function": {"name": name, "arguments": args}}]
    return m
T = lambda name, c: {"role": "tool", "name": name, "tool_call_id": "call_1", "content": c}

CASES = {
    "declare_one":         ([U("Weather in Paris?")], [WEATHER]),
    "declare_system":      ([S("Be terse."), U("Weather in Paris?")], [WEATHER]),
    "declare_rich":        ([U("Find things")], [SEARCH]),
    "declare_nested":      ([U("Plan it")], [NESTED]),
    "declare_noargs":      ([U("ping?")], [NOARGS]),
    "declare_quoted":      ([U("say hi")], [QUOTED]),
    "declare_several":     ([U("Do stuff")], [WEATHER, SEARCH, NOARGS]),
    "loop_after_call":     ([U("Weather in Paris?"), C("get_weather", {"city": "Paris"}), T("get_weather", "18C sunny")], [WEATHER]),
    "loop_keeps_thinking": ([U("Weather in Paris?"), C("get_weather", {"city": "Paris"}, thinking="need the weather"), T("get_weather", "18C sunny")], [WEATHER]),
    "loop_text_is_analysis": ([U("Weather in Paris?"), C("get_weather", {"city": "Paris"}, c="Let me check."), T("get_weather", "18C sunny")], [WEATHER]),
    "loop_two_calls":      ([U("Weather?"), C("get_weather", {"city": "Paris"}, thinking="first"), T("get_weather", "18C"), C("get_weather", {"city": "Rome"}, thinking="second"), T("get_weather", "24C")], [WEATHER]),
    "finished_loop_drops_analysis": ([U("Weather in Paris?"), C("get_weather", {"city": "Paris"}, thinking="need the weather"), T("get_weather", "18C sunny"), A("It is 18C."), U("And Rome?")], [WEATHER]),
    "args_nested_unicode": ([U("go"), C("create_event", {"title": "Café ☕", "when": {"date": "2026-10-01", "hour": 9}, "guests": [{"name": "Zoë"}, {"name": "Li"}], "notes": None, "ok": True, "n": 1.5}), T("create_event", "done")], [NESTED]),
    "result_special_chars": ([U("go"), C("say", {"text": "x"}), T("say", "He said \"hi\" <b>it's</b> & ok\nnext line")], [QUOTED]),
    "result_json_text":    ([U("go"), C("get_weather", {"city": "Paris"}), T("get_weather", '{"temp": 18, "sky": "sunny"}')], [WEATHER]),
    "tools_with_history":  ([S("Be terse."), U("Hi"), A("Hello!"), U("Weather in Paris?")], [WEATHER]),
}
date = re.compile(r"Current date: \d{4}-\d{2}-\d{2}")
cases = []
for name, (msgs, tools) in CASES.items():
    p = tok.apply_chat_template(msgs, tools=tools, chat_template=template, add_generation_prompt=True, tokenize=False)
    cases.append({"name": name, "messages": msgs, "tools": tools, "prompt": date.sub("Current date: {DATE}", p)})
json.dump({"checkpoint": "gpt-oss-20b", "chat_template": template, "cases": cases}, open(out, "w"), indent=1, ensure_ascii=False)
print(f"{len(cases)} cases -> {out}")
