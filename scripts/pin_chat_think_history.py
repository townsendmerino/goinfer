#!/usr/bin/env python
"""Golden for chat/reasoning.go's HISTORY rule: how each real chat template re-renders an assistant turn's reasoning.

    ~/g4venv/bin/python scripts/pin_chat_think_history.py ~/models testdata/chat_think_goldens/think_history.json

For each checkpoint, HuggingFace's apply_chat_template(add_generation_prompt=True) over conversations that exercise the rule:
an assistant turn BEFORE the last user query (reasoning stripped), one AFTER it (the in-progress tool loop: reasoning kept),
with and without reasoning, reasoning given as `reasoning_content` / `reasoning` or left as <think>..</think> inside the
content, whitespace around it, and tool loops. Each in enable_thinking unset / False / True (the history rule must not depend
on it). Qwen3.5 renders tool calls as XML where goinfer renders Hermes JSON, so its tool-loop cases are compared only up to the
first tool call; Qwen3 and Gemma 4 tool loops are byte-exact.
"""
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
from transformers import AutoTokenizer

CKPTS = ["qwen3-4b", "qwen3.5-0.8b", "qwen3.5-9b", "gemma-4-26b-a4b-it"]
TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "Weather for a city",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
U = lambda c: {"role": "user", "content": c}
def A(c, r=None, key="reasoning_content", tc=None):
    m = {"role": "assistant", "content": c}
    if r is not None: m[key] = r
    if tc: m["tool_calls"] = tc
    return m
TC = lambda city: [{"type": "function", "function": {"name": "get_weather", "arguments": {"city": city}}}]
T = lambda c: {"role": "tool", "name": "get_weather", "content": c}
def TCID(name, cid, **args):
    return [{"type": "function", "id": cid, "function": {"name": name, "arguments": args}}]
TID = lambda cid, c: {"role": "tool", "tool_call_id": cid, "content": c}
TAGGED = "<think>\nplan A\n</think>\n\nHello!"
GEMMA_CH = "<|channel>thought\nplan A\n<channel|>Hello!"
CASES = {
    "old_turn":                [U("Hi"), A("Hello!", "plan A"), U("And?")],
    "after_query":             [U("Hi"), A("Hello!", "plan A")],
    "after_query_noreason":    [U("Hi"), A("Hello!")],
    "two_after":               [U("Hi"), A("C1", "plan 1"), A("C2", "plan 2")],
    "two_after_first_noreason": [U("Hi"), A("C1"), A("C2", "plan 2")],
    "tags_old":                [U("Hi"), A(TAGGED), U("And?")],
    "tags_new":                [U("Hi"), A(TAGGED)],
    "whitespace_new":          [U("Hi"), A("\n\nHello!  ", "  plan A  \n")],
    "reasoning_key_new":       [U("Hi"), A("Hello!", "plan A", key="reasoning")],
    "gemma_channel_old":       [U("Hi"), A(GEMMA_CH), U("And?")],
    "gemma_channel_new":       [U("Hi"), A(GEMMA_CH)],
}
TOOL_CASES = {
    "tool_loop":      [U("Weather in Paris?"), A("", "need the weather", tc=TC("Paris")), T("18C sunny")],
    "tool_loop_text": [U("Weather in Paris?"), A("Checking.", "need the weather", tc=TC("Paris")), T("18C sunny")],
    "tool_loop_two":  [U("Weather?"), A("", "first Paris", tc=TC("Paris")), T("18C"), A("", "now Rome", tc=TC("Rome")), T("24C")],
    "tool_loop_old":  [U("Weather in Paris?"), A("", "need the weather", tc=TC("Paris")), T("18C"), A("It is 18C."), U("And Rome?")],
    # Assistant TEXT beside a call: where the template writes it, and whether it closes the turn (Gemma 4's canonical template writes it
    # after the results and closes the turn; goinfer's own rendering writes it before the call). Each also without reasoning.
    "text_then_user":     [U("Weather in Paris?"), A("Checking.", "need the weather", tc=TC("Paris")), T("18C sunny"), U("thanks")],
    "text_then_answer":   [U("Weather in Paris?"), A("Checking.", "need the weather", tc=TC("Paris")), T("18C sunny"), A("It is 18C.", "got it")],
    "text_parallel":      [U("Weather in Paris and Rome?"), A("Checking both.", "two cities", tc=TC("Paris") + TC("Rome")), T("18C"), T("24C")],
    "text_noreason":      [U("Weather in Paris?"), A("Checking.", None, tc=TC("Paris")), T("18C sunny")],
    "call_pending":       [U("Weather in Paris?"), A("", "need the weather", tc=TC("Paris"))],
    # A call nobody answered, then another assistant message; and results that name no function, only the call they answer, out of order.
    "call_unanswered_then_assistant": [U("Weather in Paris?"), A("", "need it", tc=TC("Paris")), A("Never mind.")],
    "ids_resolve_names":  [U("Weather and time?"), A("", "both", tc=TCID("get_weather", "c1", city="Paris") + TCID("get_time", "c2", zone="CET")), TID("c2", "21:30"), TID("c1", "18C")],
    "text_sequential":    [U("Weather?"), A("First.", "one", tc=TC("Paris")), T("18C"), A("Second.", "two", tc=TC("Rome")), T("24C")],
}
MODES = {"unset": {}, "false": {"enable_thinking": False}, "true": {"enable_thinking": True}}

def main():
    root, out = os.path.expanduser(sys.argv[1]), sys.argv[2]
    res = []
    for name in CKPTS:
        tok = AutoTokenizer.from_pretrained(os.path.join(root, name))
        entry = {"checkpoint": name, "chat_template": tok.chat_template, "cases": []}
        for cname, msgs in list(CASES.items()) + list(TOOL_CASES.items()):
            tools = cname in TOOL_CASES
            if cname.startswith("gemma_channel") and not name.startswith("gemma-4"):
                continue  # Qwen's template has no notion of Gemma's channel markers
            if cname == "reasoning_key_new" and not name.startswith("gemma-4"):
                continue  # only Gemma's template reads `reasoning`; Qwen's reads reasoning_content alone
            c = {"name": cname, "tools": tools, "messages": msgs, "prompts": {}}
            for mname, kw in MODES.items():
                try:
                    c["prompts"][mname] = tok.apply_chat_template(msgs, tools=TOOLS if tools else None, add_generation_prompt=True, tokenize=False, **kw)
                except Exception as e:
                    c["prompts"][mname] = "ERR " + str(e)[:80]
            entry["cases"].append(c)
        res.append(entry)
    json.dump(res, open(out, "w"), indent=1)
    errs = sum(1 for e in res for c in e["cases"] for p in c["prompts"].values() if p.startswith("ERR "))
    print("wrote", out, sum(len(e["cases"]) for e in res), "cases;", errs, "template errors")

if __name__ == "__main__":
    main()
