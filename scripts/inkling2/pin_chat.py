# Usage: python3 scripts/inkling2/pin_chat.py <dir with the release's tokenizer_config.json + chat_template.jinja> chat/testdata/inkling2_chat_goldens.json
# Renders each case with transformers' apply_chat_template over Inkling-2-Mini's own chat_template.jinja.
import json, sys
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(sys.argv[1])
weather = {"type": "function", "function": {"name": "get_weather", "description": "Get the weather <now> & later",
           "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}}, "required": ["city"]}}}
cases = [
    {"name": "user_only", "turns": [{"role": "user", "content": "Hi there."}]},
    {"name": "system_user", "system": "You are terse.", "turns": [{"role": "user", "content": "Explain DNS."}]},
    {"name": "multi_turn", "system": "Be kind.", "turns": [{"role": "user", "content": "One?"}, {"role": "assistant", "content": "Two."}, {"role": "user", "content": "Three?"}]},
    {"name": "reasoning_replay", "turns": [{"role": "user", "content": "2+2?"}, {"role": "assistant", "content": "4", "reasoning": "Add them."}, {"role": "user", "content": "And 3+3?"}]},
    {"name": "effort_none", "effort": "none", "turns": [{"role": "user", "content": "Quick: capital of France?"}]},
    {"name": "effort_low", "effort": "low", "turns": [{"role": "user", "content": "Name a color."}]},
    {"name": "tools_declared", "tools": [weather], "turns": [{"role": "user", "content": "Weather in Paris?"}]},
    {"name": "tool_loop", "tools": [weather], "turns": [
        {"role": "user", "content": "Weather in Paris for 2 days?"},
        {"role": "assistant", "content": "", "reasoning": "Call the tool.", "tool_calls": [{"name": "get_weather", "arguments": {"days": 2, "city": "Paris"}}]},
        {"role": "tool", "tool_name": "get_weather", "content": "{\"temp\": 12}"},
        {"role": "user", "content": "Thanks; and tomorrow?"}]},
]
for c in cases:
    msgs = []
    if c.get("system"):
        msgs.append({"role": "system", "content": c["system"]})
    for t in c["turns"]:
        m = {"role": t["role"], "content": t["content"]}
        if t.get("reasoning"):
            m["reasoning_content"] = t["reasoning"]
        if t.get("tool_calls"):
            m["tool_calls"] = [{"type": "function", "function": {"name": x["name"], "arguments": x["arguments"]}} for x in t["tool_calls"]]
        if t.get("tool_name"):
            m["name"] = t["tool_name"]
        msgs.append(m)
    kw = {}
    if c.get("effort"):
        kw["reasoning_effort"] = c["effort"]
    if c.get("tools"):
        kw["tools"] = c["tools"]
        c["tools"] = [{"name": x["function"]["name"], "description": x["function"]["description"], "parameters": x["function"]["parameters"]} for x in c["tools"]]
    c["want"] = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)
json.dump(cases, open(sys.argv[2], "w"), indent=1, ensure_ascii=False)
print(len(cases))
