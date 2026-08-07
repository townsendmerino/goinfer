#!/usr/bin/env python3
"""Byte-exact tool-calling render goldens from HF apply_chat_template(tools=…),
one file per family, for the chat package's RenderTools tests. Captures the
declaration prompt, a model tool-call emission, and a tool-result turn.

    ~/g4venv/bin/python scripts/gen_tool_goldens.py
"""
import json, os, datetime
from transformers import AutoTokenizer

OUT = os.path.expanduser("~/mycode/goinfer/testdata/chat_goldens")
os.makedirs(OUT, exist_ok=True)
PINNED_DATE, TODAY = "01 Jan 2025", datetime.datetime.now().strftime("%d %b %Y")

FAMILIES = {
    "chatml": "Qwen/Qwen2.5-Coder-0.5B-Instruct",
    "llama3": "unsloth/Llama-3.2-1B-Instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "gemma4": "google/gemma-4-12B-it-qat-q4_0-unquantized",
}

# N-22: pin each repo to the commit SHA the committed goldens were built from, so an upstream
# chat_template edit can't silently change a byte-exact fixture on regeneration. None = main (drift).
REVISIONS = {repo: None for repo in FAMILIES.values()}

TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["location"],
        },
    },
}]

DECLARE = [{"role": "user", "content": "Weather in Paris?"}]
FULL = [
    {"role": "user", "content": "Weather in Paris?"},
    {"role": "assistant", "content": "", "tool_calls": [
        {"id": "abc123def", "type": "function", "function": {"name": "get_weather",
         "arguments": {"location": "Paris", "unit": "celsius"}}}]},
    {"role": "tool", "name": "get_weather", "tool_call_id": "abc123def", "content": "18C, sunny"},
]

for fam, repo in FAMILIES.items():
    rev = REVISIONS.get(repo)
    if rev is None:
        print(f"WARNING {fam} ({repo}): unpinned revision — goldens may drift; set REVISIONS[{repo!r}] to a commit SHA")
    try:
        tok = AutoTokenizer.from_pretrained(repo, revision=rev)
    except Exception as e:
        print(f"SKIP {fam}: {e}")
        continue
    out = {"family": fam, "repo": repo, "revision": rev, "tools": TOOLS, "cases": []}
    for name, conv in [("declare", DECLARE), ("call_result", FULL)]:
        try:
            s = tok.apply_chat_template(conv, tools=TOOLS, tokenize=False, add_generation_prompt=True)
            out["cases"].append({"name": name, "messages": conv, "rendered": s.replace(TODAY, PINNED_DATE)})
        except Exception as e:
            out["cases"].append({"name": name, "messages": conv, "error": str(e)})
    path = f"{OUT}/tools_{fam}.json"
    json.dump(out, open(path, "w"), indent=1, ensure_ascii=False)
    print(f"wrote {path}")
    for c in out["cases"]:
        print(f"  {fam}/{c['name']}: " + (repr(c['rendered']) if 'rendered' in c else "ERR " + c['error'][:60]))
